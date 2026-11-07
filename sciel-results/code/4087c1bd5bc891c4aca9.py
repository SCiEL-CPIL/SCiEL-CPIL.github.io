# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer for RE34: HV-guided local search with LHS init and efficient polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_2d(self, pts, ref):
        m = len(pts)
        if m == 0:
            return 0.0
        if m == 1:
            return float((ref[0] - pts[0, 0]) * (ref[1] - pts[0, 1]))
        idx = np.argsort(pts[:, 0])
        pts = pts[idx]
        hv = 0.0
        for i in range(m):
            x_cur = pts[i, 0]
            x_next = pts[i + 1, 0] if i + 1 < m else ref[0]
            y_min = np.min(pts[:i + 1, 1])
            hv += (x_next - x_cur) * (ref[1] - y_min)
        return hv

    def _hv_3d(self, pts, ref):
        n = len(pts)
        if n == 0:
            return 0.0
        if n == 1:
            return float(np.prod(ref - pts[0]))
        idx = np.argsort(pts[:, 0])
        pts = pts[idx]
        hv = 0.0
        for i in range(n):
            x_cur = pts[i, 0]
            x_next = pts[i + 1, 0] if i + 1 < n else ref[0]
            yz = pts[:i + 1, 1:3]
            hv += (x_next - x_cur) * self._hv_2d(yz, ref[1:3])
        return hv

    def _is_pareto(self, F):
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        # Vectorized dominance check
        pareto = np.ones(n, dtype=bool)
        for i in range(n):
            if not pareto[i]:
                continue
            others = np.delete(F, i, axis=0)
            if len(others) == 0:
                continue
            dominated = np.any(np.all(others <= F[i], axis=1) & np.any(others < F[i], axis=1))
            if dominated:
                pareto[i] = False
        return pareto

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([1705.0, 12.0, 0.3])

        rng = np.random

        # --- Phase 1: Initial sampling with LHS + random ---
        n_lhs = 30
        n_rand = 15
        n_init = min(n_lhs + n_rand, budget)

        X_list = []
        lhs_X = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = np.random.permutation(n_lhs)
            lhs_X[:, d] = lo[d] + (hi[d] - lo[d]) * (perm + np.random.uniform(size=n_lhs)) / n_lhs
        lhs_X = np.clip(lhs_X, lo, hi)
        X_list.append(lhs_X)

        rand_X = lo + np.random.uniform(size=(n_rand, dim)) * (hi - lo)
        rand_X = np.clip(rand_X, lo, hi)
        X_list.append(rand_X)

        X_init = np.vstack(X_list)
        F_init = np.array([problem(x) for x in X_init])
        used = len(X_init)

        all_X = X_init.copy()
        all_F = F_init.copy()

        # Build initial Pareto archive
        valid = np.all(F_init < ref, axis=1)
        if np.any(valid):
            pF = F_init[valid]
            pX = X_init[valid]
            p_mask = self._is_pareto(pF)
            pareto_X = pX[p_mask].copy()
            pareto_F = pF[p_mask].copy()
        else:
            pareto_X = np.empty((0, dim))
            pareto_F = np.empty((0, n_obj))

        # --- Phase 2: Iterative improvement with adaptive mutation ---
        max_archive = 30

        while used < budget:
            if time.time() > deadline:
                break

            progress = used / budget

            # Parent selection: weighted by HV contribution proxy (distance to reference)
            if len(pareto_F) > 0:
                # Use distance-based proxy for speed
                dists = np.linalg.norm(ref - pareto_F, axis=1)
                # Lower distance = better, so invert for weighting
                weights = 1.0 / (dists + 1e-6)
                # Add some exploration: temperature
                temp = 1.5 * (1.0 - progress) + 0.3
                weights = np.power(weights, 1.0 / temp)
                probs = weights / weights.sum()
                parent_idx = np.random.choice(len(pareto_F), p=probs)
                parent = pareto_X[parent_idx]
            else:
                parent = lo + np.random.uniform(size=dim) * (hi - lo)

            # Adaptive mutation
            if progress < 0.3:
                # Exploration phase: larger steps
                base_step = 0.8
                use_coord = np.random.uniform() < 0.2
            elif progress < 0.7:
                # Transition: medium steps
                base_step = 0.4
                use_coord = np.random.uniform() < 0.5
            else:
                # Exploitation: small steps
                base_step = 0.15
                use_coord = np.random.uniform() < 0.8

            step = base_step * (0.5 + np.random.uniform())

            if not use_coord:
                child = parent + step * (hi - lo) * np.random.standard_normal(dim)
            else:
                child = parent.copy()
                n_perturb = np.random.randint(1, 3)
                coords = np.random.choice(dim, size=n_perturb, replace=False)
                for c in coords:
                    child[c] = parent[c] + step * (hi[c] - lo[c]) * np.random.standard_normal()
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            all_X = np.vstack([all_X, [child]])
            all_F = np.vstack([all_F, [fc]])

            # Update Pareto archive
            if np.all(fc < ref):
                dominated = False
                for pf in pareto_F:
                    if np.all(pf <= fc) and np.any(pf < fc):
                        dominated = True
                        break

                if not dominated:
                    pareto_X = np.vstack([pareto_X, [child]])
                    pareto_F = np.vstack([pareto_F, [fc]])
                    # Remove points dominated by new point
                    keep = np.ones(len(pareto_F), dtype=bool)
                    for i in range(len(pareto_F) - 1):
                        if np.all(pareto_F[i] >= fc) and np.any(pareto_F[i] > fc):
                            keep[i] = False
                    pareto_X = pareto_X[keep]
                    pareto_F = pareto_F[keep]

                    # Prune archive if too large
                    if len(pareto_F) > max_archive:
                        p_mask = self._is_pareto(pareto_F)
                        pareto_X = pareto_X[p_mask]
                        pareto_F = pareto_F[p_mask]
                        if len(pareto_F) > max_archive:
                            # Keep points closest to reference
                            dists = np.linalg.norm(ref - pareto_F, axis=1)
                            top_idx = np.argsort(dists)[:max_archive]
                            pareto_X = pareto_X[top_idx]
                            pareto_F = pareto_F[top_idx]

        # --- Phase 3: Coordinate descent polish on top points ---
        if used < budget and len(pareto_F) > 0:
            # Sort by distance to reference (closer = better)
            dists = np.linalg.norm(ref - pareto_F, axis=1)
            top_k = min(4, len(pareto_F))
            top_idx = np.argsort(dists)[:top_k]

            for pi in top_idx:
                if used >= budget or time.time() > deadline - 10:
                    break
                x_cur = pareto_X[pi].copy()
                f_cur = problem(x_cur)
                used += 1

                # Coordinate descent: try small steps in each direction
                for coord in range(dim):
                    if used >= budget or time.time() > deadline - 5:
                        break
                    for direction in [-1, 1]:
                        if used >= budget:
                            break
                        x_try = x_cur.copy()
                        step_size = 0.1 * (hi[coord] - lo[coord])
                        x_try[coord] += direction * step_size
                        x_try = np.clip(x_try, lo, hi)

                        f_try = problem(x_try)
                        used += 1

                        # Accept if it improves any objective without worsening others
                        if np.all(f_try <= f_cur) and np.any(f_try < f_cur):
                            x_cur = x_try
                            f_cur = f_try
                        elif np.all(f_try < ref):
                            # Update archive with this point
                            all_X = np.vstack([all_X, [x_try]])
                            all_F = np.vstack([all_F, [f_try]])

                            dominated = False
                            for pf in pareto_F:
                                if np.all(pf <= f_try) and np.any(pf < f_try):
                                    dominated = True
                                    break

                            if not dominated:
                                pareto_X = np.vstack([pareto_X, [x_try]])
                                pareto_F = np.vstack([pareto_F, [f_try]])
                                keep = np.ones(len(pareto_F), dtype=bool)
                                for i in range(len(pareto_F) - 1):
                                    if np.all(pareto_F[i] >= f_try) and np.any(pareto_F[i] > f_try):
                                        keep[i] = False
                                pareto_X = pareto_X[keep]
                                pareto_F = pareto_F[keep]

                                if len(pareto_F) > max_archive:
                                    p_mask = self._is_pareto(pareto_F)
                                    pareto_X = pareto_X[p_mask]
                                    pareto_F = pareto_F[p_mask]
                                    if len(pareto_F) > max_archive:
                                        dists2 = np.linalg.norm(ref - pareto_F, axis=1)
                                        top_idx2 = np.argsort(dists2)[:max_archive]
                                        pareto_X = pareto_X[top_idx2]
                                        pareto_F = pareto_F[top_idx2]

        return all_X

# EVOLVE-BLOCK-END
