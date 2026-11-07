# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34: Differential Evolution with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_2d(self, pts, ref):
        """2D hypervolume. pts: (m,2) array, all < ref. ref: (2,) array."""
        m = len(pts)
        if m == 0:
            return 0.0
        if m == 1:
            return float((ref[0] - pts[0, 0]) * (ref[1] - pts[0, 1]))
        idx = np.argsort(pts[:, 0])
        pts = pts[idx]
        x_next = np.empty(m)
        x_next[:-1] = pts[1:, 0]
        x_next[-1] = ref[0]
        x_cur = pts[:, 0]
        y_min = np.minimum.accumulate(pts[:, 1])
        widths = x_next - x_cur
        heights = ref[1] - y_min
        hv = np.sum(widths * heights)
        return float(hv)

    def _hv_3d(self, pts, ref):
        """3D hypervolume. pts: (n,3) array, all < ref. ref: (3,) array."""
        n = len(pts)
        if n == 0:
            return 0.0
        if n == 1:
            return float(np.prod(ref - pts[0]))
        idx = np.argsort(pts[:, 0])
        pts = pts[idx]
        x_next = np.empty(n)
        x_next[:-1] = pts[1:, 0]
        x_next[-1] = ref[0]
        x_cur = pts[:, 0]
        widths = x_next - x_cur
        hvs = np.zeros(n)
        for i in range(n):
            yz = pts[:i + 1, 1:3]
            hvs[i] = self._hv_2d(yz, ref[1:3])
        hv = np.sum(widths * hvs)
        return float(hv)

    def _is_pareto(self, F):
        """Return boolean mask of Pareto-optimal points in F (n x 3)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        le = np.all(F[:, np.newaxis, :] <= F[np.newaxis, :, :], axis=2)
        neq = np.any(F[:, np.newaxis, :] < F[np.newaxis, :, :], axis=2)
        dominates = le & neq
        dominated = np.any(dominates, axis=0)
        pareto = ~dominated
        return pareto

    def _dominates(self, F1, F2):
        """Check if F1 dominates F2 (both 1D arrays)."""
        return np.all(F1 <= F2) and np.any(F1 < F2)

    def _is_dominated(self, F_new, F_archive):
        """Check if F_new is dominated by any point in F_archive."""
        if len(F_archive) == 0:
            return False
        le = np.all(F_archive <= F_new, axis=1)
        neq = np.any(F_archive < F_new, axis=1)
        return np.any(le & neq)

    def _remove_dominated(self, F_new, X_archive, F_archive):
        """Remove points in archive dominated by F_new. Returns new archives."""
        if len(F_archive) == 0:
            return X_archive, F_archive
        le = np.all(F_new[None, :] <= F_archive, axis=2)
        neq = np.any(F_new[None, :] < F_archive, axis=2)
        dominated_mask = np.any(le & neq, axis=0)
        if not np.any(dominated_mask):
            return X_archive, F_archive
        keep = ~dominated_mask
        return X_archive[keep], F_archive[keep]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([1705.0, 12.0, 0.3])

        # Population size for DE
        pop_size = 20
        n_generations = budget // pop_size

        # Initialize population with LHS + random
        X = np.zeros((pop_size, dim))
        n_lhs = min(pop_size, 15)
        n_rand = pop_size - n_lhs

        # LHS initialization
        for d in range(dim):
            perm = np.random.permutation(n_lhs)
            X[:n_lhs, d] = lo[d] + (hi[d] - lo[d]) * (perm + 0.5) / n_lhs

        # Random initialization
        X[n_lhs:] = lo + np.random.uniform(size=(n_rand, dim)) * (hi - lo)
        X = np.clip(X, lo, hi)

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Build initial Pareto archive
        valid = np.all(F < ref, axis=1)
        if np.any(valid):
            pF = F[valid]
            pX = X[valid]
            p_mask = self._is_pareto(pF)
            pareto_X = pX[p_mask].copy()
            pareto_F = pF[p_mask].copy()
        else:
            pareto_X = np.empty((0, dim))
            pareto_F = np.empty((0, n_obj))

        # Track best HV
        best_hv = 0.0
        if len(pareto_F) > 0:
            best_hv = self._hv_3d(pareto_F, ref)

        # DE parameters
        F_mut = 0.8  # Mutation scale
        CR = 0.9     # Crossover rate

        # Adaptive parameters
        gen = 0
        while used < budget:
            if time.time() > deadline:
                break

            progress = used / budget

            # Adaptive mutation factor: decrease over time for convergence
            F_adaptive = 0.8 * (1.0 - 0.5 * progress) + 0.2
            CR_adaptive = 0.9 - 0.3 * progress

            # Generate offspring for each individual
            for i in range(pop_size):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Differential Evolution: v = x_i + F * (x_r1 - x_r2)
                # Select r1, r2, r3 distinct from i
                if pop_size >= 3:
                    candidates = list(range(pop_size))
                    candidates.remove(i)
                    r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)
                else:
                    r1, r2, r3 = 0, 1, 2

                # Mutation
                v = X[i] + F_adaptive * (X[r1] - X[r2])

                # Crossover (binomial)
                u = np.random.rand(dim)
                j_rand = np.random.randint(dim)
                trial = np.where(u < CR_adaptive, v, X[i])
                trial[j_rand] = v[j_rand]  # Ensure at least one gene from mutant

                # Clip to bounds
                trial = np.clip(trial, lo, hi)

                # Evaluate
                trial_F = problem(trial)
                used += 1

                # Update population
                if np.all(trial_F <= F[i]):
                    X[i] = trial
                    F[i] = trial_F

                # Update Pareto archive
                if np.all(trial_F < ref):
                    if not self._is_dominated(trial_F, pareto_F):
                        pareto_X = np.vstack([pareto_X, [trial]])
                        pareto_F = np.vstack([pareto_F, [trial_F]])
                        pareto_X, pareto_F = self._remove_dominated(trial_F[None, :], pareto_X, pareto_F)

                        # Prune if archive too large
                        if len(pareto_F) > 50:
                            p_mask = self._is_pareto(pareto_F)
                            pareto_X = pareto_X[p_mask]
                            pareto_F = pareto_F[p_mask]

            # Update best HV
            if len(pareto_F) > 0:
                current_hv = self._hv_3d(pareto_F, ref)
                if current_hv > best_hv:
                    best_hv = current_hv

            gen += 1

        # Final phase: local search on top Pareto points with remaining budget
        if used < budget and time.time() < deadline - 5 and len(pareto_F) > 0:
            # Get top 5 Pareto points
            n_top = min(5, len(pareto_F))
            top_idx = np.random.choice(len(pareto_F), size=n_top, replace=False)
            top_X = pareto_X[top_idx]

            polish_budget = min(budget - used, 20)
            for k in range(polish_budget):
                if time.time() > deadline - 2:
                    break
                parent = top_X[k % len(top_X)]
                step = 0.03 + 0.05 * np.random.uniform()
                child = parent + step * (hi - lo) * np.random.standard_normal(dim)
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1

                if np.all(fc < ref):
                    if not self._is_dominated(fc, pareto_F):
                        pareto_X = np.vstack([pareto_X, [child]])
                        pareto_F = np.vstack([pareto_F, [fc]])
                        pareto_X, pareto_F = self._remove_dominated(fc[None, :], pareto_X, pareto_F)

        return X

# EVOLVE-BLOCK-END
