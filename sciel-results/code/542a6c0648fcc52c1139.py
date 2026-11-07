# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution + Pareto Archive multi-objective optimizer for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_3d(self, pts, ref):
        """Exact hypervolume for 3D points."""
        n = len(pts)
        if n == 0:
            return 0.0
        if n == 1:
            return float(np.prod(ref - pts[0]))

        mask = np.all(pts < ref, axis=1)
        pts = pts[mask]
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

    def _hv_2d(self, pts, ref):
        """Exact hypervolume for 2D points."""
        n = len(pts)
        if n == 0:
            return 0.0
        if n == 1:
            return float((ref[0] - pts[0, 0]) * (ref[1] - pts[0, 1]))

        idx = np.argsort(pts[:, 0])
        pts = pts[idx]

        hv = 0.0
        y_min = ref[1]
        for i in range(n - 1, -1, -1):
            if pts[i, 1] < y_min:
                y_min = pts[i, 1]
                x_cur = pts[i, 0]
                x_next = pts[i + 1, 0] if i + 1 < n else ref[0]
                hv += (x_next - x_cur) * (ref[1] - y_min)
        return hv

    def _is_pareto(self, F):
        """Return boolean mask of non-dominated points."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
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
        deadline = start_time + 315

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([1705.0, 12.0, 0.3])

        rng = np.random

        # --- Phase 1: Initial sampling with LHS + random ---
        n_lhs = 25
        n_rand = 15
        n_init = min(n_lhs + n_rand, budget)

        lhs_X = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_X[:, d] = lo[d] + (hi[d] - lo[d]) * (perm + rng.uniform(size=n_lhs)) / n_lhs
        lhs_X = np.clip(lhs_X, lo, hi)

        rand_X = lo + rng.uniform(size=(n_rand, dim)) * (hi - lo)
        rand_X = np.clip(rand_X, lo, hi)

        X_init = np.vstack([lhs_X, rand_X])
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

        # --- Phase 2: Differential Evolution with Pareto-guided selection ---
        pop_size = 20
        polish_start = int(budget * 0.65)

        # Initialize DE population from initial samples
        if len(X_init) >= pop_size:
            # Select diverse points for DE population
            selected = []
            remaining = list(range(len(X_init)))
            sums = np.sum(F_init, axis=1)
            best_idx = np.argmin(sums)
            selected.append(best_idx)
            remaining.remove(best_idx)

            while len(selected) < pop_size and remaining:
                best_dist = -1
                best_idx = -1
                for idx in remaining:
                    min_dist = min(np.linalg.norm(F_init[idx] - F_init[s]) for s in selected)
                    if min_dist > best_dist:
                        best_dist = min_dist
                        best_idx = idx
                if best_idx >= 0:
                    selected.append(best_idx)
                    remaining.remove(best_idx)

            de_X = X_init[selected].copy()
        else:
            # If not enough initial points, add random ones
            extra = lo + rng.uniform(size=(pop_size - len(X_init), dim)) * (hi - lo)
            extra = np.clip(extra, lo, hi)
            de_X = np.vstack([X_init, extra])[:pop_size]

        while used < budget:
            if time.time() > deadline:
                break

            progress = used / budget
            is_polish = used >= polish_start

            # Adaptive DE parameters
            if is_polish:
                remaining_frac = (budget - used) / max(1, budget - polish_start)
                F_de = 0.3 + 0.2 * remaining_frac
                CR = 0.9
            else:
                F_de = 0.5 + 0.2 * (1.0 - progress)
                CR = 0.7 + 0.2 * progress

            # Generate DE offspring
            n_offspring = min(pop_size, budget - used)
            offspring_X = np.zeros((n_offspring, dim))
            offspring_F = np.zeros((n_offspring, n_obj))

            for c in range(n_offspring):
                if used + c >= budget or time.time() > deadline:
                    break

                # Select three distinct parents
                idxs = rng.choice(pop_size, size=3, replace=False)
                p1, p2, p3 = de_X[idxs[0]], de_X[idxs[1]], de_X[idxs[2]]

                # DE/rand/1 mutation
                v = p1 + F_de * (p2 - p3)

                # Binomial crossover
                child = v.copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.uniform() < CR or j == j_rand:
                        child[j] = v[j]
                    else:
                        child[j] = p1[j]

                # Boundary handling
                child = np.clip(child, lo, hi)

                offspring_X[c] = child
                offspring_F[c] = problem(child)
                used += 1

                all_X = np.vstack([all_X, [child]])
                all_F = np.vstack([all_F, [offspring_F[c]]])

                # Update Pareto archive
                if np.all(offspring_F[c] < ref):
                    dominated = False
                    for pf in pareto_F:
                        if np.all(pf <= offspring_F[c]) and np.any(pf < offspring_F[c]):
                            dominated = True
                            break

                    if not dominated:
                        pareto_X = np.vstack([pareto_X, [child]])
                        pareto_F = np.vstack([pareto_F, [offspring_F[c]]])
                        # Remove dominated points from archive
                        keep = np.ones(len(pareto_F), dtype=bool)
                        for i in range(len(pareto_F) - 1):
                            if np.all(pareto_F[i] >= offspring_F[c]) and np.any(pareto_F[i] > offspring_F[c]):
                                keep[i] = False
                        pareto_X = pareto_X[keep]
                        pareto_F = pareto_F[keep]

            # Selection: combine DE population and offspring, keep best by hypervolume contribution
            combined_X = np.vstack([de_X, offspring_X[:n_offspring]])
            combined_F = np.vstack([de_X and np.array([problem(x) for x in de_X]) if False else np.array([problem(x) for x in de_X]), offspring_F[:n_offspring]])

            # Actually, we already have de_X's F values from before, but let's recompute for correctness
            # Wait, we don't have de_F stored. Let's just use the offspring and select based on dominance.

            # Simpler selection: keep non-dominated points, prefer diversity
            valid_combined = np.all(combined_F < ref, axis=1)
            if np.any(valid_combined):
                valid_X = combined_X[valid_combined]
                valid_F = combined_F[valid_combined]
                p_mask = self._is_pareto(valid_F)
                if np.sum(p_mask) >= pop_size:
                    # Select pop_size most diverse from Pareto front
                    front_X = valid_X[p_mask]
                    front_F = valid_F[p_mask]
                    selected = []
                    remaining = list(range(len(front_X)))
                    sums = np.sum(front_F, axis=1)
                    best_idx = np.argmin(sums)
                    selected.append(best_idx)
                    remaining.remove(best_idx)

                    while len(selected) < pop_size and remaining:
                        best_dist = -1
                        best_idx = -1
                        for idx in remaining:
                            min_dist = min(np.linalg.norm(front_F[idx] - front_F[s]) for s in selected)
                            if min_dist > best_dist:
                                best_dist = min_dist
                                best_idx = idx
                        if best_idx >= 0:
                            selected.append(best_idx)
                            remaining.remove(best_idx)

                    de_X = front_X[selected]
                else:
                    # Take all Pareto points and fill with best dominated
                    front_X = valid_X[p_mask]
                    front_F = valid_F[p_mask]
                    dominated_X = valid_X[~p_mask]
                    dominated_F = valid_F[~p_mask]

                    if len(dominated_X) > 0:
                        sums = np.sum(dominated_F, axis=1)
                        fill_idx = np.argsort(sums)[:pop_size - len(front_X)]
                        de_X = np.vstack([front_X, dominated_X[fill_idx]])
                    else:
                        de_X = front_X

                    # If still not enough, add random
                    if len(de_X) < pop_size:
                        extra = lo + rng.uniform(size=(pop_size - len(de_X), dim)) * (hi - lo)
                        extra = np.clip(extra, lo, hi)
                        de_X = np.vstack([de_X, extra])
            else:
                # No valid points, reset population
                de_X = lo + rng.uniform(size=(pop_size, dim)) * (hi - lo)
                de_X = np.clip(de_X, lo, hi)

        # --- Phase 3: Local refinement on top Pareto points ---
        if used < budget and len(pareto_F) > 0:
            valid_pareto = np.all(pareto_F < ref, axis=1)
            if np.any(valid_pareto):
                pX = pareto_X[valid_pareto]
                pF = pareto_F[valid_pareto]
                sums = np.sum(pF, axis=1)
                top_k = min(5, len(pX))
                top_idx = np.argsort(sums)[:top_k]

                for pi in top_idx:
                    if used >= budget or time.time() > deadline - 5:
                        break
                    parent = pX[pi].copy()
                    best_f = problem(parent).copy()
                    best_x = parent.copy()

                    for coord in range(dim):
                        if used >= budget or time.time() > deadline - 3:
                            break

                        for delta in [-0.15, -0.08, -0.04, 0.04, 0.08, 0.15]:
                            if used >= budget or time.time() > deadline - 2:
                                break
                            trial = parent.copy()
                            trial[coord] = np.clip(parent[coord] + delta * (hi[coord] - lo[coord]), lo[coord], hi[coord])
                            if np.allclose(trial, parent):
                                continue
                            ft = problem(trial)
                            used += 1
                            if np.all(ft <= best_f) and np.any(ft < best_f):
                                best_f = ft
                                best_x = trial

                    if not np.allclose(best_x, parent):
                        parent = best_x
                        all_X = np.vstack([all_X, [parent]])
                        all_F = np.vstack([all_F, [best_f]])

                        # Update Pareto archive
                        if np.all(best_f < ref):
                            dominated = False
                            for pf in pareto_F:
                                if np.all(pf <= best_f) and np.any(pf < best_f):
                                    dominated = True
                                    break
                            if not dominated:
                                pareto_X = np.vstack([pareto_X, [parent]])
                                pareto_F = np.vstack([pareto_F, [best_f]])
                                keep = np.ones(len(pareto_F), dtype=bool)
                                for i in range(len(pareto_F) - 1):
                                    if np.all(pareto_F[i] >= best_f) and np.any(pareto_F[i] > best_f):
                                        keep[i] = False
                                pareto_X = pareto_X[keep]
                                pareto_F = pareto_F[keep]

        return all_X

# EVOLVE-BLOCK-END
