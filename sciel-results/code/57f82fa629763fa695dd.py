"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid evolutionary multi-objective search with informed initialization and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _non_dominated_indices(self, F):
        """Return indices of non-dominated points in F (n x n_obj)."""
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return np.where(~dominated)[0]

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)

        all_X = []
        all_F = []
        used = 0

        def eval_point(x):
            nonlocal used
            if used >= self.budget or time.time() > deadline:
                return None
            f = problem(x)
            all_X.append(x.copy())
            all_F.append(np.array(f, dtype=float))
            used += 1
            return f

        # Phase 1: Informed initialization
        # WFG7 benefits from points near the center of the search space
        # Mix: 50% random, 30% center-biased, 20% stratified
        pop_size = min(20, self.budget // 5)
        for _ in range(pop_size):
            if time.time() > deadline:
                break
            r = rng.random()
            if r < 0.5:
                # Fully random
                x = lo + (hi - lo) * rng.random(self.dim)
            elif r < 0.8:
                # Center-biased: sample from a narrower range around the midpoint
                mid = (lo + hi) / 2.0
                spread = (hi - lo) * 0.3
                x = mid + spread * (rng.random(self.dim) * 2 - 1)
                x = np.clip(x, lo, hi)
            else:
                # Stratified: each dimension picks from a few levels
                x = np.array([
                    lo[i] + (hi[i] - lo[i]) * (k + 0.5) / 4.0
                    for k in [int(rng.integers(4)) for i in range(self.dim)]
                ])
            eval_point(x)

        # Phase 2: Evolutionary search with adaptive mutation
        if used < self.budget and all_F:
            F = np.array(all_F)
            X = np.array(all_X)
            n = len(F)

            # Adaptive step size: start at 0.15, shrink over time
            step_scale = 0.15
            shrink_rate = 0.97

            while used < self.budget and time.time() < deadline:
                # Find non-dominated points
                nd_idx = self._non_dominated_indices(F)
                if len(nd_idx) == 0:
                    nd_idx = np.arange(n)

                # Select parent from non-dominated set
                parent_idx = nd_idx[rng.integers(len(nd_idx))]
                parent = X[parent_idx].copy()

                # Mutate with adaptive step
                mutation = step_scale * (hi - lo) * rng.standard_normal(self.dim)
                child = parent + mutation
                child = np.clip(child, lo, hi)

                fc = eval_point(child)
                if fc is None:
                    break

                # Add to population
                X = np.vstack([X, child])
                F = np.vstack([F, np.array(fc)])
                n += 1

                # Remove worst point (highest sum of objectives) to keep population manageable
                if n > 30:
                    sums = np.sum(F, axis=1)
                    worst_idx = int(np.argmax(sums))
                    X = np.delete(X, worst_idx, axis=0)
                    F = np.delete(F, worst_idx, axis=0)
                    n -= 1

                # Shrink step size
                step_scale *= shrink_rate

        # Phase 3: Local refinement around best non-dominated points
        if used < self.budget and all_F:
            F_all = np.array(all_F)
            X_all = np.array(all_X)
            nd_idx = self._non_dominated_indices(F_all)

            if len(nd_idx) > 0:
                # Refine up to 5 best non-dominated points
                n_refine = min(5, len(nd_idx))
                # Pick the ones with lowest objective sums among non-dominated
                nd_sums = np.sum(F_all[nd_idx], axis=1)
                refine_order = nd_idx[np.argsort(nd_sums)[:n_refine]]

                for idx in refine_order:
                    if used >= self.budget or time.time() > deadline:
                        break
                    x0 = X_all[idx].copy()
                    f0 = F_all[idx].copy()
                    local_step = 0.05

                    for dim_i in range(self.dim):
                        if used >= self.budget or time.time() > deadline:
                            break
                        # Try perturbations in this dimension
                        for sign in [-1, 1]:
                            if used >= self.budget or time.time() > deadline:
                                break
                            x_try = x0.copy()
                            x_try[dim_i] += sign * local_step * (hi[dim_i] - lo[dim_i])
                            x_try = np.clip(x_try, lo, hi)

                            f_try = np.array(problem(x_try), dtype=float)
                            all_X.append(x_try.copy())
                            all_F.append(f_try)
                            used += 1

                            # Update if it improves (not dominated by f0)
                            if np.all(f_try <= f0) and np.any(f_try < f0):
                                x0 = x_try.copy()
                                f0 = f_try.copy()

                    local_step *= 0.5

        return all_X
# EVOLVE-BLOCK-END
