"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Sobol quasi-random sweep + local Gaussian refinement around non-dominated points."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Phase 1: Sobol quasi-random initialization for uniform coverage
        # Use a power-of-2 number of points for Sobol
        sobol_n = 128
        while sobol_n < budget // 2:
            sobol_n *= 2
        sobol_n = min(sobol_n, budget // 2)

        sampler = qmc.Sobol(d=dim, scramble=True)
        points = sampler.random(sobol_n)
        X = lo + (hi - lo) * points
        F = np.array([problem(x) for x in X])
        used = sobol_n

        # Track all evaluated points and their objective values
        all_X = [X]
        all_F = [F]

        def get_non_dominated_indices(F_arr):
            """Return indices of non-dominated points in F_arr (minimization)."""
            n = len(F_arr)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(i + 1, n):
                    if dominated[j]:
                        continue
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        dominated[j] = True
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]

        # Phase 2: Local Gaussian perturbations around non-dominated points
        remaining = budget - used
        if remaining > 0:
            nd_idx = get_non_dominated_indices(F)
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F))

            # Adaptive mutation scale: start moderate, shrink as we refine
            base_scale = 0.15
            n_refine_per_point = max(1, remaining // max(len(nd_idx), 1))
            # Cap total evaluations
            n_refine_per_point = min(n_refine_per_point, remaining)

            # Use a fixed seed for reproducibility within the run
            rng = np.random.default_rng(42)

            refine_count = 0
            for scale_iter in range(5):
                if used >= budget:
                    break
                current_scale = base_scale * (0.6 ** scale_iter)
                for idx in nd_idx:
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    n_children = max(1, min(n_refine_per_point, budget - used))
                    for _ in range(n_children):
                        if used >= budget or time.time() > deadline:
                            break
                        child = X[idx] + current_scale * (hi - lo) * rng.standard_normal(dim)
                        child = np.clip(child, lo, hi)
                        fc = problem(child)
                        used += 1
                        refine_count += 1
                        # Add to our tracking
                        all_X.append(child.reshape(1, -1))
                        all_F.append(fc.reshape(1, -1))

            # Collect all points for final non-dominated extraction
            if all_X:
                X_all = np.vstack(all_X)
                F_all = np.vstack(all_F)
            else:
                X_all, F_all = X, F

            # Final phase: if we still have budget, do fine-grained local search
            # around the best non-dominated points
            remaining = budget - used
            if remaining > 0 and time.time() < deadline:
                nd_idx_final = get_non_dominated_indices(F_all)
                if len(nd_idx_final) > 0:
                    # Sort by sum of objectives, take top few
                    sums = np.sum(F_all[nd_idx_final], axis=1)
                    top_idx = nd_idx_final[np.argsort(sums)[:min(5, len(nd_idx_final))]]
                    for idx in top_idx:
                        if used >= budget or time.time() > deadline:
                            break
                        for _ in range(min(remaining, 20)):
                            if used >= budget or time.time() > deadline:
                                break
                            child = X_all[idx] + 0.03 * (hi - lo) * rng.standard_normal(dim)
                            child = np.clip(child, lo, hi)
                            fc = problem(child)
                            used += 1
                            all_X.append(child.reshape(1, -1))
                            all_F.append(fc.reshape(1, -1))

        # Return the last population (return value is ignored by evaluator)
        return X
# EVOLVE-BLOCK-END
