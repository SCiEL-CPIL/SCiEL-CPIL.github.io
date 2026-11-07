"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer on ZDT6 using structured evolutionary search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Initial population - spread across the space
        pop_size = min(40, budget // 5)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def non_dominated_mask(f_arr):
            """Return boolean mask of non-dominated points."""
            n = len(f_arr)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                # Check if any other point dominates i
                for j in range(n):
                    if i == j or not mask[j]:
                        continue
                    if np.all(f_arr[j] <= f_arr[i]) and np.any(f_arr[j] < f_arr[i]):
                        mask[i] = False
                        break
            return mask

        def crowding_distance(f_arr, nd_mask):
            """Compute crowding distance for non-dominated points."""
            n = len(f_arr)
            cd = np.zeros(n)
            if not np.any(nd_mask):
                return cd
            nd_indices = np.where(nd_mask)[0]
            for obj in range(f_arr.shape[1]):
                sorted_idx = np.argsort(f_arr[nd_indices, obj])
                cd[nd_indices[sorted_idx[0]]] = np.inf
                cd[nd_indices[sorted_idx[-1]]] = np.inf
                f_range = f_arr[nd_indices, obj].max() - f_arr[nd_indices, obj].min()
                if f_range == 0:
                    continue
                for k in range(1, len(sorted_idx) - 1):
                    i_prev = nd_indices[sorted_idx[k - 1]]
                    i_next = nd_indices[sorted_idx[k + 1]]
                    cd[nd_indices[sorted_idx[k]]] += (f_arr[i_next, obj] - f_arr[i_prev, obj]) / f_range
            return cd

        # Adaptive mutation step
        step_scale = 0.15

        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Find non-dominated points
            nd_mask = non_dominated_mask(F)
            
            # Select parent from non-dominated set using crowding
            if np.any(nd_mask):
                nd_indices = np.where(nd_mask)[0]
                cd = crowding_distance(F, nd_mask)
                # Weighted selection: prefer higher crowding distance
                weights = cd[nd_indices]
                weights = weights / weights.sum()
                parent_idx = nd_indices[rng.choice(len(nd_indices), p=weights)]
            else:
                parent_idx = rng.integers(len(F))

            # Adaptive mutation
            step = step_scale * (hi - lo) * rng.standard_normal(dim)
            child = X[parent_idx] + step
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replace worst point (by sum of objectives, but only if child is better or non-dominated)
            # Use a smarter replacement: find point most likely to be dominated by child
            best_replace = -1
            best_score = np.inf
            for i in range(len(F)):
                # Score: how much worse is F[i] compared to child?
                # If child dominates F[i], high priority
                if np.all(fc <= F[i]) and np.any(fc < F[i]):
                    score = -1000  # child dominates, replace immediately
                else:
                    # Use normalized sum as proxy
                    score = np.sum(F[i]) - np.sum(fc)
                if score < best_score:
                    best_score = score
                    best_replace = i

            X[best_replace] = child
            F[best_replace] = fc

            # Adapt step size based on progress
            if used % 20 == 0:
                step_scale *= 0.95  # gradually reduce step size

        return X
# EVOLVE-BLOCK-END
