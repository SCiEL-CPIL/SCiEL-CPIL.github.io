"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer using NSGA-II-style selection with tournament
    selection, crowding distance for diversity, and adaptive mutation.
    """
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Phase 1: Initial random population
        init_pop = min(50, budget)
        X_init = lo + (hi - lo) * np.random.random((init_pop, dim))
        F_init = np.array([problem(x) for x in X_init])
        evals_used = init_pop

        # Store all evaluated points
        all_X = [x.copy() for x in X_init]
        all_F = [f.copy() for f in F_init]

        # Helper: compute non-dominated front indices using vectorized approach
        def get_non_dominated_indices(F_list):
            n = len(F_list)
            if n == 0:
                return []
            F_arr = np.array(F_list)
            # Create dominance matrix: F_arr[j] <= F_arr[i] for all obj, and < for at least one
            # F_arr[j] <= F_arr[i] for all j,i
            leq = np.all(F_arr[:, None, :] <= F_arr[None, :, :], axis=2)  # leq[j,i] = True if F[j] <= F[i]
            lt = np.all(F_arr[:, None, :] <= F_arr[None, :, :], axis=2) & np.any(F_arr[:, None, :] < F_arr[None, :, :], axis=2)
            # lt[j,i] = True if F[j] strictly dominates F[i]
            # A point i is non-dominated if no j strictly dominates it
            dominated = np.any(lt, axis=0)  # dominated[i] = True if any j dominates i
            nd_indices = np.where(~dominated)[0].tolist()
            return nd_indices

        # Helper: compute crowding distance for a set of points
        def crowding_distance(F_list):
            n = len(F_list)
            if n <= 2:
                return [float('inf')] * n
            F_arr = np.array(F_list)
            cd = np.zeros(n)
            for m in range(n_obj):
                sorted_idx = np.argsort(F_arr[:, m])
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                for i in range(1, n - 1):
                    if F_arr[sorted_idx[i + 1], m] - F_arr[sorted_idx[i - 1], m] > 1e-10:
                        cd[sorted_idx[i]] += (F_arr[sorted_idx[i + 1], m] - F_arr[sorted_idx[i - 1], m]) / (F_arr[sorted_idx[i + 1], m] - F_arr[sorted_idx[i - 1], m] + 1e-10)
            return cd

        # Phase 2: NSGA-II-style evolutionary refinement
        remaining = budget - evals_used
        mutation_scale_start = 0.25
        mutation_scale_end = 0.015
        tournament_size = 3

        # Track best non-dominated points for final polish
        best_nd_X = []
        best_nd_F = []

        while evals_used < budget:
            if time.time() > deadline:
                break

            # Get non-dominated front from all evaluated points
            nd_idx = get_non_dominated_indices(all_F)

            if len(nd_idx) == 0:
                break

            # Update best non-dominated set
            best_nd_X = [all_X[i] for i in nd_idx]
            best_nd_F = [all_F[i] for i in nd_idx]

            # Compute crowding distance for the non-dominated front
            cd = crowding_distance(best_nd_F)

            # Tournament selection: pick tournament_size random non-dominated points,
            # choose the one with the largest crowding distance
            if len(nd_idx) > 1:
                candidates = np.random.choice(len(nd_idx), size=min(tournament_size, len(nd_idx)), replace=False)
                best_candidate = candidates[np.argmax([cd[c] for c in candidates])]
                parent_local = nd_idx[best_candidate]
            else:
                parent_local = nd_idx[0]

            parent_X = all_X[parent_local]

            # Adaptive mutation step size
            progress = 1.0 - (budget - evals_used) / max(1, remaining)
            scale = mutation_scale_start + (mutation_scale_end - mutation_scale_start) * progress
            scale = max(scale, mutation_scale_end)

            # Gaussian mutation
            mutation = scale * (hi - lo) * np.random.standard_normal(dim)
            child_X = np.clip(parent_X + mutation, lo, hi)

            # Evaluate child
            child_F = problem(child_X)
            evals_used += 1

            # Add to archive
            all_X.append(child_X)
            all_F.append(child_F)

        # Phase 3: Final polish - refine the best non-dominated points with small steps
        if best_nd_X and evals_used < budget:
            polish_budget = min(20, budget - evals_used)
            for point_idx in range(min(len(best_nd_X), 10)):
                if evals_used >= budget or time.time() > deadline:
                    break
                base_X = best_nd_X[point_idx]
                # Try a few small perturbations around this point
                for _ in range(2):
                    if evals_used >= budget or time.time() > deadline:
                        break
                    small_scale = 0.01 * (hi - lo)
                    mutation = small_scale * np.random.standard_normal(dim)
                    polish_X = np.clip(base_X + mutation, lo, hi)
                    polish_F = problem(polish_X)
                    evals_used += 1
                    all_X.append(polish_X)
                    all_F.append(polish_F)

        return None
# EVOLVE-BLOCK-END
