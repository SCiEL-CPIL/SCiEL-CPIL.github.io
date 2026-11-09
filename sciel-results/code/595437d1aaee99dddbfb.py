"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer using adaptive evolutionary search with
    diversity-aware parent selection and vectorized non-dominated sorting.
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
        budget = self.budget

        # Phase 1: Initial random population
        init_pop = 40
        X_init = lo + (hi - lo) * np.random.random((init_pop, dim))
        all_X = [x.copy() for x in X_init]
        all_F = [np.array(problem(x), dtype=np.float64) for x in X_init]
        evals_used = init_pop

        # Helper: Vectorized non-dominated indices
        def get_nd_indices(F_list):
            n = len(F_list)
            if n == 0:
                return np.array([], dtype=int)
            F_arr = np.array(F_list)
            # Broadcasting: F_i (N, 1, M) vs F_j (1, N, M)
            # F_j <= F_i means j dominates i in component-wise sense
            # We need: exists j != i such that F_j <= F_i and F_j != F_i
            # Let's compute dominance matrix D where D[i,j] is True if j dominates i
            # D[i,j] = (F[j] <= F[i]).all() and (F[j] < F[i]).any()
            
            # Efficient way:
            # less_equal = np.all(F_arr[np.newaxis, :, :] <= F_arr[:, np.newaxis, :], axis=2) # (N, N)
            # less_than = np.any(F_arr[np.newaxis, :, :] < F_arr[:, np.newaxis, :], axis=2)
            # dominated_by_someone = np.any(less_equal & less_than, axis=1)
            
            # Note: F_arr[np.newaxis, :, :] is (1, N, M). F_arr[:, np.newaxis, :] is (N, 1, M).
            # So less_equal[i, j] is (F[j] <= F[i]).all()
            
            F_i = F_arr[:, np.newaxis, :] # (N, 1, M)
            F_j = F_arr[np.newaxis, :, :] # (1, N, M)
            
            le = np.all(F_j <= F_i, axis=2) # (N, N)
            lt = np.any(F_j < F_i, axis=2) # (N, N)
            
            dominated_mask = np.any(le & lt, axis=1) # (N,)
            nd_indices = np.where(~dominated_mask)[0]
            return nd_indices

        # Helper: Select parent with diversity bias
        def select_parent(nd_indices, F_list):
            if len(nd_indices) == 0:
                return np.random.randint(len(F_list))
            if len(nd_indices) == 1:
                return nd_indices[0]
            
            # Calculate isolation: min distance to other ND points
            F_nd = np.array([F_list[i] for i in nd_indices])
            n_nd = len(F_nd)
            min_dists = np.zeros(n_nd)
            
            for k in range(n_nd):
                dists = np.linalg.norm(F_nd - F_nd[k], axis=1)
                dists[k] = np.inf # Exclude self
                min_dists[k] = np.min(dists)
            
            # 70% chance: pick most isolated
            if np.random.random() < 0.7:
                idx_most_isolated = np.argmax(min_dists)
                return nd_indices[idx_most_isolated]
            else:
                # 30% chance: pick random from ND
                idx_random = np.random.randint(n_nd)
                return nd_indices[idx_random]

        # Phase 2: Evolutionary Refinement
        mutation_scale_start = 0.3
        mutation_scale_end = 0.02
        polish_threshold = int(budget * 0.9) # Start polish in last 10%
        
        while evals_used < budget:
            if time.time() > deadline:
                break
            
            nd_idx = get_nd_indices(all_F)
            parent_local = select_parent(nd_idx, all_F)
            parent_X = all_X[parent_local]
            
            # Determine step size
            if evals_used < polish_threshold:
                progress = 1.0 - (budget - evals_used) / max(1, budget - init_pop)
                scale = mutation_scale_start + (mutation_scale_end - mutation_scale_start) * progress
                scale = max(scale, mutation_scale_end)
            else:
                # Polish phase: very small steps
                scale = 0.01
            
            # Mutation
            # 5% chance of global restart to escape local optima
            if np.random.random() < 0.05:
                child_X = lo + (hi - lo) * np.random.random(dim)
            else:
                mutation = scale * (hi - lo) * np.random.standard_normal(dim)
                child_X = np.clip(parent_X + mutation, lo, hi)
            
            child_F = problem(child_X)
            evals_used += 1
            
            all_X.append(child_X)
            all_F.append(child_F)

        return None
# EVOLVE-BLOCK-END
