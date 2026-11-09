import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # State
        X = []
        F = []
        eval_count = 0
        
        def eval_point(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            fx = np.array(problem(x))
            X.append(x)
            F.append(fx)
            eval_count += 1
            return fx
            
        def check_time():
            return time.time() < deadline - 5.0

        # Helper: Check if a point is non-dominated by the current set F
        def is_non_dominated(f_new, F_current):
            for f_old in F_current:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return False
            return True

        # Helper: Local Search (Simple Random Perturbation)
        # This is the structural change: exploiting local smoothness of WFG4
        def local_search(x, f, max_steps=5):
            nonlocal eval_count
            current_x = x.copy()
            current_f = f
            for _ in range(max_steps):
                if not check_time() or eval_count >= budget - 10: # Reserve budget
                    break
                # Generate a neighbor
                neighbor = current_x + np.random.randn(dim) * 0.1 * (hi - lo)
                neighbor = np.clip(neighbor, lo, hi)
                
                f_neighbor = eval_point(neighbor)
                if f_neighbor is None:
                    break
                
                # Accept if non-dominated by current best set
                # We check against the global F list for simplicity, or just the current point?
                # Checking against global F ensures we don't go backwards, but might be slow.
                # For a memetic step, accepting if it improves the current point or is non-dominated by it is good.
                # Let's accept if it's non-dominated by the current point AND non-dominated by the global set (approx).
                # Actually, just checking if it's non-dominated by the current point is enough for local improvement.
                if is_non_dominated(f_neighbor, [current_f]):
                    current_x = neighbor
                    current_f = f_neighbor
            return current_x, current_f

        # 1. Initial Sampling: LHS (Reduced size to allow local search)
        n_init = 25
        X_init = np.zeros((n_init, dim))
        for i in range(dim):
            bin_edges = np.linspace(lo[i], hi[i], n_init + 1)
            samples = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) * np.random.rand(n_init)
            np.random.shuffle(samples)
            X_init[:, i] = samples
            
        # Evaluate and refine initial points
        for i in range(n_init):
            if not check_time():
                break
            fx = eval_point(X_init[i])
            if fx is None:
                break
            # Perform local search
            # Use a small step count to conserve budget
            X_init[i], fx = local_search(X_init[i], fx, max_steps=4)
            
        # Helper: Get non-dominated indices
        def get_non_dominated_indices(F_list):
            nd_indices = []
            for i, f_i in enumerate(F_list):
                dominated = False
                for j, f_j in enumerate(F_list):
                    if i == j: continue
                    if np.all(f_j <= f_i) and np.any(f_j < f_i):
                        dominated = True
                        break
                if not dominated:
                    nd_indices.append(i)
            return nd_indices

        # 2. Expansion Phase: Refine non-dominated points
        if eval_count < budget:
            nd_indices = get_non_dominated_indices(F)
            if len(nd_indices) > 0:
                # Generate offspring from non-dominated points
                offspring_to_eval = []
                for idx in nd_indices:
                    parent_x = X[idx]
                    # Generate 2 children
                    for _ in range(2):
                        if eval_count >= budget - 5:
                            break
                        child_x = parent_x + np.random.randn(dim) * 0.2 * (hi - lo)
                        child_x = np.clip(child_x, lo, hi)
                        offspring_to_eval.append(child_x)
                        
                # Evaluate and local search offspring
                for ox in offspring_to_eval:
                    if not check_time() or eval_count >= budget - 2:
                        break
                    of = eval_point(ox)
                    if of is None:
                        break
                    # Local search 1 step
                    ox, of = local_search(ox, of, max_steps=1)

        return X
