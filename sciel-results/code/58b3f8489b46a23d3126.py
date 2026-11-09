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
            X.append(x.copy())
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

        # Helper: Local Search with adaptive step size and diversity preservation
        def local_search(x, f, max_steps=6, step_scale=0.1):
            nonlocal eval_count
            current_x = x.copy()
            current_f = f
            step_size = step_scale * (hi - lo)
            best_x = current_x.copy()
            best_f = current_f.copy()
            
            for step in range(max_steps):
                if not check_time() or eval_count >= budget - 5:
                    break
                # Generate a neighbor
                neighbor = current_x + np.random.randn(dim) * step_size
                neighbor = np.clip(neighbor, lo, hi)
                
                f_neighbor = eval_point(neighbor)
                if f_neighbor is None:
                    break
                
                # Accept if non-dominated by current point OR if it's better in some objectives
                if is_non_dominated(f_neighbor, [current_f]):
                    current_x = neighbor
                    current_f = f_neighbor
                    step_size *= 0.85
                    # Track best
                    if np.all(f_neighbor <= best_f) and np.any(f_neighbor < best_f):
                        best_x = neighbor.copy()
                        best_f = f_neighbor.copy()
                else:
                    step_size *= 0.75
            
            return best_x, best_f

        # Helper: Get non-dominated indices using vectorized comparison
        def get_non_dominated_indices(F_list):
            if len(F_list) <= 1:
                return list(range(len(F_list)))
            
            F_arr = np.array(F_list)
            n = len(F_arr)
            nd_mask = np.ones(n, dtype=bool)
            
            for i in range(n):
                if not nd_mask[i]:
                    continue
                # Check if any other point dominates i
                for j in range(n):
                    if i == j or not nd_mask[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        nd_mask[i] = False
                        break
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        nd_mask[j] = False
            return np.where(nd_mask)[0].tolist()

        # Phase 1: Initial sampling with stratified LHS for good coverage
        # Use 30 initial points with stratified sampling
        n_init = 30
        X_init = np.zeros((n_init, dim))
        
        # Stratified sampling: divide each dimension into n_init bins, pick one random point per bin
        for i in range(dim):
            bin_width = (hi[i] - lo[i]) / n_init
            # Create n_init bins and pick one random offset per bin
            offsets = np.random.rand(n_init) * bin_width
            for j in range(n_init):
                X_init[j, i] = lo[i] + j * bin_width + offsets[j]
        
        # Evaluate initial points
        for i in range(n_init):
            if not check_time() or eval_count >= budget - 10:
                break
            fx = eval_point(X_init[i])
            if fx is None:
                break
        
        # Phase 2: Refine promising regions using local search on non-dominated points
        nd_indices = get_non_dominated_indices(F)
        
        # Sort non-dominated points by hypervolume contribution (approximate: sum of objectives)
        if len(nd_indices) > 0:
            nd_scores = [np.sum(F[i]) for i in nd_indices]
            sorted_nd = sorted(zip(nd_indices, nd_scores), key=lambda x: x[1])
            
            # Refine top 15 non-dominated points
            for idx, score in sorted_nd[:15]:
                if not check_time() or eval_count >= budget - 15:
                    break
                x = X[idx]
                f = F[idx]
                # Local search with moderate step
                _, _ = local_search(x, f, max_steps=5, step_scale=0.08)
        
        # Phase 3: Generate diverse offspring from the current Pareto front
        nd_indices = get_non_dominated_indices(F)
        if len(nd_indices) > 0:
            # Pick up to 10 diverse non-dominated points
            if len(nd_indices) > 10:
                # Use simple diversity: pick evenly spaced in objective space
                F_nd = np.array([F[i] for i in nd_indices])
                # Normalize objectives
                F_norm = (F_nd - F_nd.min(axis=0)) / (F_nd.max(axis=0) + 1e-10)
                # Pick points that are far apart in normalized objective space
                selected = [nd_indices[0]]
                for idx in nd_indices[1:]:
                    f_idx = F_nd[nd_indices.index(idx)] if idx in nd_indices else None
                    if f_idx is None:
                        continue
                    # Check if this point is sufficiently different from all selected
                    is_diverse = True
                    for sel_idx in selected:
                        f_sel = F_nd[nd_indices.index(sel_idx)] if sel_idx in nd_indices else None
                        if f_sel is None:
                            continue
                        if np.linalg.norm(F_norm[nd_indices.index(idx)] - F_norm[nd_indices.index(sel_idx)]) < 0.3:
                            is_diverse = False
                            break
                    if is_diverse:
                        selected.append(idx)
                    if len(selected) >= 10:
                        break
            else:
                selected = nd_indices
            
            # Generate 2 children per selected parent
            for idx in selected:
                if not check_time() or eval_count >= budget - 10:
                    break
                parent_x = X[idx]
                for _ in range(2):
                    if eval_count >= budget - 5:
                        break
                    # Crossover-like: mix with random point
                    if len(X) > 1:
                        other_idx = np.random.choice(len(X))
                        if other_idx != idx:
                            child_x = 0.5 * parent_x + 0.5 * X[other_idx] + np.random.randn(dim) * 0.05 * (hi - lo)
                        else:
                            child_x = parent_x + np.random.randn(dim) * 0.08 * (hi - lo)
                    else:
                        child_x = parent_x + np.random.randn(dim) * 0.08 * (hi - lo)
                    
                    child_x = np.clip(child_x, lo, hi)
                    cf = eval_point(child_x)
                    if cf is None:
                        break
                    # Quick local search
                    _, _ = local_search(child_x, cf, max_steps=3, step_scale=0.05)
        
        # Phase 4: Final intensive refinement of the best non-dominated points
        nd_indices = get_non_dominated_indices(F)
        if len(nd_indices) > 0:
            # Pick top 8 non-dominated points for final polish
            nd_scores = [np.sum(F[i]) for i in nd_indices]
            sorted_nd = sorted(zip(nd_indices, nd_scores), key=lambda x: x[1])
            
            for idx, score in sorted_nd[:8]:
                if not check_time() or eval_count >= budget - 3:
                    break
                x = X[idx]
                f = F[idx]
                # Fine-grained local search
                _, _ = local_search(x, f, max_steps=4, step_scale=0.03)
        
        return X
