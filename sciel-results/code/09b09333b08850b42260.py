import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        bounds_range = hi - lo

        all_X = []
        all_F = []
        eval_count = 0

        ref_point = np.array([3000.0, 0.06])

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return None
            if time.time() > deadline:
                return None
            f = problem(x)
            f = np.array(f, dtype=float)
            if not np.all(np.isfinite(f)):
                return None
            all_X.append(x.copy())
            all_F.append(f.copy())
            eval_count += 1
            return f

        def get_pareto_front_2d(F):
            """Efficient 2D non-dominated sorting. O(N log N)."""
            if len(F) == 0:
                return np.array([], dtype=int)
            sorted_idx = np.argsort(F[:, 0])
            F_sorted = F[sorted_idx]
            min_f2 = np.inf
            nd_indices = []
            for i in range(len(F_sorted)):
                if F_sorted[i, 1] < min_f2:
                    nd_indices.append(sorted_idx[i])
                    min_f2 = F_sorted[i, 1]
            return np.array(nd_indices, dtype=int)

        def calc_hv_2d(F_list, ref):
            """Calculate exact 2D hypervolume."""
            if not F_list:
                return 0.0
            F = np.array(F_list)
            mask = (F[:, 0] < ref[0]) & (F[:, 1] < ref[1])
            if not np.any(mask):
                return 0.0
            F = F[mask]
            nd_idx = get_pareto_front_2d(F)
            F_nd = F[nd_idx]
            if len(F_nd) == 0:
                return 0.0
            hv = 0.0
            for i in range(len(F_nd)):
                f1 = F_nd[i, 0]
                f2 = F_nd[i, 1]
                next_f1 = F_nd[i+1, 0] if i + 1 < len(F_nd) else ref[0]
                width = next_f1 - f1
                height = ref[1] - f2
                hv += width * height
            return hv

        def hv_contribution_2d(point, nd_points_sorted, ref):
            """Calculate HV contribution of a specific point in sorted ND set."""
            p1, p2 = point
            if p1 >= ref[0] or p2 >= ref[1]:
                return 0.0
            idx = -1
            for i in range(len(nd_points_sorted)):
                if np.allclose(nd_points_sorted[i], point):
                    idx = i
                    break
            if idx == -1:
                return 0.0
            next_f1 = nd_points_sorted[idx+1, 0] if idx + 1 < len(nd_points_sorted) else ref[0]
            width = next_f1 - p1
            height = ref[1] - p2
            return width * height

        # --- Phase 1: Strategic Initialization ---
        # Focus on lower-left region (small f1, small f2) since ref is [3000, 0.06]
        # Use stratified sampling with emphasis on low values
        n_init = 50
        
        # Create stratified samples with bias toward lower values
        init_points = []
        n_strat = 30
        n_rand = n_init - n_strat
        
        for i in range(n_strat):
            # Use power-law distribution to bias toward lower values
            r = np.random.random(self.dim)
            x = lo + bounds_range * (r ** 2.0)  # Bias toward low end
            init_points.append(x)
        
        for i in range(n_rand):
            x = lo + bounds_range * np.random.random(self.dim)
            init_points.append(x)
        
        init_points = np.array(init_points)
        
        for x in init_points:
            if time.time() > deadline or eval_count >= self.budget:
                break
            evaluate(x)

        if eval_count >= self.budget:
            return all_X

        # --- Phase 2: DE with HV-guided selection + Local Search ---
        
        pop_size = 45
        F_mutation = 0.8
        CR = 0.9
        no_improve_count = 0
        best_hv = calc_hv_2d(all_F, ref_point)
        
        def get_top_population_indices(size):
            if len(all_F) < size:
                return list(range(len(all_F)))
            
            F_arr = np.array(all_F)
            nd_idx = get_pareto_front_2d(F_arr)
            
            valid_nd = []
            for i in nd_idx:
                f = all_F[i]
                if f[0] < ref_point[0] and f[1] < ref_point[1]:
                    valid_nd.append(i)
            
            if len(valid_nd) == 0:
                return np.random.choice(len(all_F), size, replace=False).tolist()
            
            valid_nd_sorted = sorted(valid_nd, key=lambda i: all_F[i][0])
            F_sorted_nd = np.array([all_F[i] for i in valid_nd_sorted])
            
            contribs = {}
            for k in range(len(valid_nd_sorted)):
                idx = valid_nd_sorted[k]
                p1, p2 = F_sorted_nd[k]
                next_f1 = F_sorted_nd[k+1, 0] if k + 1 < len(valid_nd_sorted) else ref_point[0]
                width = next_f1 - p1
                height = ref_point[1] - p2
                contribs[idx] = width * height
            
            sorted_indices = sorted(contribs.keys(), key=lambda i: contribs[i], reverse=True)
            selected = sorted_indices[:size]
            
            if len(selected) < size:
                remaining = size - len(selected)
                available = [i for i in range(len(all_F)) if i not in selected]
                if available:
                    rand_fill = np.random.choice(available, min(remaining, len(available)), replace=False)
                    selected.extend(rand_fill.tolist())
            
            return selected

        pop_indices = get_top_population_indices(pop_size)
        
        def local_search(x_start, max_steps=8):
            """Simple local search with adaptive step sizes."""
            best_x = x_start.copy()
            best_f = evaluate(x_start)
            if best_f is None:
                return best_x, best_f
            
            base_step = 0.1
            for step in range(max_steps):
                if time.time() > deadline or eval_count >= self.budget:
                    break
                
                # Try random perturbations with decreasing step size
                step_size = base_step * (0.7 ** step)
                improved = False
                
                for trial in range(4):
                    if time.time() > deadline or eval_count >= self.budget:
                        break
                    
                    # Random direction
                    direction = np.random.randn(self.dim)
                    direction /= np.linalg.norm(direction)
                    
                    x_new = best_x + step_size * direction
                    x_new = np.clip(x_new, lo, hi)
                    
                    f_new = evaluate(x_new)
                    if f_new is None:
                        continue
                    
                    # Check if improvement in HV terms (simple: check if dominates or non-dominated)
                    if f_new[0] <= best_f[0] and f_new[1] <= best_f[1] and (f_new[0] < best_f[0] or f_new[1] < best_f[1]):
                        best_x = x_new
                        best_f = f_new
                        improved = True
                    elif not (best_f[0] <= f_new[0] and best_f[1] <= f_new[1] and (best_f[0] < f_new[0] or best_f[1] < f_new[1])):
                        # Non-dominated - accept with some probability
                        if np.random.random() < 0.3:
                            best_x = x_new
                            best_f = f_new
                            improved = True
                
                if not improved:
                    break
            
            return best_x, best_f

        while eval_count < self.budget and time.time() < deadline:
            # Refresh population periodically
            if len(pop_indices) < pop_size or eval_count % 12 == 0:
                pop_indices = get_top_population_indices(pop_size)
                
            if len(pop_indices) < 3:
                break

            # Adaptive parameters
            if no_improve_count > 6:
                F_mutation *= 0.85
                no_improve_count = 0
                if F_mutation < 0.1:
                    F_mutation = 0.1
            
            # Pick parents
            parents = np.random.choice(pop_indices, 3, replace=False)
            x1 = all_X[parents[0]]
            x2 = all_X[parents[1]]
            x3 = all_X[parents[2]]

            # Mutation
            v = x1 + F_mutation * (x2 - x3)
            
            # Crossover
            u = np.random.random(self.dim) < CR
            y = np.where(u, v, x1)
            
            # Clip
            y = np.clip(y, lo, hi)

            fy = evaluate(y)
            if fy is None:
                break

            # Occasionally do local search from the new point
            if np.random.random() < 0.25 and eval_count < self.budget - 10:
                y, fy = local_search(y, max_steps=6)
                if fy is None:
                    break

            # Update best HV
            current_hv = calc_hv_2d(all_F, ref_point)
            if current_hv > best_hv + 1e-9:
                best_hv = current_hv
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Selection
            target_idx = np.random.choice(pop_indices)
            ft = all_F[target_idx]
            
            if fy[0] <= ft[0] and fy[1] <= ft[1] and (fy[0] < ft[0] or fy[1] < ft[1]):
                pop_indices[pop_indices.index(target_idx)] = eval_count - 1
            elif not (ft[0] <= fy[0] and ft[1] <= fy[1] and (ft[0] < fy[0] or ft[1] < fy[1])):
                if np.random.random() < 0.5:
                    pop_indices[pop_indices.index(target_idx)] = eval_count - 1

        # --- Phase 3: Final local search on best points ---
        if eval_count < self.budget and time.time() < deadline:
            F_arr = np.array(all_F)
            nd_idx = get_pareto_front_2d(F_arr)
            valid_nd = [i for i in nd_idx if all_F[i][0] < ref_point[0] and all_F[i][1] < ref_point[1]]
            
            # Sort by HV contribution and pick top candidates for final polish
            if valid_nd:
                valid_nd_sorted = sorted(valid_nd, key=lambda i: all_F[i][0])
                F_sorted_nd = np.array([all_F[i] for i in valid_nd_sorted])
                contribs = {}
                for k in range(len(valid_nd_sorted)):
                    idx = valid_nd_sorted[k]
                    p1, p2 = F_sorted_nd[k]
                    next_f1 = F_sorted_nd[k+1, 0] if k + 1 < len(valid_nd_sorted) else ref_point[0]
                    width = next_f1 - p1
                    height = ref_point[1] - p2
                    contribs[idx] = width * height
                
                top_candidates = sorted(contribs.keys(), key=lambda i: contribs[i], reverse=True)[:10]
                
                for idx in top_candidates:
                    if eval_count >= self.budget or time.time() > deadline:
                        break
                    local_search(all_X[idx], max_steps=10)

        return all_X
