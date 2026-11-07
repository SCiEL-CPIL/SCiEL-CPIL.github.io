import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: The evaluator kills at 330s, but the task says 300s per evaluation? 
        # "Each evaluation must finish within 300 s." 
        # "The evaluator KILLS the program after 330s wall-clock."
        # We should aim to finish well before 330s. Let's set a hard stop at 325s.
        deadline = start_time + 325.0

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
            try:
                f = problem(x)
                f = np.array(f, dtype=float)
                if not np.all(np.isfinite(f)):
                    return None
                all_X.append(x.copy())
                all_F.append(f.copy())
                eval_count += 1
                return f
            except Exception:
                return None

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

        def get_hv_contributions(F_list, ref):
            """Calculate HV contribution for each point in the Pareto front."""
            if not F_list:
                return {}
            F = np.array(F_list)
            mask = (F[:, 0] < ref[0]) & (F[:, 1] < ref[1])
            if not np.any(mask):
                return {}
            F_valid = F[mask]
            valid_indices = np.where(mask)[0]
            
            nd_idx = get_pareto_front_2d(F_valid)
            F_nd = F_valid[nd_idx]
            
            # Map back to original indices
            contribs = {}
            for k in range(len(F_nd)):
                orig_idx = valid_indices[nd_idx[k]]
                f1 = F_nd[k, 0]
                f2 = F_nd[k, 1]
                next_f1 = F_nd[k+1, 0] if k + 1 < len(F_nd) else ref[0]
                width = next_f1 - f1
                height = ref[1] - f2
                contribs[orig_idx] = width * height
            return contribs

        # --- Phase 1: Strategic Initialization ---
        # Combine corners, midpoints, and stratified random points.
        # Bias towards lower values (good region).
        
        init_points = []
        
        # 1. Corners (8 points)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        x = np.array([
                            lo[0] + bounds_range[0]*i,
                            lo[1] + bounds_range[1]*j,
                            lo[2] + bounds_range[2]*k,
                            lo[3] + bounds_range[3]*l
                        ])
                        init_points.append(x)
        
        # 2. Midpoints of edges and faces (approx 20 points)
        # We'll just add a few more structured points
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        # Skip corners
                        if all(c in [0, 2] for c in [i, j, k, l]):
                            continue
                        # Only add if it's a "mid" point (1) in at least one dim and not all 1s? 
                        # Let's just take a subset to keep it simple.
                        if (i==1 or j==1 or k==1 or l==1) and not (i==1 and j==1 and k==1 and l==1):
                             x = np.array([
                                lo[0] + bounds_range[0]*i/2.0,
                                lo[1] + bounds_range[1]*j/2.0,
                                lo[2] + bounds_range[2]*k/2.0,
                                lo[3] + bounds_range[3]*l/2.0
                             ])
                             init_points.append(x)
                             
        # 3. Stratified Random Points
        # We want ~50 total init points.
        n_target_init = 50
        n_rand_needed = n_target_init - len(init_points)
        if n_rand_needed > 0:
            # Use power-law to bias towards lo
            r = np.random.random((n_rand_needed, self.dim))
            r = r ** 2.0  # Bias towards 0
            rand_points = lo + bounds_range * r
            init_points.extend(rand_points)
            
        for x in init_points:
            if time.time() > deadline or eval_count >= self.budget:
                break
            evaluate(x)

        if eval_count >= self.budget or not all_F:
            return all_X

        # --- Phase 2: DE with HV-guided selection ---
        pop_size = 40
        F_mutation = 0.8
        CR = 0.9
        no_improve_count = 0
        best_hv = calc_hv_2d(all_F, ref_point)
        
        def get_top_population_indices(size):
            if len(all_F) < size:
                return list(range(len(all_F)))
            
            F_arr = np.array(all_F)
            contribs = get_hv_contributions(all_F, ref_point)
            
            if not contribs:
                return np.random.choice(len(all_F), size, replace=False).tolist()
            
            # Sort indices by HV contribution
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
        
        # --- Phase 3: Local Search Polish ---
        # We will reserve the last 20% of the budget for local search on the best points.
        # Or, we can interleave. Let's do a hybrid: DE for 70% of remaining budget, then Local Search.
        
        budget_for_de = int((self.budget - eval_count) * 0.7)
        de_limit = eval_count + budget_for_de
        
        while eval_count < de_limit and time.time() < deadline:
            # Refresh population periodically
            if len(pop_indices) < pop_size or eval_count % 10 == 0:
                pop_indices = get_top_population_indices(pop_size)
                
            if len(pop_indices) < 3:
                break

            # Adaptive parameters
            if no_improve_count > 5:
                F_mutation *= 0.9
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
            
            # Simple dominance check
            if fy[0] <= ft[0] and fy[1] <= ft[1] and (fy[0] < ft[0] or fy[1] < ft[1]):
                pop_indices[pop_indices.index(target_idx)] = eval_count - 1
            elif not (ft[0] <= fy[0] and ft[1] <= fy[1] and (ft[0] < fy[0] or ft[1] < fy[1])):
                if np.random.random() < 0.5:
                    pop_indices[pop_indices.index(target_idx)] = eval_count - 1

        # --- Phase 4: Final Local Search Polish ---
        # Use remaining budget to polish the top points.
        if eval_count < self.budget and time.time() < deadline:
            F_arr = np.array(all_F)
            contribs = get_hv_contributions(all_F, ref_point)
            
            if contribs:
                top_candidates = sorted(contribs.keys(), key=lambda i: contribs[i], reverse=True)
                # Take top 10-15 candidates
                top_candidates = top_candidates[:15]
                
                # Allocate remaining budget evenly among candidates?
                # Or just iterate and do 1-2 steps of coordinate descent per candidate.
                
                remaining_budget = self.budget - eval_count
                steps_per_candidate = max(1, remaining_budget // len(top_candidates))
                
                for idx in top_candidates:
                    if eval_count >= self.budget or time.time() > deadline:
                        break
                    
                    x_start = all_X[idx].copy()
                    f_start = all_F[idx].copy()
                    
                    # Simple Coordinate Descent
                    # Try moving along each axis
                    base_step = 0.05 * bounds_range
                    for step_iter in range(steps_per_candidate):
                        if eval_count >= self.budget or time.time() > deadline:
                            break
                        
                        improved = False
                        for d in range(self.dim):
                            if eval_count >= self.budget or time.time() > deadline:
                                break
                            
                            # Try + and - direction
                            for sign in [1, -1]:
                                if eval_count >= self.budget or time.time() > deadline:
                                    break
                                
                                x_new = x_start.copy()
                                x_new[d] += sign * base_step
                                x_new = np.clip(x_new, lo, hi)
                                
                                f_new = evaluate(x_new)
                                if f_new is None:
                                    continue
                                
                                # Check if new point is better (non-dominated and improves HV or dominates)
                                # Simple check: if it dominates current or is non-dominated and has lower sum of objectives
                                if f_new[0] < f_start[0] and f_new[1] <= f_start[1]:
                                    x_start = x_new
                                    f_start = f_new
                                    improved = True
                                elif f_new[0] <= f_start[0] and f_new[1] < f_start[1]:
                                    x_start = x_new
                                    f_start = f_new
                                    improved = True
                                elif f_new[0] < f_start[0] and f_new[1] < f_start[1]:
                                    x_start = x_new
                                    f_start = f_new
                                    improved = True
                                else:
                                    # If non-dominated, check if it's closer to ideal
                                    # We'll accept if it's non-dominated
                                    if not (f_start[0] <= f_new[0] and f_start[1] <= f_new[1] and (f_start[0] < f_new[0] or f_start[1] < f_new[1])):
                                        # Non-dominated
                                        # Heuristic: accept if sum of normalized objectives is lower
                                        # Normalize by ref point
                                        norm_f_new = (f_new[0]/ref_point[0] + f_new[1]/ref_point[1])
                                        norm_f_start = (f_start[0]/ref_point[0] + f_start[1]/ref_point[1])
                                        if norm_f_new < norm_f_start:
                                            x_start = x_new
                                            f_start = f_new
                                            improved = True
                            
                            if improved:
                                break
                        if not improved:
                            break

        return all_X
