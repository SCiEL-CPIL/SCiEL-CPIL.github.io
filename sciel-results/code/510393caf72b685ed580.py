import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 310.0  # Safety margin
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            used += 1
            all_X.append(x.copy())
            all_F.append(f.copy())
            return f

        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return []
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if dominates(F[p], F[q]):
                        S[p].append(q)
                        n_dom[q] += 1
                    elif dominates(F[q], F[p]):
                        S[q].append(p)
                        n_dom[p] += 1
            
            fronts = []
            current_front = []
            for p in range(n):
                if n_dom[p] == 0:
                    rank[p] = 0
                    current_front.append(p)
            
            while current_front:
                fronts.append(current_front)
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            rank[q] = rank[p] + 1
                            next_front.append(q)
                current_front = next_front
            return fronts

        def crowding_distance(F, front_indices):
            if not front_indices:
                return {}
            m = len(front_indices)
            if m <= 2:
                return {i: float('inf') for i in front_indices}
            
            dist = {i: 0.0 for i in front_indices}
            for k in range(n_obj):
                sorted_front = sorted(front_indices, key=lambda i: F[i][k])
                dist[sorted_front[0]] = float('inf')
                dist[sorted_front[-1]] = float('inf')
                
                f_min = F[sorted_front[0]][k]
                f_max = F[sorted_front[-1]][k]
                f_range = f_max - f_min
                if f_range == 0:
                    continue
                for i in range(1, m - 1):
                    dist[sorted_front[i]] += (F[sorted_front[i+1]][k] - F[sorted_front[i-1]][k]) / f_range
            return dist

        # --- Strategy: Halton initialization + Hypervolume-guided DE + Local Polish ---
        
        def halton_sequence(n, dim):
            """Generate Halton sequence points in [0,1]^dim"""
            points = []
            for i in range(1, n + 1):
                point = []
                for base in range(2, dim + 2):
                    f = 1.0
                    r = 0.0
                    i_val = i
                    while i_val > 0:
                        f /= base
                        r += f * (i_val % base)
                        i_val //= base
                    point.append(r)
                points.append(point)
            return np.array(points)
        
        # 1. Halton initialization for better coverage (40% of budget)
        n_init = int(budget * 0.4)  # 80 points
        halton_pts = halton_sequence(n_init, dim)
        for i in range(n_init):
            if used >= budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * halton_pts[i]
            evaluate(x)
        
        # 2. Additional random sampling to fill remaining init budget
        n_rand_init = int(budget * 0.1)  # 20 more points
        for _ in range(n_rand_init):
            if used >= budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.random(dim)
            evaluate(x)

        # Convert to arrays for faster processing
        X_arr = np.array(all_X)
        F_arr = np.array(all_F)
        
        # Get initial Pareto front
        fronts = non_dominated_sort(F_arr)
        if not fronts:
            return all_X
            
        pareto_indices = fronts[0]
        pareto_F = F_arr[pareto_indices]
        
        # Sort Pareto front to help with local search direction
        # If we have only 1 point, handle it
        if len(pareto_F) == 0:
            return all_X
        
        # 2. Hypervolume-guided DE for the remaining budget
        
        N_pop = 25  # Population size for DE
        CR = 0.8
        F_base = 0.5
        
        def compute_hypervolume(F, ref_point):
            """Compute 2D hypervolume contribution"""
            if len(F) == 0:
                return 0.0
            # Filter points dominated by reference
            valid = F[np.all(F < ref_point, axis=1)]
            if len(valid) == 0:
                return 0.0
            # Sort by first objective
            valid = valid[np.argsort(valid[:, 0])]
            hv = 0.0
            prev_y = ref_point[1]
            for i in range(len(valid)):
                x = valid[i, 0]
                y = valid[i, 1]
                if y < prev_y:
                    hv += (ref_point[0] - x) * (prev_y - y)
                    prev_y = y
            return hv
        
        def hypervolume_contribution(F, ref_point, point_idx):
            """Compute hypervolume contribution of a single point"""
            # HV with point - HV without point
            hv_with = compute_hypervolume(F, ref_point)
            F_without = np.delete(F, point_idx, axis=0)
            hv_without = compute_hypervolume(F_without, ref_point)
            return hv_with - hv_without
        
        ref_point = np.array([3000.0, 0.06])
        
        # Initialize DE population from current Pareto front
        # Select diverse points using hypervolume contribution
        if len(pareto_F) >= N_pop:
            # Compute HV contributions and select top N_pop
            hv_contribs = np.array([hypervolume_contribution(pareto_F, ref_point, i) 
                                    for i in range(len(pareto_F))])
            top_indices = np.argsort(hv_contribs)[::-1][:N_pop]
            pop_F = pareto_F[top_indices]
            pop_X = X_arr[pareto_indices[top_indices]]
        else:
            # Use all Pareto front points
            pop_F = list(pareto_F)
            pop_X = list(X_arr[pareto_indices])
            
            # Add diverse random points from archive to reach N_pop
            n_needed = N_pop - len(pop_F)
            archive_indices = np.array([i for i in range(len(X_arr)) 
                                       if i not in pareto_indices])
            for _ in range(n_needed):
                if used >= budget or time.time() > deadline:
                    break
                if len(archive_indices) > 0:
                    idx = np.random.choice(archive_indices)
                    x = X_arr[idx]
                    f = F_arr[idx]
                else:
                    x = lo + (hi - lo) * np.random.random(dim)
                    f = evaluate(x)
                pop_F.append(f)
                pop_X.append(x)
                
            if len(pop_F) < N_pop:
                N_pop = len(pop_F)
                if N_pop < 2:
                    return all_X

        pop_X = np.array(pop_X)
        pop_F = np.array(pop_F)
        
        # Main loop with hypervolume-based selection
        remaining_budget = budget - used
        n_de_generations = max(1, remaining_budget // N_pop)
        
        for gen in range(n_de_generations):
            if used >= budget or time.time() > deadline:
                break
            
            # Adaptive F: decrease as we progress
            progress = gen / max(1, n_de_generations)
            F_factor = F_base * (1.0 - 0.3 * progress) * (0.7 + 0.6 * np.random.rand())
            
            offspring_X = []
            offspring_F = []
            
            # Generate N_pop offspring
            for i in range(N_pop):
                if used >= budget or time.time() > deadline:
                    break
                
                if N_pop < 4:
                    trial = pop_X[i] + 0.1 * (hi - lo) * np.random.randn(dim)
                    trial = np.clip(trial, lo, hi)
                else:
                    candidates = [j for j in range(N_pop) if j != i]
                    r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)
                    v = pop_X[r1] + F_factor * (pop_X[r2] - pop_X[r3])
                    
                    # Exponential crossover
                    trial = pop_X[i].copy()
                    j_rand = np.random.randint(dim)
                    for j in range(dim):
                        if j == j_rand:
                            for k in range(j_rand, dim):
                                trial[k] = v[k]
                            break
                    trial = np.clip(trial, lo, hi)
                
                f = evaluate(trial)
                offspring_X.append(trial)
                offspring_F.append(f)
            
            if not offspring_X:
                break
                
            # Combine population and offspring
            combined_X = np.vstack([pop_X, np.array(offspring_X)])
            combined_F = np.vstack([pop_F, np.array(offspring_F)])
            
            # Non-dominated sort
            fronts = non_dominated_sort(combined_F)
            if not fronts:
                break
            
            # Select next population using hypervolume contribution for front 0
            new_pop_X = []
            new_pop_F = []
            
            # First, add all points from front 0 (Pareto front)
            if len(fronts) > 0:
                front0 = fronts[0]
                # Compute HV contributions for front 0 points
                front0_F = combined_F[front0]
                if len(front0_F) > N_pop:
                    hv_contribs = np.array([hypervolume_contribution(front0_F, ref_point, i) 
                                            for i in range(len(front0_F))])
                    top_idx = np.argsort(hv_contribs)[::-1][:N_pop]
                    for idx in top_idx:
                        orig_idx = front0[idx]
                        new_pop_X.append(combined_X[orig_idx])
                        new_pop_F.append(combined_F[orig_idx])
                else:
                    for idx in front0:
                        new_pop_X.append(combined_X[idx])
                        new_pop_F.append(combined_F[idx])
            
            # Fill remaining with best from next fronts using crowding distance
            if len(new_pop_X) < N_pop:
                for rank in range(1, len(fronts)):
                    if len(new_pop_X) >= N_pop:
                        break
                    front = fronts[rank]
                    dists = crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: dists.get(i, 0.0), reverse=True)
                    for idx in sorted_front:
                        if len(new_pop_X) >= N_pop:
                            break
                        new_pop_X.append(combined_X[idx])
                        new_pop_F.append(combined_F[idx])
            
            if len(new_pop_X) == 0:
                break
                
            pop_X = np.array(new_pop_X)
            pop_F = np.array(new_pop_F)
        
        # 3. Final local polish on best Pareto points
        if used < budget and time.time() < deadline - 5:
            # Get current best Pareto front from archive
            X_arr_final = np.array(all_X)
            F_arr_final = np.array(all_F)
            fronts_final = non_dominated_sort(F_arr_final)
            
            if fronts_final:
                pareto_final = fronts_final[0]
                # Polish top 5 Pareto points
                n_polish = min(5, len(pareto_final))
                for p_idx in range(n_polish):
                    if used >= budget or time.time() > deadline - 3:
                        break
                    
                    base_x = X_arr_final[pareto_final[p_idx]]
                    # Nelder-Mead style local search
                    step_size = 0.05 * (hi - lo)
                    for _ in range(10):
                        if used >= budget or time.time() > deadline - 3:
                            break
                        
                        # Try small perturbations
                        best_f = F_arr_final[pareto_final[p_idx]]
                        best_x = base_x
                        
                        for _ in range(8):
                            if used >= budget:
                                break
                            perturbation = np.random.randn(dim) * step_size
                            trial_x = np.clip(base_x + perturbation, lo, hi)
                            trial_f = np.array(problem(trial_x), dtype=float)
                            used += 1
                            all_X.append(trial_x.copy())
                            all_F.append(trial_f.copy())
                            
                            # Check if this improves hypervolume
                            if np.all(trial_f < ref_point):
                                # Keep if it's non-dominated by current best
                                if not np.any(np.all(F_arr_final[pareto_final[p_idx]] <= trial_f) and 
                                             np.any(F_arr_final[pareto_final[p_idx]] < trial_f)):
                                    if np.all(trial_f <= best_f) and np.any(trial_f < best_f):
                                        best_x = trial_x
                                        best_f = trial_f
                        
                        step_size *= 0.7  # Shrink step
        
        return all_X
