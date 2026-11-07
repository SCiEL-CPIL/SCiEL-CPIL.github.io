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

        # --- Strategy: Halton initialization + Hypervolume-guided CMA-ES + Local Polish ---
        
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
        
        # 2. Hypervolume-guided CMA-ES for the remaining budget
        
        N_pop = 25  # Population size for CMA-ES
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
        
        # Initialize CMA-ES population from current Pareto front
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
        
        # CMA-ES state initialization
        # Mean is the centroid of the initial population
        mean = np.mean(pop_X, axis=0)
        # Initial sigma: range of the bounds / sqrt(dim)
        sigma = np.max(hi - lo) / np.sqrt(dim)
        # Initial covariance matrix: identity
        C = np.eye(dim)
        # Inverse and square root of C (for sampling)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        
        # CMA-ES parameters
        lambda_ = N_pop
        mu = lambda_ // 2  # Number of selected parents
        # Weights for selected parents
        w = np.zeros(lambda_)
        w[:mu] = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Path lengths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Main loop with hypervolume-based selection
        remaining_budget = budget - used
        n_cma_generations = max(1, remaining_budget // N_pop)
        
        for gen in range(n_cma_generations):
            if used >= budget or time.time() > deadline:
                break
            
            # Sample lambda_ points from N(mean, sigma^2 * C)
            offspring_X = []
            offspring_F = []
            
            # Precompute C_sqrt using Cholesky-like decomposition (use eigh for stability)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)  # Ensure positive
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
            
            for i in range(N_pop):
                if used >= budget or time.time() > deadline:
                    break
                
                z = np.random.randn(dim)
                trial = mean + sigma * (C_sqrt @ z)
                trial = np.clip(trial, lo, hi)
                
                f = evaluate(trial)
                offspring_X.append(trial)
                offspring_F.append(f)
            
            if not offspring_X:
                break
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine population and offspring
            combined_X = np.vstack([pop_X, offspring_X])
            combined_F = np.vstack([pop_F, offspring_F])
            
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
            
            # Update CMA-ES state
            # Sort offspring by some quality metric (use HV contribution or simple rank)
            # For CMA-ES, we typically sort by single-objective quality. 
            # Here, we use a weighted sum or just take the best mu by non-dominance rank + crowding.
            
            # Simple approach: sort offspring by sum of objectives (approximation)
            # Or better: use the non-dominated sort ranks from the combined set, but restrict to offspring
            # Let's just sort offspring by their sum of objectives for simplicity in CMA-ES update
            # This is a heuristic since CMA-ES is single-objective, but we are using it as a distribution sampler.
            
            # Actually, for multi-objective, we should select the best mu offspring based on the archive.
            # Let's select the mu best offspring based on their rank in the combined non-dominated sort.
            
            # Re-rank just the offspring
            offspring_fronts = non_dominated_sort(offspring_F)
            if not offspring_fronts:
                continue
                
            selected_offspring_indices = []
            # Take from front 0 of offspring
            if len(offspring_fronts) > 0:
                selected_offspring_indices.extend(offspring_fronts[0])
            
            # If we need more, take from next fronts
            if len(selected_offspring_indices) < mu:
                for rank in range(1, len(offspring_fronts)):
                    if len(selected_offspring_indices) >= mu:
                        break
                    selected_offspring_indices.extend(offspring_fronts[rank])
            
            # Truncate to mu
            selected_offspring_indices = selected_offspring_indices[:mu]
            
            if len(selected_offspring_indices) == 0:
                continue
                
            # Get the selected offspring X
            y_selected = offspring_X[selected_offspring_indices]
            
            # Update mean
            mean_new = np.sum(w[:, np.newaxis] * y_selected, axis=0)
            
            # Update path lengths
            # Need C^{-1/2} for p_sigma update
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            mean_diff = (mean_new - mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ mean_diff
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * mean_diff
            
            # Update sigma (hessian damping)
            hs = np.sqrt((1 - (1 - c_c)**2) / c_c) / dim
            hs = np.clip(hs, 0, 1)
            hs = 1 + hs * (np.linalg.norm(p_c) / np.sqrt(dim) - 1)
            sigma = sigma * np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            # Clip sigma to prevent explosion
            sigma = np.clip(sigma, 1e-6, np.max(hi - lo))
            
            # Update covariance matrix
            delta_sigma = (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1) * c_sigma * (2 - c_sigma)
            rank_1 = np.outer(p_c, p_c)
            rank_mu = np.sum(w[:, np.newaxis, np.newaxis] * np.outer(y_selected - mean, y_selected - mean), axis=0) / sigma**2
            
            C = (1 + c_1 + c_mu - c_1 * c_sigma * (2 - c_sigma) * delta_sigma) * C \
                - c_1 * (np.outer(p_c, p_c) - (1 - c_c) * (1 - c_c**2) * C) \
                + c_mu * rank_mu
            
            # Ensure C is positive semi-definite
            eigvals_C, eigvecs_C = np.linalg.eigh(C)
            eigvals_C = np.maximum(eigvals_C, 1e-10)
            C = eigvecs_C @ np.diag(eigvals_C) @ eigvecs_C.T
            
            mean = mean_new
        
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
