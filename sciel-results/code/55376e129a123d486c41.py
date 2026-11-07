import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # 1. Generate Weight Vectors on Simplex - structured grid
        H = 5
        weights_list = []
        for i in range(H + 1):
            for j in range(H + 1 - i):
                k = H - i - j
                w = np.array([i / H, j / H, k / H])
                weights_list.append(w)
        N = len(weights_list)
        weights = np.array(weights_list)
        
        # 2. Define Neighbors - 6 nearest for better exploration
        neighbors = []
        for i in range(N):
            dists = np.linalg.norm(weights - weights[i], axis=1)
            sorted_idx = np.argsort(dists)
            neigh = [sorted_idx[j] for j in range(1, 7) if sorted_idx[j] != i]
            neighbors.append(neigh)
        
        # 3. Initialize Population using LHS
        X = np.zeros((N, dim))
        for d in range(dim):
            edges = np.linspace(lo[d], hi[d], N + 1)
            centers = np.array([(edges[i] + edges[i+1]) / 2 for i in range(N)])
            np.random.shuffle(centers)
            X[:, d] = centers
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = N
        
        # Initialize Ideal Point
        z_star = np.min(F, axis=0)
        
        # 4. Evolution Loop
        budget_for_main = budget - 40  # Reserve 40 for polish
        
        i_counter = 0
        gen = 0
        
        while used < budget_for_main:
            if time.time() > deadline - 5:
                break
            
            i = i_counter % N
            i_counter += 1
            gen += 1
            
            # Adaptive mutation factor: start higher, decrease over time
            progress = used / max(budget_for_main, 1)
            F_mut = 0.9 - 0.3 * progress  # 0.9 -> 0.6
            
            # Select parents from neighbors
            candidate_indices = neighbors[i]
            if len(candidate_indices) < 3:
                candidate_indices = np.random.choice(N, 3, replace=False)
            else:
                dists_to_i = np.array([np.linalg.norm(weights[c] - weights[i]) for c in candidate_indices])
                probs = 1.0 / (dists_to_i + 1e-6)
                probs = probs / probs.sum()
                candidate_indices = np.random.choice(candidate_indices, 3, replace=False, p=probs)
            
            r1, r2, r3 = candidate_indices
            
            # DE Mutation
            v = X[r1] + F_mut * (X[r2] - X[r3])
            
            # Binomial Crossover
            j_rand = np.random.randint(dim)
            CR = 0.5
            trial = np.zeros(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = X[i][j]
            
            trial = np.clip(trial, lo, hi)
            
            f_trial = np.array(problem(trial))
            used += 1
            
            z_star = np.minimum(z_star, f_trial)
            
            # Update Population (MOEA/D with Tchebycheff)
            indices_to_check = [i] + neighbors[i]
            
            for k in indices_to_check:
                d_trial = np.max(weights[k] * np.abs(f_trial - z_star))
                d_current = np.max(weights[k] * np.abs(F[k] - z_star))
                
                if d_trial < d_current:
                    X[k] = trial
                    F[k] = f_trial
        
        # 5. Hypervolume-aware local polish using Nelder-Mead on smooth surrogate
        if time.time() < deadline - 10:
            # Compute approximate HV contribution for each point
            ref_point = np.array([1.1, 1.3, 1.1])
            # Filter points that are better than reference
            valid_mask = np.all(F < ref_point, axis=1)
            
            # Simple HV contribution: volume of box dominated by point but not by others
            # Use a fast approximation: for each point, compute the volume of the region
            # where it is the "best" in at least one objective
            hv_contrib = np.zeros(N)
            for i in range(N):
                if not valid_mask[i]:
                    continue
                # The contribution is roughly the product of ranges where this point
                # is the minimum in each objective dimension
                contrib = 1.0
                for obj in range(n_obj):
                    # Range from F[i][obj] to the next best in that objective
                    others = np.delete(F, i, axis=0)
                    others_valid = others[valid_mask[np.delete(np.arange(N), i)]]
                    if len(others_valid) == 0:
                        best_other = ref_point[obj]
                    else:
                        best_other = np.min(others_valid[obj])
                    if F[i][obj] >= best_other:
                        contrib = 0.0
                        break
                    contrib *= (best_other - F[i][obj])
                hv_contrib[i] = contrib
            
            # Sort by HV contribution (descending)
            sorted_idx = np.argsort(-hv_contrib)
            
            # Allocate polish budget proportionally to HV contribution
            remaining_budget = budget - used
            if remaining_budget <= 0:
                return X
            
            # Give more budget to high-contribution points
            total_contrib = hv_contrib[sorted_idx].sum() + 1e-10
            for rank, idx in enumerate(sorted_idx):
                if used >= budget or time.time() > deadline - 3:
                    break
                
                # Allocate budget proportional to contribution, minimum 2, max 10
                alloc = int(remaining_budget * hv_contrib[idx] / total_contrib)
                alloc = max(2, min(alloc, 10))
                
                if alloc <= 0:
                    continue
                
                # Nelder-Mead local search on Tchebycheff surrogate
                w = weights[idx]
                x0 = X[idx].copy()
                best_f = F[idx].copy()
                best_val = np.max(w * np.abs(best_f - z_star))
                
                # Initialize simplex
                simplex = [x0.copy()]
                step = 0.05
                for d in range(dim):
                    pt = x0.copy()
                    pt[d] += step
                    pt = np.clip(pt, lo, hi)
                    simplex.append(pt)
                
                # Evaluate initial simplex
                simplex_f = [problem(simplex[0])]
                for s in range(1, dim + 1):
                    simplex_f.append(problem(simplex[s]))
                    used += 1
                    if used >= budget or time.time() > deadline - 3:
                        break
                
                if used >= budget or time.time() > deadline - 3:
                    break
                
                # Nelder-Mead iterations
                max_iter = min(alloc, dim + 1)
                for _ in range(max_iter):
                    if used >= budget or time.time() > deadline - 3:
                        break
                    
                    # Sort simplex by objective value
                    obj_vals = [np.max(w * np.abs(f - z_star)) for f in simplex_f]
                    order = np.argsort(obj_vals)
                    simplex = [simplex[order[i]] for i in range(dim + 1)]
                    simplex_f = [simplex_f[order[i]] for i in range(dim + 1)]
                    obj_vals = [obj_vals[order[i]] for i in range(dim + 1)]
                    
                    # Centroid of best dim points
                    centroid = np.mean(simplex[:dim], axis=0)
                    
                    # Reflection
                    alpha = 1.0
                    x_ref = centroid + alpha * (centroid - simplex[dim])
                    x_ref = np.clip(x_ref, lo, hi)
                    f_ref = problem(x_ref)
                    used += 1
                    
                    if obj_vals[0] <= np.max(w * np.abs(f_ref - z_star)) < obj_vals[-2]:
                        simplex[dim] = x_ref
                        simplex_f[dim] = f_ref
                        continue
                    
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - simplex[dim])
                    x_exp = np.clip(x_exp, lo, hi)
                    f_exp = problem(x_exp)
                    used += 1
                    
                    if np.max(w * np.abs(f_exp - z_star)) < np.max(w * np.abs(f_ref - z_star)):
                        simplex[dim] = x_exp
                        simplex_f[dim] = f_exp
                    else:
                        simplex[dim] = x_ref
                        simplex_f[dim] = f_ref
                        continue
                    
                    # Contraction
                    rho = 0.5
                    if np.max(w * np.abs(f_ref - z_star)) < obj_vals[0]:
                        x_con = centroid + rho * (simplex[0] - centroid)
                    else:
                        x_con = centroid - rho * (simplex[dim] - centroid)
                    x_con = np.clip(x_con, lo, hi)
                    f_con = problem(x_con)
                    used += 1
                    
                    if np.max(w * np.abs(f_con - z_star)) < max(np.max(w * np.abs(f_ref - z_star)), obj_vals[0]):
                        simplex[dim] = x_con
                        simplex_f[dim] = f_con
                    else:
                        # Shrink
                        for s in range(1, dim + 1):
                            simplex[s] = (simplex[0] + simplex[s]) / 2.0
                            simplex_f[s] = problem(simplex[s])
                            used += 1
                            if used >= budget:
                                break
                        if used >= budget:
                            break
                
                # Update with best found
                best_in_simplex = np.argmin([np.max(w * np.abs(f - z_star)) for f in simplex_f])
                if np.max(w * np.abs(simplex_f[best_in_simplex] - z_star)) < best_val:
                    X[idx] = simplex[best_in_simplex]
                    F[idx] = simplex_f[best_in_simplex]
                    z_star = np.minimum(z_star, F[idx])
        
        # 6. Final tiny perturbations for all points
        if time.time() < deadline - 2:
            for i in range(N):
                if used >= budget:
                    break
                if time.time() > deadline - 2:
                    break
                
                w = weights[i]
                best_val = np.max(w * np.abs(F[i] - z_star))
                best_x = X[i].copy()
                best_f = F[i].copy()
                
                for _ in range(2):
                    if used >= budget:
                        break
                    if time.time() > deadline - 2:
                        break
                    
                    perturbation = np.random.uniform(-0.02, 0.02, dim)
                    new_x = np.clip(best_x + perturbation, lo, hi)
                    new_f = np.array(problem(new_x))
                    used += 1
                    new_val = np.max(w * np.abs(new_f - z_star))
                    
                    if new_val < best_val:
                        best_val = new_val
                        best_x = new_x.copy()
                        best_f = new_f.copy()
                
                if best_val < np.max(w * np.abs(F[i] - z_star)):
                    X[i] = best_x
                    F[i] = best_f
                    z_star = np.minimum(z_star, F[i])
        
        return X
