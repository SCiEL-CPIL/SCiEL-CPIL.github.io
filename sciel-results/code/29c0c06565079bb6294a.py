    import numpy as np
    from scipy.optimize import minimize
    import time

    class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            start_time = time.time()
            deadline = start_time + 290
            
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            
            # Phase 1: Random Initialization
            init_count = 40
            X_init = np.random.uniform(lo, hi, size=(init_count, self.dim))
            F_init = np.array([problem(x) for x in X_init])
            
            all_X = [X_init]
            all_F = [F_init]
            used = init_count
            
            # Phase 2: Local Optimization with Weighted Sums
            # Generate weight vectors on the simplex
            # We want to cover the front.
            n_weights = 10
            # Use a simple grid or random points
            # Random points on simplex
            weights = np.random.dirichlet(np.ones(self.n_obj), size=n_weights)
            # Ensure we have the extreme points too?
            # Let's add the 3 axes and the centroid
            extra_weights = np.array([
                [1, 0, 0], [0, 1, 0], [0, 0, 1], [1/3, 1/3, 1/3]
            ])
            # Combine and shuffle? Or just use the 10 random + 4 extra = 14?
            # Budget check: 200 - 40 = 160.
            # If we do 14 runs, 160/14 = 11.4 evals/run.
            # L-BFGS-B with maxfun=10 is okay.
            
            weights = np.vstack([weights, extra_weights])
            # Remove duplicates? Not critical.
            
            remaining_budget = self.budget - used
            maxfun_per_run = max(5, int(remaining_budget / len(weights)))
            
            for w in weights:
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break
                
                def obj_func(x, w=w):
                    f = problem(x)
                    return np.dot(f, w)
                
                # Find best start from current best set (Init + previous results)
                # We should use the best point found so far for this weight
                # But we only have F_init and maybe previous results?
                # Let's just use F_init for simplicity, or maintain a global best list.
                
                # To improve, let's maintain a list of all evaluated points
                # and pick the best one for this weight from that list.
                
                # For now, use F_init.
                weighted_sums = F_init @ w
                best_idx = np.argmin(weighted_sums)
                x0 = X_init[best_idx]
                
                res = minimize(
                    obj_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxfun': maxfun_per_run, 'maxiter': maxfun_per_run}
                )
                
                if res.success or res.status == 1:
                    x_res = res.x
                    f_res = problem(x_res)
                    
                    all_X.append(x_res.reshape(1, -1))
                    all_F.append(f_res.reshape(1, -1))
                    
                    used += maxfun_per_run # Conservative estimate
            
            # Phase 3: Evolutionary Refinement if budget remains
            if used < self.budget and time.time() < deadline:
                # Get current non-dominated set
                X_curr = np.vstack(all_X)
                F_curr = np.vstack(all_F)
                
                # Simple non-dominated sort
                n_points = len(F_curr)
                non_dom_indices = []
                for i in range(n_points):
                    dominated = False
                    for j in range(n_points):
                        if i == j: continue
                        if np.all(F_curr[j] <= F_curr[i]) and np.any(F_curr[j] < F_curr[i]):
                            dominated = True
                            break
                    if not dominated:
                        non_dom_indices.append(i)
                
                if len(non_dom_indices) > 0:
                    X_nd = X_curr[non_dom_indices]
                    F_nd = F_curr[non_dom_indices]
                    
                    # Use remaining budget for evolution
                    remaining = self.budget - used
                    # Do a few generations
                    # Each generation creates len(X_nd) offspring
                    n_gen = remaining // max(1, len(X_nd))
                    
                    for gen in range(n_gen):
                        if time.time() > deadline:
                            break
                        if used >= self.budget:
                            break
                        
                        offspring_X = []
                        for _ in range(len(X_nd)):
                            p1, p2 = np.random.choice(len(X_nd), 2, replace=False)
                            alpha = np.random.rand()
                            child = alpha * X_nd[p1] + (1 - alpha) * X_nd[p2]
                            if np.random.rand() < 0.3:
                                gene = np.random.randint(self.dim)
                                child[gene] = np.random.uniform(lo[gene], hi[gene])
                            child = np.clip(child, lo, hi)
                            offspring_X.append(child)
                        
                        offspring_X = np.array(offspring_X)
                        offspring_F = np.array([problem(x) for x in offspring_X])
                        
                        used += len(offspring_X)
                        
                        all_X.append(offspring_X)
                        all_F.append(offspring_F)
                        
                        # Update non-dominated set
                        X_comb = np.vstack([X_nd, offspring_X])
                        F_comb = np.vstack([F_nd, offspring_F])
                        
                        # New non-dominated
                        n_comb = len(F_comb)
                        new_non_dom = []
                        for i in range(n_comb):
                            dominated = False
                            for j in range(n_comb):
                                if i == j: continue
                                if np.all(F_comb[j] <= F_comb[i]) and np.any(F_comb[j] < F_comb[i]):
                                    dominated = True
                                    break
                            if not dominated:
                                new_non_dom.append(i)
                                
                        X_nd = X_comb[new_non_dom]
                        F_nd = F_comb[new_non_dom]
                        
                        # Limit size of ND set to avoid explosion? 
                        # If it gets too big, we might run out of time for dominance checks.
                        # But 40-50 points is fine.
                        
            X_final = np.vstack(all_X)
            F_final = np.vstack(all_F)
            return X_final
    