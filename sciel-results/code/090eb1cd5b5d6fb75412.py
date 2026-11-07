import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0 # Safety margin for 300s limit
        d = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            try:
                fx = func(x)
            except Exception:
                fx = np.inf
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial Exploration with Latin Hypercube Sampling
        n_init = 40
        if budget <= n_init:
            n_init = max(1, budget // 2)
        
        # Generate LHS points
        # For 5D, LHS is simple: divide [0,1] into n_init intervals, pick random point in each, permute per dimension
        lhs_points = np.zeros((n_init, d))
        for i in range(n_init):
            lhs_points[i] = np.random.uniform(0, 1, d)
            # Actually, proper LHS:
            # 1. Create grid of intervals
            # 2. For each dim, assign one interval to each row, permuted
            # Let's do a proper LHS
            pass
            
        # Proper LHS implementation
        lhs_points = np.zeros((n_init, d))
        for dim in range(d):
            # Random permutation of intervals
            perm = np.random.permutation(n_init)
            for i in range(n_init):
                lhs_points[i, dim] = (perm[i] + np.random.uniform(0, 1)) / n_init
                
        # Map [0, 1]^d to [-5, 5]^d
        lhs_points = lhs_points * 10.0 - 5.0
        
        # Evaluate initial points
        for i in range(n_init):
            if time.time() - start_time > timeout:
                break
            evaluate(lhs_points[i])
            
        if best_x is None:
            # Fallback if no successful call
            best_x = np.zeros(d)
            evaluate(best_x)

        # Phase 2: Nelder-Mead Local Refinement
        # Initialize simplex around best_x
        simplex = [best_x.copy()]
        for i in range(d):
            # Expand in each direction
            x_new = best_x.copy()
            x_new[i] += 0.5
            x_new = np.clip(x_new, -5.0, 5.0)
            simplex.append(x_new)
            
        # Evaluate simplex vertices
        simplex_f = []
        for i in range(len(simplex)):
            if time.time() - start_time > timeout:
                break
            simplex_f.append(evaluate(simplex[i]))
            
        # Nelder-Mead iterations
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma_nm = 0.5 # shrinkage
        
        while calls_made < budget:
            if time.time() - start_time > timeout:
                break
                
            # Sort simplex by function value
            indices = np.argsort(simplex_f)
            simplex = [simplex[i] for i in indices]
            simplex_f = [simplex_f[i] for i in indices]
            
            # Compute centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5.0, 5.0)
            f_ref = evaluate(x_ref)
            
            if f_ref < simplex_f[0]:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5.0, 5.0)
                f_exp = evaluate(x_exp)
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    simplex_f[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
            elif f_ref < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = x_ref
                simplex_f[-1] = f_ref
            else:
                # Contraction
                if f_ref < simplex_f[-1]:
                    # Outside contraction
                    x_con = centroid + rho * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - rho * (centroid - x_ref)
                x_con = np.clip(x_con, -5.0, 5.0)
                f_con = evaluate(x_con)
                
                if f_con < min(f_ref, simplex_f[-1]):
                    simplex[-1] = x_con
                    simplex_f[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex_f[i] = evaluate(simplex[i])
            
            # Check for convergence
            if len(simplex_f) > 0 and (max(simplex_f) - min(simplex_f)) < 1e-10:
                # Try local random search to escape plateau
                if calls_made < budget:
                    x_rand = best_x + np.random.normal(0, 0.1, d)
                    x_rand = np.clip(x_rand, -5.0, 5.0)
                    f_rand = evaluate(x_rand)
                    if f_rand < best_f:
                        # Reset simplex around new best
                        simplex = [best_x.copy()]
                        for i in range(d):
                            x_new = best_x.copy()
                            x_new[i] += 0.1
                            x_new = np.clip(x_new, -5.0, 5.0)
                            simplex.append(x_new)
                        simplex_f = []
                        for i in range(len(simplex)):
                            if time.time() - start_time > timeout:
                                break
                            simplex_f.append(evaluate(simplex[i]))
                        continue
                    else:
                        # Break out of Nelder-Mead and do more random search
                        break

        # Phase 3: Local Random Search for fine-tuning
        while calls_made < budget:
            if time.time() - start_time > timeout:
                break
            
            # Adaptive step size based on recent progress
            sigma_rs = 0.1
            y = best_x + sigma_rs * np.random.standard_normal(d)
            y = np.clip(y, -5.0, 5.0)
            evaluate(y)

        return best_x, best_f
