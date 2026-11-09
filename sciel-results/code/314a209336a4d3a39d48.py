import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 280.0
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        evals_used = 0
        
        def time_ok():
            return time.time() < deadline
        
        def evaluate(x):
            nonlocal best_f, best_x, evals_used
            x = np.clip(x, -5, 5)
            f = func(x)
            evals_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Multi-start Nelder-Mead with 10 diverse initial points
        # First 10 evaluations to seed, then aggressive local refinement
        
        # Generate 10 diverse initial points spread across the domain
        # Use a structured grid + random jitter for diversity
        init_points = []
        
        # Center point
        init_points.append(np.zeros(dim))
        
        # Points near the corners of [-4, 4]^5 (where minimizer lies)
        # Use a quasi-random spread
        for i in range(9):
            # Use deterministic diverse points
            p = np.zeros(dim)
            # Spread across dimensions
            for d in range(dim):
                # Alternate between -3, 0, 3 with some variation
                val = ((i * (d + 1)) % 3 - 1) * 3.0
                p[d] = val
            # Add small deterministic jitter based on index
            p += 0.5 * np.array([(i * 7 + d * 13) % 10 - 5 for d in range(dim)]) / 10.0
            p = np.clip(p, -4.5, 4.5)
            init_points.append(p)
        
        # Evaluate all 10 initial points
        for p in init_points:
            if not time_ok() or evals_used >= budget:
                break
            evaluate(p)
        
        if not time_ok() or evals_used >= budget:
            return best_x, best_f
        
        # Now perform Nelder-Mead refinement from the best starting point
        # and also from other promising points
        
        def nelder_mead(x0, f0, max_iter):
            nonlocal evals_used, best_f, best_x
            if not time_ok():
                return
            
            # Initialize simplex: x0 plus dim perturbed points
            simplex = [x0.copy()]
            fs = [f0]
            
            # Perturbation size
            delta = 0.5
            
            for d in range(dim):
                x_pert = x0.copy()
                x_pert[d] += delta
                x_pert = np.clip(x_pert, -5, 5)
                f_pert = evaluate(x_pert)
                simplex.append(x_pert)
                fs.append(f_pert)
            
            if not time_ok():
                return
            
            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            
            for it in range(max_iter):
                if not time_ok() or evals_used >= budget:
                    break
                
                # Sort simplex by function value
                order = np.argsort(fs)
                simplex = [simplex[i].copy() for i in order]
                fs = [fs[i] for i in order]
                
                # Check convergence
                if fs[-1] - fs[0] < 1e-10:
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = evaluate(x_ref)
                
                if not time_ok():
                    break
                
                if f_ref < fs[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = evaluate(x_exp)
                    
                    if not time_ok():
                        break
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fs[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fs[-1] = f_ref
                elif f_ref < fs[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    fs[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < fs[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (x_ref - centroid)
                    
                    x_con = np.clip(x_con, -5, 5)
                    f_con = evaluate(x_con)
                    
                    if not time_ok():
                        break
                    
                    if f_con < min(f_ref, fs[-1]):
                        simplex[-1] = x_con
                        fs[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = x0 + sigma * (simplex[i] - x0)
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fs[i] = evaluate(simplex[i])
                            if not time_ok():
                                break
                        if not time_ok():
                            break
                        x0 = simplex[0].copy()
        
        # Try Nelder-Mead from the top 3 best initial points
        # Sort initial points by their function values
        # We need to re-evaluate to get sorted order, but we already have them
        # Let's just use the best 3 from our tracked best
        
        # Actually, let's track the initial evaluations properly
        # Re-do: we have 10 points evaluated, find the best 3
        
        # Since we don't have the mapping stored, let's just do NM from best_x
        # and also try a few more diverse starts
        
        # First, refine from current best
        remaining_budget = budget - evals_used
        if remaining_budget > 20:
            nm_iters = min(remaining_budget - 5, 150)
            nelder_mead(best_x.copy(), best_f, nm_iters)
        
        if not time_ok() or evals_used >= budget:
            return best_x, best_f
        
        # Try a second NM from a different promising region
        # Pick a point that's different from best_x
        second_start = None
        for p in init_points:
            if np.linalg.norm(p - best_x) > 2.0:
                second_start = p
                break
        
        if second_start is not None:
            remaining_budget = budget - evals_used
            if remaining_budget > 20:
                f_start = evaluate(second_start)
                nm_iters = min(remaining_budget - 5, 100)
                nelder_mead(second_start.copy(), f_start, nm_iters)
        
        if not time_ok() or evals_used >= budget:
            return best_x, best_f
        
        # Final fine-tuning: (1+1)-ES style local search
        remaining = budget - evals_used
        if remaining > 5 and time_ok():
            x_local = best_x.copy()
            f_local = best_f
            sigma_local = 0.1
            
            for _ in range(remaining):
                if not time_ok() or evals_used >= budget:
                    break
                
                z = rng.standard_normal(dim)
                x_new = x_local + sigma_local * z
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                evals_used += 1
                
                if f_new < f_local:
                    f_local = f_new
                    x_local = x_new.copy()
                    best_f = f_local
                    best_x = x_local.copy()
                    sigma_local *= 1.2
                else:
                    sigma_local *= 0.85
                
                sigma_local = np.clip(sigma_local, 1e-6, 1.0)
        
        return best_x, best_f
