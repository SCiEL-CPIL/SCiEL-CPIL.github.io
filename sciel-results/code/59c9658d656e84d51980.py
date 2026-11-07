import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-Start Nelder-Mead Simplex Optimizer.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # We will use multiple starts to avoid local minima.
        # Let's try 5 different starting points.
        n_starts = 5
        calls_made = 0
        x_best = None
        f_best = np.inf
        
        # Generate diverse starting points
        # Use a Latin Hypercube-like distribution or just random points in [-4, 4]
        # to ensure we cover the space where the minimum is likely ([-4, 4]^5).
        starts = np.random.uniform(-4.0, 4.0, size=(n_starts, n))
        
        # Allocate budget to each start
        # We need to reserve some budget for the simplex method's internal steps.
        # Nelder-Mead in scipy doesn't expose a strict call count limit easily in older versions,
        # but we can control it by wrapping the function or just trusting it converges fast.
        # However, to be safe with the "500 calls" hard limit, we should monitor calls.
        
        # Since scipy.optimize.minimize with Nelder-Mead might make many calls,
        # and we have a strict budget, it is safer to implement a simple Nelder-Mead 
        # ourselves or use a wrapper.
        
        # Let's implement a simple Nelder-Mead step counter or just use a small budget per start.
        # Actually, scipy's Nelder-Mead can be finicky with call counts if not careful.
        # Let's write a lightweight Nelder-Mead implementation to strictly control calls.
        
        def nelder_mead(x0, max_iter):
            nonlocal calls_made, x_best, f_best
            
            # Initialize simplex
            # x0 is shape (n,)
            simplex = [x0.copy() + np.random.normal(0, 0.1, n) for _ in range(n)]
            simplex.append(x0.copy())
            
            # Evaluate initial simplex
            fvals = []
            for i in range(n + 1):
                if calls_made >= budget:
                    return
                f = func(simplex[i])
                calls_made += 1
                fvals.append(f)
                
                if f < f_best:
                    f_best = f
                    x_best = simplex[i].copy()
            
            fvals = np.array(fvals)
            
            # Parameters
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            for _ in range(max_iter):
                if calls_made >= budget:
                    break
                
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = fvals[order]
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                
                if calls_made >= budget:
                    break
                f_r = func(x_r)
                calls_made += 1
                if f_r < f_best:
                    f_best = f_r
                    x_best = x_r.copy()
                
                # Check if reflection is good
                if fvals[0] <= f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                elif f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    
                    if calls_made >= budget:
                        break
                    f_e = func(x_e)
                    calls_made += 1
                    if f_e < f_best:
                        f_best = f_e
                        x_best = x_e.copy()
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                else:
                    # Contraction
                    # If f_r < fvals[-1], do outside contraction, else inside
                    if f_r < fvals[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (x_r - centroid)
                    
                    x_c = np.clip(x_c, -5.0, 5.0)
                    
                    if calls_made >= budget:
                        break
                    f_c = func(x_c)
                    calls_made += 1
                    if f_c < f_best:
                        f_best = f_c
                        x_best = x_c.copy()
                    
                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            
                            if calls_made >= budget:
                                return
                            fvals[i] = func(simplex[i])
                            calls_made += 1
                            if fvals[i] < f_best:
                                f_best = fvals[i]
                                x_best = simplex[i].copy()
                        
                        break # If we shrunk, we might want to stop this run or restart
                        # For simplicity, we stop this specific Nelder-Mead run.
                
                # Check for convergence
                if np.max(fvals) - np.min(fvals) < 1e-8:
                    break

        # Distribute budget among starts
        # Simple strategy: give each start a fraction of the budget.
        # Nelder-Mead typically needs ~100-200 iterations for dim 5.
        # Let's allocate 100 calls per start, total 500.
        budget_per_start = budget // n_starts
        
        for i in range(n_starts):
            if calls_made >= budget:
                break
            x0 = starts[i]
            nelder_mead(x0, max_iter=budget_per_start)
            
        return x_best, f_best
