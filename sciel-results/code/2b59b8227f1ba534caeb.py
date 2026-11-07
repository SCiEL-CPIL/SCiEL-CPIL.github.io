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
        deadline = start_time + 285  # Safety margin before 300s limit

        def check_time():
            return time.time() > deadline

        # 1. Deterministic Sobol Sampling for Initial Exploration
        # This ensures we cover the space [-5, 5]^5 efficiently without random clustering.
        # We use the first N points of a Sobol sequence.
        # For dim=5, we need to generate enough points.
        # scipy.stats.qmc.Sobol is available.
        from scipy.stats.qmc import Sobol
        
        # We will use a mix of Sobol points and local refinement.
        # Let's reserve ~20% of the budget for the global search phase.
        n_global = 100
        if budget <= n_global:
            # Fallback for extremely small budgets
            x = np.random.uniform(-5, 5, dim)
            f = func(x)
            return x, f

        # Generate Sobol points
        # Sobol requires power of 2 number of points for best properties, but works for others too.
        # We'll generate 128 points (2^7) which is plenty for 5D.
        sampler = Sobol(d=dim, scramble=True, seed=42)
        # Sample 128 points in [0, 1]^d
        Y = sampler.random(128)
        
        # Map to [-5, 5]^d
        X_global = Y * 10.0 - 5.0
        
        # Evaluate all global points
        F_global = np.array([func(x) for x in X_global])
        evals = len(X_global)
        
        best_idx = np.argmin(F_global)
        best_x = X_global[best_idx].copy()
        best_f = F_global[best_idx]
        
        if check_time():
            return best_x, best_f

        # 2. Local Refinement using Nelder-Mead
        # Nelder-Mead is robust to rotation and ill-conditioning in low dimensions.
        # We will run Nelder-Mead from the best global point.
        # We can also try a few other good points if time permits.
        
        from scipy.optimize import minimize
        
        # Wrap the function to ensure it returns a float and handles clipping (though func already clips)
        def objective(x):
            return func(x)
        
        # Run Nelder-Mead from the best point
        # maxiter is set high, but we rely on the deadline.
        # However, scipy's minimize doesn't have a time limit, only iteration limits.
        # We need to be careful. Let's use a wrapper that checks time?
        # No, we can't easily interrupt scipy.optimize.minimize from inside.
        # Instead, we can run it with a reasonable maxiter and then check time.
        # Or, we can implement a simple line search ourselves if needed, but Nelder-Mead is standard.
        
        # Let's try Nelder-Mead with a tight maxiter to avoid getting stuck.
        # If it doesn't converge, we can restart or switch methods.
        
        # Options for Nelder-Mead
        options = {
            'maxiter': 100,
            'xatol': 1e-8,
            'fatol': 1e-8
        }
        
        res = minimize(objective, best_x, method='Nelder-Mead', options=options)
        
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x.copy()
        
        evals += res.nfev
        
        if check_time():
            return best_x, best_f

        # 3. Further refinement with Coordinate Descent
        # Schaffers F7 is ill-conditioned but often has structure that coordinate descent can exploit
        # if the axes are aligned, but here they are rotated.
        # However, after Nelder-Mead, we are close.
        # Let's do a few rounds of coordinate descent with line search.
        
        # Simple coordinate descent
        step_size = 0.1
        for _ in range(50):
            if check_time() or evals >= budget:
                break
            
            improved = False
            for i in range(dim):
                # Try positive step
                x_plus = best_x.copy()
                x_plus[i] += step_size
                x_plus = np.clip(x_plus, -5, 5)
                f_plus = func(x_plus)
                evals += 1
                
                if f_plus < best_f:
                    best_f = f_plus
                    best_x = x_plus
                    improved = True
                    # Try to extend the step
                    for _ in range(10):
                        x_next = best_x.copy()
                        x_next[i] += step_size
                        x_next = np.clip(x_next, -5, 5)
                        f_next = func(x_next)
                        evals += 1
                        if f_next < best_f:
                            best_f = f_next
                            best_x = x_next
                        else:
                            break
                
                # Try negative step
                x_minus = best_x.copy()
                x_minus[i] -= step_size
                x_minus = np.clip(x_minus, -5, 5)
                f_minus = func(x_minus)
                evals += 1
                
                if f_minus < best_f:
                    best_f = f_minus
                    best_x = x_minus
                    improved = True
                    # Try to extend the step
                    for _ in range(10):
                        x_next = best_x.copy()
                        x_next[i] -= step_size
                        x_next = np.clip(x_next, -5, 5)
                        f_next = func(x_next)
                        evals += 1
                        if f_next < best_f:
                            best_f = f_next
                            best_x = x_next
                        else:
                            break
                
                if check_time() or evals >= budget:
                    break
            
            if not improved:
                # Reduce step size
                step_size *= 0.5
                if step_size < 1e-6:
                    break
        
        return best_x, best_f
