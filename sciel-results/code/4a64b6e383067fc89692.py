import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        self.calls = 0
        rng = np.random.default_rng(42)

        def wrapper(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exhausted")
            self.calls += 1
            return func(x)

        best_x = None
        best_f = np.inf

        # Phase 1: Multi-start initialization
        # We use a mix of center and random points
        n_starts = 10
        candidates = []
        
        # Always try the origin
        x0 = np.zeros(dim)
        try:
            f0 = wrapper(x0)
            candidates.append((f0, x0.copy()))
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
        except ValueError:
            pass

        # Random starts
        for _ in range(n_starts - 1):
            if self.calls >= budget:
                break
            x = rng.uniform(-4.0, 4.0, dim)
            try:
                f = wrapper(x)
                candidates.append((f, x.copy()))
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            except ValueError:
                break

        # Sort candidates by fitness
        candidates.sort(key=lambda x: x[0])
        
        # Phase 2: Local Optimization using Powell's Method (Derivative-Free)
        # Powell is very effective for Rosenbrock and doesn't require gradients.
        # It uses function evaluations only.
        
        # We will run Powell from the top few candidates to increase robustness.
        # Budget allocation: We have roughly (budget - calls_so_far) left.
        # Powell's method in scipy has 'maxiter' and 'maxfev'. 
        # 'maxfev' is the total function evaluations allowed.
        
        # We'll run from the best candidate first.
        # If it uses too many calls, we stop.
        
        for i, (f0, x0) in enumerate(candidates[:3]):
            if self.calls >= budget:
                break
            
            # Estimate remaining budget for this run
            # We want to leave some budget for potential other starts or polishing.
            # But usually, the first good start converges fast.
            # Let's give the first start most of the budget, and less to subsequent ones.
            if i == 0:
                maxfev = budget - self.calls
            else:
                # If we are here, the first run didn't exhaust the budget.
                # Give a smaller portion, e.g., 20% of remaining.
                maxfev = max(10, int((budget - self.calls) * 0.2))
            
            try:
                # Powell method parameters
                # ftol, xtol, gtol are convergence criteria.
                # For Rosenbrock, we want high precision.
                res = minimize(
                    wrapper, 
                    x0, 
                    method='Powell', 
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': 1000, 
                        'maxfev': maxfev,
                        'ftol': 1e-12,
                        'xtol': 1e-12
                    }
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                # If Powell fails or exceeds budget, break
                break

        # Phase 3: Polish with Nelder-Mead if budget remains
        # Nelder-Mead is another derivative-free method, often good for final polishing.
        if best_x is not None and self.calls < budget:
            remaining = budget - self.calls
            if remaining > 50: # Only if we have significant budget left
                try:
                    res_nm = minimize(
                        wrapper,
                        best_x,
                        method='Nelder-Mead',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={
                            'maxiter': 1000,
                            'maxfev': remaining,
                            'xatol': 1e-10,
                            'fatol': 1e-12
                        }
                    )
                    if res_nm.fun < best_f:
                        best_f = res_nm.fun
                        best_x = res_nm.x.copy()
                except Exception:
                    pass

        return best_x, best_f
