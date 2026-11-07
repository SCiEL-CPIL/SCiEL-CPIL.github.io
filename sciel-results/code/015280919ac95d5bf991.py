import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, use 300s for search to be safe
        deadline = start_time + 300
        
        rng = np.random.default_rng(42)
        
        # 1. Initial Sampling
        # Rosenbrock has a narrow valley. Random uniform sampling is inefficient.
        # We use a small set of structured points and random points to find a decent start.
        # The origin is often a reasonable start for Rosenbrock if shifted/rotated mildly,
        # but we need to be robust.
        
        n_initial = 20
        # Include origin, some scaled ones, and random
        points = [
            np.zeros(self.dim),
            np.ones(self.dim),
            -np.ones(self.dim),
            np.ones(self.dim) * 0.5,
            np.ones(self.dim) * -0.5,
        ]
        
        random_points = rng.uniform(-5, 5, (n_initial - len(points), self.dim))
        X = np.vstack(points + list(random_points))
        F = np.array([func(x) for x in X])
        
        best_idx = np.argmin(F)
        best_x = X[best_idx].copy()
        best_f = F[best_idx]
        
        calls_used = len(X)
        remaining = self.budget - calls_used
        
        if time.time() > deadline - 5:
            return best_x, best_f

        # 2. BFGS Optimization
        # Strategy: Use BFGS with finite differences.
        # BFGS builds an approximate Hessian, which is crucial for Rosenbrock's curved valley.
        # L-BFGS-B is a variant, but standard BFGS with a tight budget control might be more predictable
        # or we can just use L-BFGS-B but with a single, very long run instead of multi-start,
        # OR we can use COBYLA or Nelder-Mead if we want to be truly gradient-free in the interface,
        # but the prompt suggests BFGS quasi-Newton.
        
        # Let's stick to the directive: "Use a gradient-free BFGS ... that builds an approximate Hessian from finite-difference gradient estimates"
        # This effectively describes what L-BFGS-B does, or standard BFGS if we provide gradients.
        # Since we can't provide analytic gradients (black box), we rely on FD.
        # scipy.optimize.minimize with method='BFGS' uses FD gradients.
        # 'L-BFGS-B' is better for bounds. Let's use L-BFGS-B as it handles bounds [-5,5] natively.
        
        # However, the "different strategy" directive asks for a structural change.
        # The current program does: 40 initial + 1 L-BFGS + 8 restarts + 1 polish.
        # The new strategy: A single, robust BFGS/L-BFGS-B run that consumes the vast majority of the budget,
        # relying on the curvature model to navigate the valley efficiently, rather than restarting.
        # Restarting can be detrimental if the first run is already in the basin.
        
        call_count = [0]
        
        def wrapper(x):
            if call_count[0] >= remaining:
                raise ValueError("Budget exceeded")
            if time.time() > deadline - 5:
                raise ValueError("Time exceeded")
            call_count[0] += 1
            return func(x)

        try:
            # Use L-BFGS-B with a high maxiter and maxfun
            # 'gtol' controls gradient norm convergence.
            # 'ftol' controls function value change.
            res = minimize(wrapper, best_x, method='L-BFGS-B', 
                           bounds=[(-5, 5)] * self.dim,
                           options={'maxiter': 1000, 'maxfun': remaining, 'maxls': 50, 'gtol': 1e-8, 'ftol': 1e-8})
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
                
        except Exception:
            pass
        
        remaining = self.budget - (calls_used + call_count[0])
        
        if time.time() > deadline - 10:
            return best_x, best_f

        # 3. If we have significant budget left and haven't converged tightly, do a local Nelder-Mead polish
        # Nelder-Mead is derivative-free and can handle the final steps well if BFGS stalled slightly.
        # This is a structural addition: using a different optimizer family (simplex) for the final phase.
        
        if remaining > 50:
            nm_call_count = [0]
            
            def nm_wrapper(x):
                if nm_call_count[0] >= remaining:
                    raise ValueError("Budget exceeded")
                if time.time() > deadline - 5:
                    raise ValueError("Time exceeded")
                nm_call_count[0] += 1
                return func(x)
            
            try:
                # Nelder-Mead with tight tolerances
                res_nm = minimize(nm_wrapper, best_x, method='Nelder-Mead',
                                  options={'maxiter': 1000, 'maxfev': remaining, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
                
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = res_nm.x.copy()
                    
            except Exception:
                pass
            
            remaining = self.budget - (calls_used + call_count[0] + nm_call_count[0])
        
        # 4. Final BFGS polish if time allows
        if remaining > 20 and time.time() < deadline - 10:
            final_call_count = [0]
            
            def final_wrapper(x):
                if final_call_count[0] >= remaining:
                    raise ValueError("Budget exceeded")
                if time.time() > deadline - 5:
                    raise ValueError("Time exceeded")
                final_call_count[0] += 1
                return func(x)
            
            try:
                res_final = minimize(final_wrapper, best_x, method='L-BFGS-B',
                                     bounds=[(-5, 5)] * self.dim,
                                     options={'maxiter': 500, 'maxfun': remaining, 'gtol': 1e-12, 'ftol': 1e-12})
                
                if res_final.fun < best_f:
                    best_f = res_final.fun
                    best_x = res_final.x.copy()
                    
            except Exception:
                pass

        return best_x, best_f
