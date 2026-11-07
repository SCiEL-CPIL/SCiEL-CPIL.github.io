"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-Start Nelder-Mead optimizer for BBOB f15 (Rastrigin rotated)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Multi-start strategy
        # For n=5, a small grid or random set of starts is effective.
        # We will use a combination of center, corners, and random points.
        
        # Generate initial points
        rng = np.random.default_rng(42)
        
        # 1. Center
        # 2. Random points
        # We want to distribute the budget. 
        # Let's use 10-20 starts.
        n_starts = 20
        
        # Create a list of starting points
        starts = []
        # Center
        starts.append(np.zeros(n))
        
        # Random points
        for _ in range(n_starts - 1):
            p = rng.uniform(-4.5, 4.5, n)
            starts.append(p)
            
        # Shuffle starts to ensure diversity in budget allocation if we stop early? 
        # No, we should probably prioritize points that seem promising? 
        # But we don't know. Let's just iterate.
        
        # However, Nelder-Mead is fast. 
        # Let's allocate budget evenly? Or adaptively?
        # Simple approach: Give each start a fixed budget slice.
        # If a start converges quickly, we can stop it and give remaining to others?
        # For simplicity and robustness: Fixed budget per start.
        
        budget_per_start = max(10, budget // n_starts)
        
        best_x = None
        best_f = np.inf
        
        # Pre-compute some points? No, let's just run.
        
        for x0 in starts:
            if budget <= 0:
                break
                
            # Define a wrapper to count calls and handle clipping if necessary (though func clips)
            # We need to be careful with the budget.
            # scipy.optimize.minimize doesn't allow easy early termination based on an external counter 
            # without a callback that raises an exception or checks a flag.
            # We can use a callback to check the budget.
            
            call_count = [0]
            limit = min(budget_per_start, budget)
            
            def callback(xk):
                call_count[0] += 1
                # If we exceed the limit for this start, we want to stop.
                # Raising an exception is a hack but effective for termination in SciPy.
                # Alternatively, we can just let it run and check afterwards, but that wastes calls.
                # Better: use maxiter? 
                # Nelder-Mead maxiter is number of iterations, not function calls.
                # Each iteration uses ~n+1 calls? No, it varies.
                # Let's just use a custom loop or accept the waste?
                # "More than 500 calls ... makes the whole result invalid".
                # So we MUST stop at 500.
                # If we use scipy, we can't easily interrupt it mid-iteration if it's in C code.
                # However, n=5 is small. One Nelder-Mead iteration is ~6-10 calls.
                # If limit is 25 calls, that's ~3-4 iterations.
                # Let's just run with a maxiter that is safe and check the global budget after.
                # But if we overshoot the *global* budget, we are invalid.
                # So we must track global budget.
                
                # Actually, the safest way with SciPy is to NOT use the internal loop for budget control 
                # if we can't guarantee stopping. 
                # But for n=5, Nelder-Mead is very fast. 
                # Let's estimate: 500 calls total. 
                # If we do 20 starts, 25 calls each. 
                # If a start takes 30 calls, we might overshoot.
                
                # Let's refine: We will implement a manual Nelder-Mead or use a simpler method?
                # No, the directive says "design a new program... from an approach family".
                # Nelder-Mead is the approach. 
                # To ensure safety, we can implement a simple Nelder-Mead ourselves to have full control over the call count.
                # This is safer for the "invalid result" constraint.
                
                return False # Don't stop inside callback, we'll manage outside.
            
            # Let's implement a simple Nelder-Mead to be safe with the budget.
            # Or, we can just use scipy but be very conservative with maxiter.
            # Let's stick to the directive: "design a new program". A custom implementation is cleaner for strict budget control.
            
            # Simple Nelder-Mead Implementation
            # Initial simplex
            alpha = 0.05 # step size for initial simplex
            simplex = [x0 + alpha * rng.standard_normal(n) for _ in range(n + 1)]
            # Ensure x0 is in the simplex
            simplex[0] = x0.copy()
            
            f_simplex = [func(s) for s in simplex]
            budget -= (n + 1)
            
            if budget <= 0:
                # Update best
                min_idx = np.argmin(f_simplex)
                if f_simplex[min_idx] < best_f:
                    best_f = f_simplex[min_idx]
                    best_x = simplex[min_idx].copy()
                break
            
            # Parameters
            alpha_nm = 1.0  # reflection
            gamma_nm = 2.0  # expansion
            rho_nm = 0.5    # contraction
            sigma_nm = 0.5  # shrink
            
            max_iter = 50 # Safety cap on iterations, not calls
            
            for _ in range(max_iter):
                if budget <= 0:
                    break
                    
                # Sort simplex
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha_nm * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                budget -= 1
                
                if fr < f_simplex[0]:
                    # Expansion
                    xe = centroid + gamma_nm * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    budget -= 1
                    
                    if fe < fr:
                        simplex[-1] = xe
                        f_simplex[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_simplex[-1] = fr
                elif fr < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    f_simplex[-1] = fr
                else:
                    # Contraction
                    # Inside contraction if fr < f_simplex[-1]
                    if fr < f_simplex[-1]:
                        xc = centroid + rho_nm * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = func(xc)
                        budget -= 1
                        
                        if fc <= fr:
                            simplex[-1] = xc
                            f_simplex[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_simplex[i] = func(simplex[i])
                                budget -= 1
                    else:
                        # Outside contraction
                        xc = centroid - rho_nm * (centroid - simplex[-1])
                        xc = np.clip(xc, -5, 5)
                        fc = func(xc)
                        budget -= 1
                        
                        if fc <= fr:
                            simplex[-1] = xc
                            f_simplex[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_simplex[i] = func(simplex[i])
                                budget -= 1
                                
                # Update best
                min_idx = np.argmin(f_simplex)
                if f_simplex[min_idx] < best_f:
                    best_f = f_simplex[min_idx]
                    best_x = simplex[min_idx].copy()
                    
                # Check for convergence (optional, but good for speed)
                # If simplex is very small, break
                if np.max(f_simplex) - np.min(f_simplex) < 1e-8:
                    break
            
        return best_x, best_f
# EVOLVE-BLOCK-END
