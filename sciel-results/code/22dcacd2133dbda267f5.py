"""Optimizer for BBOB f9 (Rosenbrock rotated) using Powell's directional search."""
import numpy as np
import time


class Optimizer:
    """Powell's directional search with multi-start for rotated Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        def powell_search(x0, max_iters=100):
            """Powell's method for minimization."""
            nonlocal budget, best_x, best_f
            
            # Initialize directions: standard basis
            directions = [np.eye(dim, dtype=float)[i] for i in range(dim)]
            x = x0.copy()
            fx = func(x)
            budget -= 1
            
            for _ in range(max_iters):
                if time.time() > deadline - 5:
                    break
                if budget < 2 * dim:
                    break
                
                old_x = x.copy()
                old_fx = fx
                max_delta = 0.0
                worst_dir = 0
                
                # Line search along each direction
                for i in range(dim):
                    if budget < 20:
                        break
                    d = directions[i]
                    # Find optimal step along direction d
                    # Use a simple bracketing + bisection approach
                    # First find a bracket
                    step = 1.0
                    # Try to find a point that improves
                    best_step = 0.0
                    best_fx = fx
                    for trial in range(20):
                        if budget < 1:
                            break
                        x_try = x + step * d
                        x_try = np.clip(x_try, -5, 5)
                        fx_try = func(x_try)
                        budget -= 1
                        if fx_try < best_fx:
                            best_fx = fx_try
                            best_step = step
                        else:
                            break
                        step *= 0.7
                    
                    # If we found an improvement, refine with bisection
                    if best_step > 0 and best_fx < fx:
                        lo = best_step * 0.5 if best_step > 0 else 0.0
                        hi = best_step * 1.5
                        # Ensure we have a valid bracket
                        x_lo = np.clip(x + lo * d, -5, 5)
                        x_hi = np.clip(x + hi * d, -5, 5)
                        fx_lo = func(x_lo)
                        budget -= 1
                        fx_hi = func(x_hi)
                        budget -= 1
                        
                        # Bisection to find minimum in [lo, hi]
                        for _ in range(8):
                            if budget < 1:
                                break
                            mid = (lo + hi) / 2
                            x_mid = np.clip(x + mid * d, -5, 5)
                            fx_mid = func(x_mid)
                            budget -= 1
                            if fx_mid < fx_lo:
                                hi = mid
                                fx_hi = fx_mid
                            else:
                                lo = mid
                                fx_lo = fx_mid
                        
                        # Use the best point found
                        x = np.clip(x + best_step * d, -5, 5)
                        fx = best_fx
                        delta = fx - old_fx
                        if delta > max_delta:
                            max_delta = delta
                            worst_dir = i
                
                # Check convergence
                if np.linalg.norm(x - old_x) < 1e-8:
                    break
                
                # Update directions: replace worst direction with x - old_x
                delta_x = x - old_x
                if np.linalg.norm(delta_x) > 1e-10:
                    directions[worst_dir] = delta_x / np.linalg.norm(delta_x)
                
                # Update best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            return x, fx

        # Multi-start Powell search
        n_starts = 3
        for s in range(n_starts):
            if time.time() > deadline - 10:
                break
            if budget < 2 * dim + 10:
                break
            
            if s == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-4, 4, dim)
            
            x, fx = powell_search(x0, max_iters=50)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Final refinement if budget remains
        if budget > 2 * dim + 10 and best_x is not None:
            if time.time() > deadline - 10:
                pass
            else:
                x_final, fx_final = powell_search(best_x, max_iters=30)
                if fx_final < best_f:
                    best_f = fx_final
                    best_x = x_final.copy()

        if best_x is None:
            best_x = np.zeros(dim)

        return best_x, best_f
