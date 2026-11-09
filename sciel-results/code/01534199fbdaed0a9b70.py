import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        # Initial point
        x0 = np.zeros(self.dim)
        
        # Counter for function calls
        call_count = [0]
        
        def counted_func(x):
            call_count[0] += 1
            return func(x)

        # Phase 1: Powell
        # Use Powell to find the basin
        # Powell is good for anisotropic functions
        try:
            res = scipy.optimize.minimize(
                counted_func, x0, method='Powell',
                options={'maxiter': 100, 'disp': False, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            x_best = res.x
            f_best = res.fun
        except Exception:
            x_best = x0
            f_best = func(x0)
            call_count[0] += 1

        # Phase 2: 1D Coordinate Descent Polish
        # Use remaining budget
        remaining = self.budget - call_count[0]
        if remaining > 0:
            # We can do a few rounds of coordinate descent
            # Each round costs 5 * (cost of 1D min)
            # 1D min with bounded scalar usually takes ~10-20 evals
            # So 1 round ~ 50-100 evals
            # We have ~150-200 evals left if Powell used 300.
            # Let's do a loop that checks budget and time.
            
            while call_count[0] < self.budget and (time.time() - self.start_time) < 280:
                improved = False
                for i in range(self.dim):
                    if call_count[0] >= self.budget or (time.time() - self.start_time) > 290:
                        break
                    
                    # Define 1D function
                    def f_1d(t):
                        x_temp = x_best.copy()
                        x_temp[i] = t
                        return counted_func(x_temp)
                    
                    # Bounded scalar minimization
                    # Use a small maxiter to save calls
                    try:
                        res_1d = scipy.optimize.minimize_scalar(
                            f_1d, bounds=(-5, 5), method='bounded',
                            options={'maxiter': 20, 'xatol': 1e-6}
                        )
                        if res_1d.success and res_1d.fun < f_best - 1e-12:
                            x_best[i] = res_1d.x
                            f_best = res_1d.fun
                            improved = True
                    except Exception:
                        pass
                
                # If no improvement in a full cycle, we might be stuck or converged
                # But for Bent Cigar, small improvements might still be possible.
                # If no improvement, break to avoid wasting budget on noise?
                # Or continue to ensure we use the budget?
                # The score is AOCC, so early progress matters.
                # If we are converged, further calls don't help AOCC if f doesn't drop.
                # But if we are stuck in a local min (unlikely for f12 which is unimodal in the valley), 
                # restarts might help. But f12 is unimodal.
                if not improved:
                    # If we haven't used much budget, maybe do a random restart?
                    # But f12 is unimodal, so Powell should have found the global min.
                    # If improved is False, we are likely at the minimum.
                    break

        return x_best, f_best
