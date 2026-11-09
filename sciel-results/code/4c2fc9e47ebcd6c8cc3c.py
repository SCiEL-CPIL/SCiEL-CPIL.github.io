import numpy as np
import time
import math
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        eval_count = 0
        
        def check_deadline():
            return time.time() < deadline - 2.0
        
        def eval_point(x):
            nonlocal best_x, best_f, eval_count
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)
        
        # --- Phase 1: Random Sampling ---
        # f11 is multimodal. We need to find the basin of the global minimum.
        n_init = min(20, max(5, int(self.budget * 0.05)))
        
        for i in range(n_init):
            if not check_deadline() or eval_count >= self.budget:
                break
            x = rng.uniform(-5.0, 5.0, self.dim)
            eval_point(x)
            
        if eval_count >= self.budget or not check_deadline():
            return best_x, best_f

        # --- Phase 2: Local Refinement using Nelder-Mead ---
        # Use fast local search to converge to local optima.
        # Nelder-Mead is derivative-free and fast for low dimensions.
        
        # We will perform multiple local searches.
        # Start from the best point found in Phase 1.
        # Then, if budget remains, try other promising points or random restarts.
        
        # Strategy:
        # 1. Run Nelder-Mead from best_x.
        # 2. If it improves, great.
        # 3. If budget remains, try Nelder-Mead from a few other initial points
        #    (e.g., points that were close to best in Phase 1, or random points).
        
        # To manage budget strictly, we wrap the objective function.
        
        def make_obj_wrapper(x0):
            # Nelder-Mead can make many function calls.
            # We need to stop if we hit budget or deadline.
            def obj(x):
                if eval_count >= self.budget or not check_deadline():
                    # Return a large value to stop optimization
                    return 1e10
                return eval_point(x)
            return obj

        # We'll try a few local optimizations.
        # The number of local optimizations depends on remaining budget.
        # Each Nelder-Mead run might take ~50-200 evaluations depending on convergence.
        
        # Let's reserve some budget for a final polish phase.
        # But first, let's just try to optimize from the best point.
        
        # List of starting points for local search
        start_points = [best_x.copy()]
        
        # Add a few random points if we have enough budget
        if self.budget - eval_count > 100:
            for _ in range(3):
                start_points.append(rng.uniform(-5.0, 5.0, self.dim))
        
        for sp in start_points:
            if eval_count >= self.budget or not check_deadline():
                break
            
            # If the starting point is not better than current best, maybe skip?
            # Actually, Nelder-Mead might find a better basin.
            # But if sp is far from best, it might be a waste.
            # Let's just run it.
            
            obj = make_obj_wrapper(sp)
            
            try:
                res = minimize(
                    obj, 
                    sp, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 200, 
                        'xatol': 1e-4, 
                        'fatol': 1e-4,
                        'adaptive': True
                    }
                )
            except Exception:
                pass
            
            # Check if the result is better
            # Note: eval_point already updates best_x/best_f if it finds something better.
            # But res.x might be the best found by the local search, even if it didn't
            # beat the global best during the run (unlikely if obj returns real values).
            # However, if the local search got stuck, res.x might be worse than sp.
            # We rely on eval_point to track the global best.
            
            # If the local search finished and found a point, ensure we evaluate it?
            # eval_point is called inside obj, so best_x is updated if any point was better.
            
        # --- Phase 3: Final Polish with (1+1)-ES ---
        # Use remaining budget for fine-grained search around the global best.
        
        remaining = self.budget - eval_count
        if remaining > 0 and check_deadline():
            
            x_cur = best_x.copy()
            f_cur = best_f
            
            # Adaptive sigma
            sigma = 0.05
            no_improve = 0
            stall_limit = 10
            
            while remaining > 0 and check_deadline():
                y = np.clip(x_cur + sigma * rng.standard_normal(self.dim), -5.0, 5.0)
                fy = eval_point(y)
                remaining -= 1
                
                if fy < f_cur:
                    x_cur = y
                    f_cur = fy
                    no_improve = 0
                    sigma *= 1.05
                else:
                    no_improve += 1
                    sigma *= 0.95
                    
                    if no_improve >= stall_limit:
                        # Try a small random jump
                        if remaining > 5:
                            y_jump = np.clip(x_cur + 0.5 * rng.standard_normal(self.dim), -5.0, 5.0)
                            fy_jump = eval_point(y_jump)
                            remaining -= 1
                            if fy_jump < f_cur:
                                x_cur = y_jump
                                f_cur = fy_jump
                                sigma = 0.1
                                no_improve = 0
                            else:
                                sigma = 0.01
                                # If still stuck, break
                                if sigma < 1e-3:
                                    break
                        else:
                            break
                
                if sigma < 1e-4:
                    break

        return best_x, best_f
