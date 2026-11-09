"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

class Optimizer:
    """Hybrid Powell + Adaptive 1D Polish Optimizer for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        # 300s limit, use 290s to be safe
        self.time_limit = 290.0
        
        d = self.dim
        N = self.budget
        
        # Global best tracking
        best_x = np.zeros(d)
        best_fx = float('inf')
        
        def eval_func(x):
            nonlocal best_x, best_fx
            if self.calls >= N:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            
            self.calls += 1
            fx = func(x)
            
            if np.isfinite(fx) and fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            return fx

        # Helper for 1D minimization along a direction
        def line_search_1d(center, direction, bracket_range=5.0):
            """
            Performs a 1D minimization along 'direction' starting from 'center'.
            Uses scipy's minimize_scalar with bounded method.
            """
            nonlocal best_x, best_fx
            
            def f1d(t):
                x_new = center + t * direction
                return eval_func(x_new)
            
            best_t = 0.0
            best_f = float('inf')
            
            try:
                # Adaptive bracket: start with a moderate range.
                # If the function is very flat, we might need a larger range,
                # but for polishing, the minimum should be close.
                res = minimize_scalar(f1d, bounds=(-bracket_range, bracket_range), method='bounded', options={'maxiter': 30, 'xatol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_t = res.x
                    x_candidate = center + best_t * direction
                    if best_f < best_fx:
                        best_fx = best_f
                        best_x = x_candidate.copy()
            except Exception:
                pass

            return best_t, best_f

        # 1. Coarse Global Sampling to identify the best basin
        rng = np.random.default_rng(42)
        
        # Define candidate points
        candidates = [np.zeros(d)]
        # Add random points
        for _ in range(8):
            candidates.append(rng.uniform(-4.5, 4.5, size=d))
        # Add points near boundaries
        candidates.append(np.array([4.0] * d))
        candidates.append(np.array([-4.0] * d))
        
        # Evaluate all candidates
        for c in candidates:
            if self.calls >= N or time.time() - self.start_time > self.time_limit:
                break
            try:
                eval_func(c)
            except StopIteration:
                break
            except Exception:
                pass

        # 2. Focused Local Optimization with Powell
        # Limit the budget for Powell to ensure time/budget is left for polishing
        if self.calls < N and time.time() - self.start_time < self.time_limit:
            try:
                # Use a significant portion of the budget, but leave some for polish
                # Powell is efficient, so 400 calls is plenty for rough convergence
                minimize(
                    eval_func,
                    best_x,
                    method='Powell',
                    options={
                        'maxiter': 200, # Limit iterations
                        'ftol': 1e-12,
                        'xatol': 1e-8,
                        'disp': False
                    }
                )
            except StopIteration:
                pass
            except Exception:
                pass

        # 3. Adaptive 1D Coordinate Descent Polish
        # This phase cleans up the residual error, especially in the narrow valley direction
        # if Powell didn't fully align or converge.
        if self.calls < N and time.time() - self.start_time < self.time_limit:
            # Determine how many rounds we can afford
            # Each round is d 1D searches. Each search takes ~10-30 calls.
            # Let's aim for 3-5 rounds if budget allows.
            max_rounds = 5
            for r in range(max_rounds):
                if self.calls >= N or time.time() - self.start_time > self.time_limit:
                    break
                
                for i in range(d):
                    if self.calls >= N or time.time() - self.start_time > self.time_limit:
                        break
                    
                    direction = np.zeros(d)
                    direction[i] = 1.0
                    
                    # Use a smaller bracket for polishing
                    # As we get closer, the step size needed is smaller
                    bracket = 2.0 / (r + 1)
                    _, _ = line_search_1d(best_x, direction, bracket_range=bracket)

        return best_x, best_fx
# EVOLVE-BLOCK-END
