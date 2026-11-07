import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
        def __init__(self, budget, dim):
            self.budget = budget
            self.dim = dim

        def __call__(self, func):
            start_time = time.time()
            deadline = start_time + 290  # Safety margin for 300s limit
            
            best_x = None
            best_f = float('inf')
            
            # We will perform multiple local optimizations
            # Strategy: Divide budget among restarts
            # Let's try 15 restarts. 500/15 ~ 33 evals each.
            # L-BFGS-B might use more or less.
            # We need to track remaining budget.
            
            remaining_budget = self.budget
            num_restarts = 15
            
            # Generate diverse start points
            # 1. Random
            # 2. Grid-like
            # 3. Near known global min location (relative to box, but shift is unknown)
            # Since shift is unknown, we rely on random coverage.
            
            rng = np.random.default_rng(42)
            
            for i in range(num_restarts):
                if remaining_budget <= 0:
                    break
                
                # Check time
                if time.time() > deadline:
                    break
                
                # Choose start point
                # Mix of random and structured
                if i == 0:
                    x0 = np.zeros(self.dim)
                elif i == 1:
                    x0 = np.ones(self.dim) * 4.0
                elif i == 2:
                    x0 = np.ones(self.dim) * -4.0
                else:
                    x0 = rng.uniform(-5, 5, self.dim)
                
                x0 = np.clip(x0, -5, 5)
                
                # Run L-BFGS-B
                # We need to limit the number of function calls in minimize
                # scipy.optimize.minimize doesn't have a direct 'max_nfev' that stops exactly,
                # but we can pass options.
                # However, it's safer to wrap the function and count calls.
                
                calls_used = [0]
                max_calls_this_run = min(remaining_budget, 50) # Cap per run
                
                def wrapped_func(x):
                    if calls_used[0] >= max_calls_this_run:
                        raise StopIteration
                    calls_used[0] += 1
                    return func(x)
                
                try:
                    res = minimize(wrapped_func, x0, method='L-BFGS-B', 
                                   bounds=[(-5, 5)] * self.dim,
                                   options={'maxiter': 100, 'maxls': 20})
                    # Check if it stopped due to our limit or convergence
                    # res.nfev is the number of function calls made by minimize
                    # Note: wrapped_func might have been called more times if minimize called it in a way that bypassed our check? No, it calls wrapped_func.
                    # But if we raise StopIteration, minimize will catch it? No, it might propagate.
                    # Better to just check res.nfev.
                    
                    if res.success or 'Iteration limit' in res.message or 'Positive directional derivative' in res.message:
                        fx = func(res.x)
                        calls_used[0] += 1 # Count the final evaluation if not done by minimize? 
                        # Actually, minimize usually evaluates the final point.
                        # To be safe, we can just use res.fun if available, but we need to count it.
                        # Let's assume res.nfev is accurate.
                        
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x.copy()
                            
                        remaining_budget -= calls_used[0]
                        
                except StopIteration:
                    # Budget exhausted for this run
                    remaining_budget = 0
                    break
                except Exception as e:
                    # If minimize fails, try to use the best point found?
                    # Or just skip.
                    pass

            # If we have budget left, do a final polish on the best solution
            if best_x is not None and remaining_budget > 10 and time.time() < deadline:
                 # Run L-BFGS-B again from best_x with tighter settings
                 calls_used = [0]
                 max_calls_this_run = remaining_budget
                 
                 def wrapped_func(x):
                     if calls_used[0] >= max_calls_this_run:
                         raise StopIteration
                     calls_used[0] += 1
                     return func(x)
                 
                 try:
                     res = minimize(wrapped_func, best_x, method='L-BFGS-B', 
                                    bounds=[(-5, 5)] * self.dim,
                                    options={'maxiter': 100, 'maxls': 20, 'gtol': 1e-8})
                     if res.fun < best_f:
                         best_f = res.fun
                         best_x = res.x.copy()
                     remaining_budget -= calls_used[0]
                 except StopIteration:
                     pass
                 except Exception:
                     pass

            return best_x, best_f
