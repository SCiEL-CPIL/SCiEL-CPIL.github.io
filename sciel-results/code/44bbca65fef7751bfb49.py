import numpy as np
import time
from scipy.optimize import dual_annealing

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_val = np.inf
        self.best_x = None
        self.calls = 0
        self.start_time = 0

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 290  # Safety margin from 330s limit
        
        # Wrapper to count calls and update best
        def wrapped_func(x):
            self.calls += 1
            # Ensure x is a numpy array of correct shape for safety
            x = np.asarray(x, dtype=float)
            val = func(x)
            if val < self.best_val:
                self.best_val = val
                self.best_x = x.copy()
            return val

        # Bounds for dual_annealing: [-5, 5]^dim
        bounds = [(-5.0, 5.0)] * self.dim
        
        # Weierstrass is highly multimodal. dual_annealing uses simulated annealing 
        # with a local minimization (Nelder-Mead or Powell) at each step.
        # maxiter controls the number of iterations in the simulated annealing.
        # initial_temp and alpha control the cooling schedule.
        # maxfun is the total number of function evaluations allowed.
        
        # We have 500 evaluations. 
        # dual_annealing might use them all. 
        # We need to ensure we don't exceed 500 calls or the time limit.
        
        # To maximize AOCC, we want to find a good solution quickly.
        # dual_annealing is a global method, so it might spend time exploring.
        # However, it is generally more robust for Weierstrass than random restarts of NM.
        
        # Let's try to run dual_annealing with a limit on maxfun slightly below budget
        # to leave some room, or just let it run until it finishes or hits maxiter.
        # The callback in dual_annealing is not as flexible for stopping mid-iteration 
        # based on external time/call counts as easily as in minimize, 
        # but we can monitor time in the wrapped_func? No, wrapped_func doesn't have access to stop signal easily without global state.
        
        # Actually, dual_annealing does not support a callback to stop early.
        # It will run until maxiter or maxfun is reached.
        # maxfun=500 is safe for the call budget.
        # Time limit: Weierstrass in 5D with 500 evals should be very fast (<< 1s).
        # So time limit is not a bottleneck for the computation itself, only for the wrapper.
        
        # One issue: dual_annealing might not update self.best_val as frequently 
        # if it uses internal caching or if the local minimizer steps are not 
        # counted individually in a way that triggers our best update? 
        # No, wrapped_func is the objective, so every eval goes through it.
        
        # Let's run dual_annealing.
        try:
            res = dual_annealing(
                wrapped_func,
                bounds=bounds,
                maxiter=100,       # Number of iterations in the SA loop
                maxfun=500,        # Max function evaluations
                initial_temp=5230.0,
                restart_temp_ratio=2e-5,
                visit=2.62,
                seed=42
            )
        except Exception:
            pass

        # If we have calls left and time, we could do a final polish with Nelder-Mead
        # starting from the best point found by dual_annealing.
        if self.best_x is not None and self.calls < self.budget:
            # Check time
            if time.time() < deadline:
                # Use remaining budget for a local polish
                remaining = self.budget - self.calls
                if remaining > 10:
                    # Define a new wrapper? No, reuse wrapped_func.
                    # But we need to be careful not to exceed budget.
                    # dual_annealing already used up to 500. 
                    # If it used less, we can polish.
                    # However, dual_annealing's maxfun=500 means it might use all 500.
                    # If it used all 500, remaining is 0.
                    
                    # Let's check if we actually have budget left.
                    # If dual_annealing stopped early due to convergence, we have budget left.
                    
                    # We can try a quick Nelder-Mead polish.
                    # But we need a callback to stop if we hit the budget.
                    
                    def nm_callback(xk):
                        return self.calls >= self.budget or time.time() > deadline
                    
                    try:
                        # Use scipy minimize for polish
                        from scipy.optimize import minimize
                        minimize(
                            wrapped_func,
                            self.best_x,
                            method='Nelder-Mead',
                            options={'maxiter': remaining, 'xatol': 1e-10, 'fatol': 1e-12},
                            callback=nm_callback
                        )
                    except Exception:
                        pass

        return self.best_x, self.best_val
