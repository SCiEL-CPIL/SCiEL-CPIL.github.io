import numpy as np
import scipy.optimize as so
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = 0
        self.time_limit = 280  # Safety margin for 300s limit

    def __call__(self, func):
        self.start_time = time.time()
        rng = np.random.default_rng(42)
        
        # 1. Initialization with Latin Hypercube Sampling
        n_init = min(30, self.budget // 5)
        # LHS in [0, 1]^d
        lhs = np.zeros((n_init, self.dim))
        for i in range(self.dim):
            perm = rng.permutation(n_init)
            lhs[:, i] = (perm + 0.5) / n_init
        
        # Map to [-5, 5]
        x_init = lhs * 10 - 5
        
        best_x = None
        best_f = np.inf
        points_evaluated = []
        
        # Evaluate initial points
        for x in x_init:
            if self.calls >= self.budget:
                break
            if time.time() - self.start_time > self.time_limit:
                break
            fx = func(x)
            self.calls += 1
            points_evaluated.append((x, fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # 2. Local Refinement Loop
        # We will try to improve the best point found so far using Nelder-Mead
        # If no improvement, we try a new random point or a perturbation.
        
        while self.calls < self.budget:
            if time.time() - self.start_time > self.time_limit:
                break
            
            # Try local search from the current best point
            # Limit function calls for this local search
            max_local_calls = min(100, self.budget - self.calls)
            
            # Nelder-Mead options
            options = {
                'maxiter': 50, # Internal iterations, but we track func calls
                'xatol': 1e-6,
                'fatol': 1e-6
            }
            
            # We need to wrap func to count calls and handle time
            def local_func(x):
                if self.calls >= self.budget:
                    return best_f # Return best known to stop? No, better to raise or return high value
                if time.time() - self.start_time > self.time_limit:
                    return best_f
                self.calls += 1
                return func(x)
            
            # Start from best_x
            # Nelder-Mead can be sensitive to initial simplex size
            # Default simplex is based on x0
            try:
                res = so.minimize(local_func, best_x, method='Nelder-Mead', options=options)
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
            
            # If local search didn't improve significantly or we are stuck,
            # generate a new candidate to escape local minima
            # Strategy: Sample a new point randomly, or perturb best_x
            # To balance exploration, let's sample a new point in the box
            # but weighted towards the center or best region?
            # For BBOB, uniform random is safe.
            
            # Generate a new random point
            x_new = rng.uniform(-5, 5, self.dim)
            
            # Evaluate it
            if self.calls < self.budget and time.time() - self.start_time < self.time_limit:
                fx_new = func(x_new)
                self.calls += 1
                if fx_new < best_f:
                    best_f = fx_new
                    best_x = x_new.copy()
            
            # Optional: If we have budget, we could try local search from x_new too
            # But Nelder-Mead from a random point might be less effective than from a good point.
            # The loop above already tried from best_x.
            # Let's just keep sampling and refining the global best.
            
            # To avoid infinite loop if Nelder-Mead uses 0 calls (unlikely) or fails
            # The loop condition `self.calls < self.budget` handles termination.
            # However, if Nelder-Mead consumes all remaining budget, the loop ends.
            # If Nelder-Mead consumes few calls, we sample a new point.
            
            # It's possible that Nelder-Mead doesn't improve. In that case, we just
            # sample a new random point and loop.
            # This creates a "Global Search with Local Polishing" pattern.
            
            # To make it more efficient, we can check if we've made progress recently.
            # If not, increase random sampling frequency.
            
        return best_x, best_f
