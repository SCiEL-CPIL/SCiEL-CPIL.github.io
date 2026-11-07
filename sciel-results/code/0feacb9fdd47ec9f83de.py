import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = 0.0
        self.time_limit = 290.0 # 10s margin from 300s

    def __call__(self, func):
        self.start_time = time.time()
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        
        def eval_func(x):
            self.calls += 1
            # Clip to box [-5, 5]
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            if f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x_clipped.copy()
            return f_val

        # 1. Initial Random Sampling
        n_init = 10
        if self.budget >= n_init:
            # Generate random points
            # Using standard normal might be better if distribution is unknown, 
            # but uniform covers the box. Let's use uniform.
            rng = np.random.default_rng(42)
            points = rng.uniform(-5, 5, size=(n_init, self.dim))
            for p in points:
                if self.calls >= self.budget:
                    break
                eval_func(p)
            
            # If we have budget left, do local optimizations
            if self.calls < self.budget:
                # Sort points by function value
                # We need to store the values. Let's re-evaluate or store.
                # Storing is better.
                vals = []
                for p in points:
                    if self.calls < self.budget:
                        vals.append(eval_func(p))
                    else:
                        break
                
                # Sort indices by value
                # Note: eval_func already updated best_f, but we need to know which point was which.
                # Actually, the loop above re-evaluates. This wastes calls.
                # Better: Evaluate once, store (x, f).
                
        # Let's rewrite the logic to be cleaner.
        
        # Reset state
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        
        def safe_eval(x):
            self.calls += 1
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            if f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x_clipped.copy()
            return f_val

        # Phase 1: Multi-Start
        n_starts = 5
        candidates = []
        
        # Generate diverse starting points
        # Mix of uniform random and maybe some structured points?
        # Uniform is good for BBOB.
        rng = np.random.default_rng(123)
        
        for i in range(n_starts):
            if self.calls >= self.budget:
                break
            x0 = rng.uniform(-5, 5, self.dim)
            f0 = safe_eval(x0)
            candidates.append((x0, f0))
            
        # Sort candidates by fitness
        candidates.sort(key=lambda x: x[1])
        
        # Phase 2: Local Optimization
        # For each of the top candidates, run Nelder-Mead
        # Budget allocation:
        # Remaining budget after starts: self.budget - self.calls
        # Let's try to optimize the top 3 candidates.
        # Each optimization gets a chunk of the remaining budget.
        
        remaining_budget = self.budget - self.calls
        if remaining_budget > 0 and len(candidates) > 0:
            
            # We will try to optimize the best candidate first.
            # If it improves significantly, we might stop or continue with less budget for others.
            
            for idx, (x_start, f_start) in enumerate(candidates):
                if self.calls >= self.budget:
                    break
                
                # Determine max iterations for this local run
                # Nelder-Mead in 5D uses ~3-4 evals per iteration.
                # Let's reserve some budget for the next candidates.
                # Simple strategy: give each local run 1/3 of remaining budget?
                # Or fixed 80 iterations?
                # 5D Nelder-Mead: 6 points in simplex.
                # 80 iterations * 3 evals/iter = 240 evals.
                # If we have 500 total, and used 5 for starts, we have 495.
                # We can afford 2 local runs of ~200 evals each.
                
                # Let's try 2 local runs.
                if idx >= 2:
                    break
                
                # Calculate max_iter
                # Leave 10% of remaining for safety/polish
                alloc = int((self.budget - self.calls) * 0.4) # 40% of remaining
                if alloc < 10:
                    break
                max_iter = max(10, alloc // 3) # Approx 3 evals per iter
                
                # Prepare initial simplex
                # x0 and x0 + delta * e_i
                delta = 0.5
                simplex = [x_start]
                for i in range(self.dim):
                    x_new = x_start.copy()
                    x_new[i] += delta
                    simplex.append(np.clip(x_new, -5, 5))
                
                # Run minimize
                # We need to pass the function that counts calls.
                # scipy minimize will call safe_eval.
                
                try:
                    res = minimize(
                        safe_eval, 
                        x_start, 
                        method='Nelder-Mead',
                        options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-5, 'initial_simplex': np.array(simplex)}
                    )
                except Exception as e:
                    # If minimize fails, we just continue
                    pass
                
                # After minimize, check if we should stop
                if self.calls >= self.budget:
                    break

        # Phase 3: Polish
        # If we have budget left, do a fine polish on the global best
        if self.calls < self.budget and self.best_x is not None:
            # Use Powell's method? Or just a smaller Nelder-Mead?
            # Powell is often better for final polish.
            try:
                # Powell requires more careful handling of bounds, but scipy handles it by projecting?
                # Actually, Powell in scipy doesn't handle bounds directly either.
                # Let's stick to Nelder-Mead with a very small step size.
                delta = 0.01
                simplex = [self.best_x]
                for i in range(self.dim):
                    x_new = self.best_x.copy()
                    x_new[i] += delta
                    simplex.append(np.clip(x_new, -5, 5))
                
                max_iter = (self.budget - self.calls) // 3
                if max_iter > 0:
                    res = minimize(
                        safe_eval, 
                        self.best_x, 
                        method='Nelder-Mead',
                        options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6, 'initial_simplex': np.array(simplex)}
                    )
            except Exception:
                pass

        return self.best_x, self.best_f
