import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        n = self.dim
        budget = self.budget
        
        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        def try_func(x):
            nonlocal best_f, best_x, calls
            if calls >= budget or time.time() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initial sampling to find a good starting point
        # Use a mix of deterministic and random points
        rng = np.random.default_rng(42)
        
        # Start with origin and some axis-aligned points
        init_points = [np.zeros(n)]
        
        # Add points along axes
        for i in range(n):
            for sign in [-1, 1]:
                pt = np.zeros(n)
                pt[i] = sign * 4.0
                init_points.append(pt)
        
        # Add random points
        n_random = 20
        random_pts = rng.uniform(-5.0, 5.0, (n_random, n))
        for pt in random_pts:
            init_points.append(pt)
        
        # Evaluate initial points
        init_results = []
        for x in init_points:
            if calls >= budget or time.time() > deadline:
                break
            f = try_func(x)
            init_results.append((f, x))
            
        if not init_results:
            return best_x, best_f

        # Sort by function value
        init_results.sort(key=lambda x: x[0])
        
        # Phase 2: Coordinate descent with adaptive step sizes
        # Use the best few initial points as starting candidates
        n_restarts = min(5, len(init_results))
        
        for k in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break
                
            x = init_results[k][1].copy()
            f = init_results[k][0]
            
            # Adaptive step size for coordinate descent
            step_sizes = np.ones(n) * 0.5  # Initial step size
            
            # Coordinate descent iterations
            max_cd_iters = 100
            for cd_iter in range(max_cd_iters):
                if calls >= budget or time.time() > deadline:
                    break
                    
                improved = False
                for i in range(n):
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    # Try moving in positive and negative direction along dimension i
                    best_local_f = f
                    best_local_x = x.copy()
                    
                    # Try a few step sizes (shrinking)
                    for step_mult in [1.0, 0.5, 0.25, 0.1]:
                        step = step_sizes[i] * step_mult
                        
                        # Positive direction
                        x_plus = x.copy()
                        x_plus[i] += step
                        if np.all(x_plus >= -5.0) and np.all(x_plus <= 5.0):
                            f_plus = try_func(x_plus)
                            if f_plus < best_local_f:
                                best_local_f = f_plus
                                best_local_x = x_plus.copy()
                                improved = True
                                # Increase step size if we made progress
                                step_sizes[i] = min(step_sizes[i] * 1.5, 2.0)
                        
                        # Negative direction
                        x_minus = x.copy()
                        x_minus[i] -= step
                        if np.all(x_minus >= -5.0) and np.all(x_minus <= 5.0):
                            f_minus = try_func(x_minus)
                            if f_minus < best_local_f:
                                best_local_f = f_minus
                                best_local_x = x_minus.copy()
                                improved = True
                                # Increase step size if we made progress
                                step_sizes[i] = min(step_sizes[i] * 1.5, 2.0)
                    
                    if improved:
                        x = best_local_x
                        f = best_local_f
                    else:
                        # No improvement, shrink step size
                        step_sizes[i] *= 0.5
                        if step_sizes[i] < 1e-4:
                            step_sizes[i] = 1e-4
                
                if not improved:
                    break
        
        return best_x, best_f
