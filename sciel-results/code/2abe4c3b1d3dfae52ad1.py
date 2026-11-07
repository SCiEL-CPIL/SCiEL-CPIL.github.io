import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        
        n = self.dim
        budget = self.budget
        
        calls = 0
        best_f = np.inf
        best_x = None
        
        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return 1e10
            calls += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = np.array(x)
            return fx

        # Phase 1: Deterministic Global Sampling
        # This ensures early progress by covering the domain systematically.
        # 1. Center
        # 2. 32 Vertices of [-5, 5]^5
        # 3. 10 Axis midpoints (±2.5 on each axis)
        # 4. 20 Random points for better coverage
        
        points_to_eval = []
        
        # Center
        points_to_eval.append(np.zeros(n))
        
        # Vertices
        for i in range(2**n):
            vertex = []
            for j in range(n):
                if (i >> j) & 1:
                    vertex.append(5.0)
                else:
                    vertex.append(-5.0)
            points_to_eval.append(np.array(vertex))
            
        # Axis midpoints
        for i in range(n):
            for sign in [-1, 1]:
                mid = np.zeros(n)
                mid[i] = sign * 2.5
                points_to_eval.append(mid)
        
        # Add some random points for better coverage
        rng = np.random.default_rng(42)
        for _ in range(20):
            points_to_eval.append(rng.uniform(-5, 5, n))
        
        # Evaluate these points
        for p in points_to_eval:
            if calls >= budget or time.time() > deadline - 10:
                break
            wrapped_func(p)
            
        # If we didn't find any points (shouldn't happen), fallback to random
        if best_x is None:
            rng = np.random.default_rng(42)
            best_x = rng.uniform(-5, 5, n)
            best_f = wrapped_func(best_x)

        # Phase 2: Multi-start Local Optimization with Nelder-Mead
        # Use the top few points from Phase 1 as starting points for local optimization
        
        if calls < budget and time.time() < deadline - 5:
            # Sort points by function value (we need to re-evaluate to get values)
            # Actually, we can just use the best point and a few others
            # Let's use a simple approach: run Nelder-Mead from the best point
            # and from a few other promising points
            
            # First, let's do a simple hill climb from the best point
            current_x = best_x.copy()
            current_f = best_f
            
            # Define a sequence of step sizes to try
            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0005]
            
            for step in step_sizes:
                if calls >= budget or time.time() > deadline - 5:
                    break
                
                improved = True
                while improved and calls < budget and time.time() < deadline - 5:
                    improved = False
                    for i in range(n):
                        if calls >= budget or time.time() > deadline - 5:
                            break
                        for sign in [-1, 1]:
                            if calls >= budget or time.time() > deadline - 5:
                                break
                            new_x = current_x.copy()
                            new_x[i] += sign * step
                            new_x = np.clip(new_x, -5.0, 5.0)
                            
                            if np.allclose(new_x, current_x):
                                continue
                                
                            new_f = wrapped_func(new_x)
                            
                            if new_f < current_f - 1e-10:
                                current_x = new_x
                                current_f = new_f
                                improved = True
                                
            # Update best if we improved
            if current_f < best_f:
                best_f = current_f
                best_x = current_x

        # Phase 3: Nelder-Mead from the best point found so far
        if calls < budget and time.time() < deadline - 5:
            try:
                from scipy.optimize import minimize
                
                def objective(x):
                    return wrapped_func(x)
                
                # Run Nelder-Mead from the best point
                result = minimize(
                    objective, 
                    best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'fatol': 1e-10, 'xatol': 1e-10}
                )
                
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x
                    
            except Exception:
                pass

        # Phase 4: Final polish with smaller steps
        if calls < budget and time.time() < deadline - 2:
            current_x = best_x.copy()
            current_f = best_f
            
            # Very fine steps for final polish
            fine_steps = [0.001, 0.0005, 0.0001, 0.00005]
            
            for step in fine_steps:
                if calls >= budget or time.time() > deadline - 2:
                    break
                
                improved = True
                while improved and calls < budget and time.time() < deadline - 2:
                    improved = False
                    for i in range(n):
                        if calls >= budget or time.time() > deadline - 2:
                            break
                        for sign in [-1, 1]:
                            if calls >= budget or time.time() > deadline - 2:
                                break
                            new_x = current_x.copy()
                            new_x[i] += sign * step
                            new_x = np.clip(new_x, -5.0, 5.0)
                            
                            if np.allclose(new_x, current_x):
                                continue
                                
                            new_f = wrapped_func(new_x)
                            
                            if new_f < current_f - 1e-12:
                                current_x = new_x
                                current_f = new_f
                                improved = True
                                
            # Update best if we improved
            if current_f < best_f:
                best_f = current_f
                best_x = current_x

        return best_x, best_f
