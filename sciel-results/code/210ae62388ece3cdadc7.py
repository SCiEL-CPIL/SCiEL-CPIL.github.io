import numpy as np
import time
from scipy.optimize import minimize_scalar

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        n_calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')

        def call_f(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget:
                return None
            x = np.asarray(x)
            if x.shape != (dim,):
                x = np.clip(x, -5.0, 5.0).reshape(dim)
            n_calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def powell_method(x0, max_iters=100):
            """
            Powell's method for derivative-free optimization.
            Maintains a set of conjugate directions.
            """
            nonlocal n_calls, best_x, best_f
            
            if n_calls >= budget:
                return best_x, best_f
            
            # Initialize directions as unit vectors
            directions = np.eye(dim)
            x = x0.copy()
            fx = call_f(x)
            if fx is None:
                return best_x, best_f
            
            for _ in range(max_iters):
                if time.time() > deadline - 5:
                    break
                if n_calls >= budget:
                    break
                
                x_old = x.copy()
                f_old = fx
                f_min_axis = float('inf')
                i_min = 0
                
                for i in range(dim):
                    if time.time() > deadline - 5:
                        return best_x, best_f
                    if n_calls >= budget:
                        return best_x, best_f
                    
                    # Minimize along direction i
                    def f_1d(t):
                        # t is the step size along direction i
                        x_new = x + t * directions[i]
                        x_new = np.clip(x_new, -5.0, 5.0)
                        return call_f(x_new)
                    
                    if n_calls >= budget:
                        return best_x, best_f
                    
                    # Use bounded scalar minimization
                    # The step size t can be roughly in [-10, 10] given the box size
                    res = minimize_scalar(f_1d, bounds=(-10, 10), method='bounded', options={'maxiter': 10, 'xatol': 1e-3})
                    
                    if n_calls >= budget:
                        return best_x, best_f
                        
                    if res.success:
                        x = x + res.x * directions[i]
                        x = np.clip(x, -5.0, 5.0)
                        fx = res.fun
                        if fx < f_min_axis:
                            f_min_axis = fx
                            i_min = i
                
                # Check convergence
                if abs(f_old - f_min_axis) < 1e-6:
                    break
                
                # Update directions
                # The new direction is the difference between the start and end of the cycle
                new_dir = x - x_old
                norm = np.linalg.norm(new_dir)
                if norm < 1e-12:
                    break
                new_dir = new_dir / norm
                
                # Replace the worst direction (the one that gave the smallest improvement)
                directions[i_min] = new_dir
            
            return best_x, best_f

        # Strategy:
        # 1. Run Powell's method from the center.
        # 2. If budget remains, perform random restarts with Powell's method.
        
        # Initial run
        powell_method(np.zeros(dim), max_iters=50)

        # Restart logic
        restarts = 0
        max_restarts = 5
        
        while restarts < max_restarts and n_calls < budget - 20 and time.time() < deadline - 10:
            # Generate a random start point
            x_rand = rng.uniform(-4, 4, dim)
            
            # Run Powell with a limited budget
            # Powell uses ~10 calls per iteration (1 per direction * 5 dirs + overhead)
            # Let's be conservative: ~10-20 calls per iteration
            remaining = budget - n_calls
            max_iters_this = min(20, remaining // 15)
            
            if max_iters_this < 2:
                break
                
            powell_method(x_rand, max_iters=max_iters_this)
            restarts += 1

        # Final polish: If we have a few calls left, do a very fine-grained local search
        remaining = budget - n_calls
        if remaining > 10 and time.time() < deadline - 2:
            x = best_x.copy()
            fx = best_f
            sigma = 0.01
            for _ in range(remaining):
                if time.time() > deadline - 1:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = call_f(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                else:
                    sigma *= 0.9
                    sigma = max(sigma, 1e-5)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
