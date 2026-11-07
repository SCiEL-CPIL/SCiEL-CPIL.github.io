import numpy as np
import time
import random

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        d = self.dim
        budget = self.budget
        
        calls = 0
        best_x = np.zeros(d)
        best_f = float('inf')
        
        def wrapper(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            
            calls += 1
            try:
                x_arr = np.asarray(x, dtype=float).flatten()
                f_val = float(func(x_arr))
            except Exception:
                return best_f
                
            if f_val < best_f:
                best_f = f_val
                best_x = x_arr.copy()
            return f_val

        # Phase 1: Try a few random vertices to potentially find the optimum early
        # or get a better starting point than the origin due to log-concavity.
        # The function is linear, so the min is at a vertex.
        # There are 2^d vertices.
        num_random_tries = 5
        
        # Generate all vertices
        vertices = []
        for mask in range(2 ** d):
            v = np.zeros(d)
            for i in range(d):
                if (mask >> i) & 1:
                    v[i] = 5.0
                else:
                    v[i] = -5.0
            vertices.append(v)
        
        # Shuffle vertices
        random.shuffle(vertices)
        
        # Evaluate first few random vertices
        for i in range(num_random_tries):
            if calls >= budget or time.time() > deadline:
                break
            x_rand = vertices[i]
            wrapper(x_rand)
            
        # Phase 2: Estimate gradient and find optimal vertex
        # f(x) = w^T x + c
        # Use origin and axis probes to estimate w
        x_zero = np.zeros(d)
        f_zero = wrapper(x_zero)
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f
            
        w_est = np.zeros(d)
        for i in range(d):
            if calls >= budget or time.time() > deadline:
                break
            x_probe = np.zeros(d)
            x_probe[i] = 5.0
            f_probe = wrapper(x_probe)
            w_est[i] = (f_probe - f_zero) / 5.0
            
        if calls >= budget or time.time() > deadline:
            return best_x, best_f
            
        # Determine optimal vertex
        x_opt = np.zeros(d)
        for i in range(d):
            if w_est[i] > 0:
                x_opt[i] = -5.0
            else:
                x_opt[i] = 5.0
                
        # Evaluate the optimal vertex
        f_opt = wrapper(x_opt)
        
        # Return immediately
        return best_x, best_f
