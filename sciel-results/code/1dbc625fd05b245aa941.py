import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        d = self.dim
        
        # Step 1: Evaluate at origin
        x0 = np.zeros(d)
        f0 = func(x0)
        best_x = x0.copy()
        best_fx = f0
        
        # Check time
        if time.time() > deadline:
            return best_x, best_fx
            
        # Step 2: Evaluate at 5 * e_i for each dimension to estimate gradient
        w_est = np.zeros(d)
        for i in range(d):
            if time.time() > deadline:
                break
            xi = np.zeros(d)
            xi[i] = 5.0
            fi = func(xi)
            w_est[i] = (fi - f0) / 5.0
            
            # Update best if this point is better
            if fi < best_fx:
                best_x = xi.copy()
                best_fx = fi
        
        # Step 3: Construct optimal vertex based on estimated gradient signs
        x_opt = np.zeros(d)
        for i in range(d):
            if w_est[i] > 1e-8:
                x_opt[i] = -5.0
            elif w_est[i] < -1e-8:
                x_opt[i] = 5.0
            else:
                x_opt[i] = 0.0
        
        # Step 4: Evaluate the optimal vertex
        if time.time() > deadline:
            return best_x, best_fx
            
        f_opt = func(x_opt)
        if f_opt < best_fx:
            best_x = x_opt.copy()
            best_fx = f_opt
        else:
            # If the estimated vertex wasn't better, try the opposite vertex
            x_opposite = -x_opt
            if time.time() > deadline:
                return best_x, best_fx
            f_opposite = func(x_opposite)
            if f_opposite < best_fx:
                best_x = x_opposite.copy()
                best_fx = f_opposite
        
        # Step 5: Use remaining budget to verify with a few random vertices
        rng = np.random.default_rng(42)
        calls_made = 7
        for _ in range(min(self.budget - calls_made, 50)):
            if time.time() > deadline:
                break
            x_rand = np.array([5.0 * (1 if rng.random() > 0.5 else -1) for _ in range(d)])
            f_rand = func(x_rand)
            if f_rand < best_fx:
                best_x = x_rand.copy()
                best_fx = f_rand
        
        return best_x, best_fx
