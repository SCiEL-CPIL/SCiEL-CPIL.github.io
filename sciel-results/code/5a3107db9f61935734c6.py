import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 290
        
        # Step 1: Evaluate center (origin)
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls_made = 1
        
        # Step 2: Evaluate 5 axis probes at distance 1.0 from origin
        h = 1.0
        probes = []
        for i in range(dim):
            if calls_made >= budget or time.time() > deadline:
                break
            xp = np.zeros(dim)
            xp[i] = h
            fp = func(xp)
            calls_made += 1
            probes.append(fp)
        
        # If we couldn't complete all probes, fallback to greedy fill
        if len(probes) < dim:
            x_best = x0.copy()
            f_best = f0
            while calls_made < budget:
                if time.time() > deadline:
                    break
                func(x_best)
                calls_made += 1
            return x_best, f_best
        
        # Step 3: Compute gradient estimates
        # f(x0 + h*e_i) - f0 ≈ h * w_i  =>  w_i ≈ (probes[i] - f0) / h
        w = (np.array(probes) - f0) / h
        
        # Step 4: Determine optimal corner
        # Minimize w^T x on [-5, 5]^dim
        # x_i* = -5 if w_i > 0, else +5
        x_opt = np.where(w > 0, -5.0, 5.0)
        
        # Step 5: Evaluate the optimal corner
        f_opt = func(x_opt)
        calls_made += 1
        
        # Step 6: Fill remaining budget with evaluations at the optimal corner
        while calls_made < budget:
            if time.time() > deadline:
                break
            func(x_opt)
            calls_made += 1
        
        return x_opt, f_opt
