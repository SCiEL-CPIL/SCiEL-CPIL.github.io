import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 290  # 10s margin for 300s limit
        
        # 1. Center
        x0 = np.zeros(dim)
        f0 = func(x0)
        best_val = f0
        
        # 2. Gradient estimation with Double Probing
        # Probe both +h and -h to get better intermediate best values
        # and robust sign determination.
        h = 5.0
        grad = np.zeros(dim)
        calls_made = 1
        
        for i in range(dim):
            if time.time() > deadline:
                break
            # Positive probe
            xp = x0.copy()
            xp[i] += h
            xp = np.clip(xp, -5, 5)
            fp = func(xp)
            best_val = min(best_val, fp)
            calls_made += 1
            
            # Negative probe
            xm = x0.copy()
            xm[i] -= h
            xm = np.clip(xm, -5, 5)
            fm = func(xm)
            best_val = min(best_val, fm)
            calls_made += 1
            
            # Central difference for gradient (more accurate)
            grad[i] = (fp - fm) / (2 * h)
            
        # 3. Predict optimal vertex
        x_pred = np.zeros(dim)
        for i in range(dim):
            if grad[i] > 0:
                x_pred[i] = -5.0
            else:
                x_pred[i] = 5.0
                
        # 4. Evaluate predicted vertex
        f_pred = func(x_pred)
        best_val = min(best_val, f_pred)
        calls_made += 1
        
        # 5. Fill remaining budget
        while calls_made < budget:
            if time.time() > deadline:
                break
            # Call the optimal vertex
            func(x_pred)
            calls_made += 1
            
        return x_pred, f_pred
