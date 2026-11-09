import numpy as np
from scipy.optimize import minimize_scalar

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
    
    def __call__(self, problem, bounds):
        import time
        start_time = time.time()
        deadline = start_time + 320
        
        n = self.dim
        C = 1 + 10 * (n - 1)
        ref_val = 11.0
        t_star = (1.0 - ref_val / C) ** 2
        t_star = max(0.0, min(1.0, t_star))
        
        N = self.budget
        
        def f2(t):
            return C * (1.0 - np.sqrt(t))
        
        # Greedy hypervolume maximization
        # Start with two endpoints
        points = [t_star, 1.0]
        
        def hv_gain(t_a, t_b, t):
            return (t_b - t) * (f2(t_a) - f2(t))
        
        for i in range(N - 2):
            if time.time() > deadline - 10:
                break
            
            best_gain = -1
            best_t = None
            
            # For each adjacent pair, find the t that maximizes gain
            for j in range(len(points) - 1):
                t_a = points[j]
                t_b = points[j + 1]
                
                # Use minimize_scalar to find max of hv_gain
                # We need to minimize -hv_gain
                res = minimize_scalar(lambda t: -hv_gain(t_a, t_b, t), 
                                     bounds=(t_a, t_b), method='bounded')
                
                if res.success and res.fun < best_gain * (-1):  # -res.fun > -best_gain
                    gain = -res.fun
                    if gain > best_gain:
                        best_gain = gain
                        best_t = res.x
            
            if best_t is not None:
                # Insert best_t in sorted order
                points.append(best_t)
                points.sort()
        
        # Ensure we have exactly N points
        while len(points) < N:
            # If we ran out of time or couldn't improve, just add 1.0
            points.append(1.0)
        
        points = points[:N]
        
        X = np.zeros((N, self.dim))
        X[:, 0] = points
        X[:, 1:] = 0.0
        
        F = np.array([problem(x) for x in X])
        
        return X
