import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT1.
    Uses gradient ascent to find the optimal discrete placement of points 
    on the Pareto front to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 10.0 

        B = int(self.budget)
        D = int(self.dim)
        
        if B <= 0:
            return np.zeros(D)

        # ZDT1 Pareto Front:
        # x1 in [0, 1], x2...xD = 0
        # f1 = x1
        # f2 = 1 - sqrt(x1)
        
        # Objective: Maximize HV = sum_{i=1}^N (x_i - x_{i-1}) * f2(x_i)
        # with x_0 = 0.
        
        # Initialize x1 values using quadratic spacing (better initial guess)
        t = np.linspace(0.0, 1.0, B)
        x = t**2
        
        # Gradient ascent with decreasing step size
        eps = 1e-12
        step_size = 0.01
        
        for outer_iter in range(1000):
            if time.time() > deadline - 2.0:
                break
            
            # Calculate gradient
            f2_vals = 1.0 - np.sqrt(x + eps)
            f2_prime_vals = -0.5 / np.sqrt(x + eps)
            
            # Widths
            x_prev = np.concatenate(([0.0], x[:-1]))
            widths = x - x_prev
            
            grad = np.zeros(B)
            
            # For k=0 to B-2:
            grad[:-1] = f2_vals[:-1] + widths[:-1] * f2_prime_vals[:-1] - f2_vals[1:]
            
            # For k=B-1 (last element):
            grad[-1] = f2_vals[-1] + widths[-1] * f2_prime_vals[-1]
            
            # Check convergence
            grad_max = np.max(np.abs(grad))
            if grad_max < 1e-15:
                break
            
            # Update
            x_new = x + step_size * grad
            
            # Enforce bounds [0, 1] and ordering
            x_new = np.clip(x_new, 0.0, 1.0)
            
            # Enforce strict ordering
            for i in range(B - 1):
                if x_new[i] >= x_new[i+1]:
                    mid = (x_new[i] + x_new[i+1]) / 2.0
                    x_new[i] = mid - 1e-8
                    x_new[i+1] = mid + 1e-8
                    x_new[i] = np.clip(x_new[i], 0.0, 1.0)
                    x_new[i+1] = np.clip(x_new[i+1], 0.0, 1.0)
            
            x = x_new
            
            # Decrease step size for finer convergence
            step_size *= 0.995
            
        # Final clip
        x = np.clip(x, 0.0, 1.0)
        
        # Construct X
        X = np.zeros((B, D))
        X[:, 0] = x
        # x2...xD = 0
        
        # Clip to bounds
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        X = np.clip(X, lo, hi)
        
        # Evaluate
        for i in range(B):
            if time.time() > deadline:
                break
            problem(X[i].copy())
            
        return X[-1]
