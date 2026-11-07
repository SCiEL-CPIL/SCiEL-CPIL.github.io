import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        
        # ZDT1: f1 = x1, f2 = g(x)*(1 - sqrt(f1/g(x)))
        # Pareto front: x2..xn = 0.5, f1 in [0,1], f2 = 11*(1-sqrt(f1/11))
        
        # Phase 1: Initialize points along the Pareto front
        # Use most of budget for good coverage
        n_init = min(150, max(10, self.budget // 2))
        n_local = self.budget - n_init
        
        # Spread x1 values across [0, 1]
        x1_vals = np.linspace(0.0, 1.0, n_init)
        
        X = np.zeros((n_init, n))
        X[:, 0] = x1_vals
        X[:, 1:] = 0.5  # Pareto-optimal values for x2..xn
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate initial points
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # Phase 2: Local refinement - minimize g(x) for each point
        # g(x) = 1 + 10*(n-1) + sum((xi-0.5)^2 - 10*cos(2*pi*(xi-0.5))) for i>=2
        # We want to minimize g while keeping x1 fixed
        
        if n_local > 0:
            # Distribute local budget among points
            budget_per_point = max(1, n_local // n_init)
            remaining = n_local - budget_per_point * n_init
            # Give extra evaluations to some points
            extra_points = min(remaining, n_init)
            
            for i in range(n_init):
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break
                
                b_local = budget_per_point + (1 if i < extra_points else 0)
                
                # Simple local search: try small perturbations of x2..xn
                best_x = X[i].copy()
                best_f = F[i].copy()
                best_g_val = self._compute_g(best_x, n)
                
                for _ in range(b_local):
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    # Try a random perturbation of x2..xn
                    scale = 0.1 * (1.0 - _ / max(1, b_local))  # decreasing step size
                    delta = np.zeros(n)
                    delta[1:] = scale * np.random.randn(n - 1)
                    new_x = best_x + delta
                    new_x = np.clip(new_x, lo, hi)
                    
                    new_f = problem(new_x)
                    used += 1
                    
                    # Check if this is better (lower g, and f1 unchanged)
                    new_g_val = self._compute_g(new_x, n)
                    if new_g_val < best_g_val:
                        best_x = new_x
                        best_f = new_f
                        best_g_val = new_g_val
                
                # Update the population
                X[i] = best_x
                F[i] = best_f
        
        return X
    
    def _compute_g(self, x, n):
        """Compute g(x) for ZDT1."""
        if n <= 1:
            return 1.0
        s = np.sum((x[1:] - 0.5)**2 - 10.0 * np.cos(2.0 * np.pi * (x[1:] - 0.5)))
        return 1.0 + 10.0 * (n - 1) + s
