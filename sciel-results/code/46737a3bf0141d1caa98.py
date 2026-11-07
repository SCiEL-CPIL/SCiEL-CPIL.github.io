import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 300s limit, keep 20s margin
        deadline = start_time + 280.0
        
        eval_count = 0
        best_x = None
        best_f = float('inf')
        
        def wrapper(x):
            nonlocal eval_count, best_x, best_f
            # Check budget
            if eval_count >= self.budget:
                # Return a dummy value, shouldn't be called
                return best_f if best_f < float('inf') else 1e100
            
            # Check time
            if time.time() > deadline:
                return best_f if best_f < float('inf') else 1e100

            try:
                x = np.clip(x, -5, 5)
                fx = float(func(x))
            except Exception:
                fx = 1e100 # Large penalty
            
            eval_count += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            return fx

        # 1. Grid Search
        # Use 3 levels: -5, 0, 5
        levels = [-5.0, 0.0, 5.0]
        # Generate all combinations
        # np.meshgrid creates arrays of shape (3,3,3,3,3)
        grid_points = list(np.meshgrid(*([levels] * self.dim)))
        grid_points = np.array(grid_points).T.reshape(-1, self.dim)
        
        grid_results = []
        
        for p in grid_points:
            if time.time() > deadline or eval_count >= self.budget:
                break
            fx = wrapper(p)
            grid_results.append((fx, p))
            
        # Sort by fitness
        grid_results.sort(key=lambda x: x[0])
        
        # 2. Local Optimization on Top 5
        top_k = 5
        starts = [p for _, p in grid_results[:top_k]]
        
        # Calculate remaining budget
        remaining = self.budget - eval_count
        if remaining <= 0 or not starts:
            return best_x, best_f
            
        # Allocate calls to each start
        # We want to use most of the remaining budget.
        # Nelder-Mead maxfev must be int.
        # If we have 257 calls and 5 starts, 50 each = 250.
        # This leaves 7 calls.
        # We can give the best start more calls if we want, but equal is simpler and robust.
        calls_per_start = max(1, remaining // len(starts))
        
        for s in starts:
            if time.time() > deadline or eval_count >= self.budget:
                break
                
            try:
                # Run Nelder-Mead
                # xatol and fatol help stop early if converged
                minimize(
                    wrapper, 
                    s, 
                    method='Nelder-Mead', 
                    options={
                        'maxfev': calls_per_start, 
                        'xatol': 1e-8, 
                        'fatol': 1e-8
                    }
                )
            except Exception:
                pass
                
        return best_x, best_f
