import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = float('inf')
        self.dim = self.dim
        self.budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 280.0

        def tracked_func(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            self.calls += 1
            fx = float(func(x))
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = np.array(x)
            return fx

        # Initialize at origin
        x = np.zeros(self.dim)
        try:
            tracked_func(x)
        except Exception:
            return self.best_x, self.best_f

        # Helper function for coordinate-wise search
        def coordinate_search(x_current, step, search_range):
            # search_range is a tuple (low, high) or None to use global [-4, 4]
            x_new = x_current.copy()
            for i in range(self.dim):
                if time.time() > deadline - 5.0:
                    return x_new
                if self.calls >= self.budget:
                    return x_new
                    
                if search_range is None:
                    low, high = -4.0, 4.0
                else:
                    low = max(-4.0, x_current[i] - search_range)
                    high = min(4.0, x_current[i] + search_range)
                    
                # Generate points
                n_points = int((high - low) / step) + 1
                if n_points < 1:
                    continue
                points = np.linspace(low, high, n_points)
                
                best_x_i = x_current[i]
                best_f_i = self.best_f # Note: this is global best, not local best for this coord
                # We need to evaluate along the line
                
                # To be efficient, we can just evaluate and update x_new[i]
                for p in points:
                    if time.time() > deadline - 5.0:
                        break
                    if self.calls >= self.budget:
                        break
                        
                    x_test = x_new.copy()
                    x_test[i] = p
                    try:
                        f_test = tracked_func(x_test)
                    except Exception:
                        break
                    
                    if f_test < best_f_i:
                        best_f_i = f_test
                        best_x_i = p
                
                x_new[i] = best_x_i
            return x_new

        # Phase 1: Coarse Coordinate Search
        # Step 1.0, Global Range
        x = coordinate_search(x, 1.0, None)
        
        # Phase 2: Fine Coordinate Search
        # Step 0.2, Local Range +/- 1.0
        x = coordinate_search(x, 0.2, 1.0)
        
        # Phase 3: Nelder-Mead Polish
        remaining = self.budget - self.calls
        if remaining > 0:
            max_iter = max(1, remaining // (self.dim + 1))
            try:
                res = minimize(
                    tracked_func, 
                    x, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter, 
                        'xatol': 1e-10, 
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.array(res.x)
            except Exception:
                pass

        return self.best_x, self.best_f
