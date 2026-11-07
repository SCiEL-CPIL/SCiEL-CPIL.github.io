import numpy as np
import scipy.optimize
import time

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
                x_arr = np.clip(x_arr, -5.0, 5.0)
                f_val = float(func(x_arr))
            except Exception:
                return best_f
                
            if f_val < best_f:
                best_f = f_val
                best_x = x_arr.copy()
            return f_val

        # Phase 1: Linear Fit using Space-Filling Design
        # Use 0 and d Sobol/Halton points to estimate gradient
        # This replaces the axis-aligned probes which may be further from the optimum on average
        
        # Generate d+1 points: 0 and d random/Sobol points
        # We use a fixed set of Sobol-like points for reproducibility and good distribution
        # Sobol sequence for d=5
        # Simple Halton sequence for d dimensions
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        points = [np.zeros(d)]
        for i in range(1, d + 1):
            p = np.array([halton(i, b) for b in range(2, d + 2)])
            # Scale to [-5, 5]
            p = p * 10.0 - 5.0
            points.append(p)
        
        # Evaluate the points
        f_values = np.zeros(d + 1)
        for i, p in enumerate(points):
            if calls >= budget or time.time() > deadline:
                break
            f_values[i] = wrapper(p)
            
        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Solve for w and c
        # f(x) = w^T x + c
        # f(0) = c
        # f(p_i) = w^T p_i + c
        c = f_values[0]
        # Matrix P is d x d (rows are p_1 ... p_d)
        P = np.array(points[1:])
        f_vec = f_values[1:] - c
        
        # Solve P^T w = f_vec? No, w^T p_i = f_vec[i]
        # So P w = f_vec? No, P is d x d.
        # Let A be the matrix where row i is p_i.
        # A w = f_vec
        try:
            w_est = np.linalg.solve(P, f_vec)
        except np.linalg.LinAlgError:
            # Fallback to least squares
            w_est, residuals, rank, s = np.linalg.lstsq(P, f_vec, rcond=None)
            
        # Determine optimal corner
        x_corner = np.zeros(d)
        for i in range(d):
            if w_est[i] > 0:
                x_corner[i] = -5.0
            else:
                x_corner[i] = 5.0
                
        f_corner = wrapper(x_corner)
        
        # Phase 2: Local Search (Keep Nelder-Mead as fallback/refinement)
        if f_values[0] < f_corner:
            x_start = points[0]
        else:
            x_start = x_corner
            
        remaining = budget - calls
        if remaining > 10 and time.time() < deadline:
            try:
                res = scipy.optimize.minimize(
                    wrapper, 
                    x_start, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 1000, 
                        'maxfev': remaining, 
                        'xatol': 1e-5, 
                        'fatol': 1e-5
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
        
        # Phase 3: Fill remaining budget
        while calls < budget and time.time() < deadline:
            wrapper(best_x)
            
        return best_x, best_f
