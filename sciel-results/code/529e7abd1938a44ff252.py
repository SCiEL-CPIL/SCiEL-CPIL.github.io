import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        d = self.dim
        N = self.budget
        deadline = self.start_time + 290.0

        def time_check():
            return time.time() < deadline

        def safe_eval(x):
            if self.calls >= N or not time_check():
                return np.inf
            self.calls += 1
            try:
                val = float(func(x))
                return val
            except Exception:
                return np.inf

        # Track best globally
        best_x = np.zeros(d)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Initial evaluation at origin
        x0 = np.zeros(d)
        f0 = safe_eval(x0)
        update_best(x0, f0)

        # Phase 1: Nelder-Mead with adaptive restarts
        # Bent Cigar is ill-conditioned; Nelder-Mead handles this better than gradient methods
        rng = np.random.default_rng(42)
        
        # Generate diverse starting points
        start_points = [x0.copy()]
        for _ in range(8):
            start_points.append(rng.uniform(-4, 4, d))
        
        # Run Nelder-Mead from each start point with limited budget
        for sp in start_points:
            if not time_check() or self.calls >= N - 50:
                break
            
            # Evaluate the start point
            f_sp = safe_eval(sp)
            update_best(sp, f_sp)
            
            if np.isinf(f_sp):
                continue
            
            # Run Nelder-Mead
            def objective(x):
                return safe_eval(x)
            
            try:
                # Use smaller initial simplex for ill-conditioned problems
                initial_simplex = sp + 0.1 * rng.standard_normal((d + 1, d))
                # Clip to bounds
                initial_simplex = np.clip(initial_simplex, -5.0, 5.0)
                
                result = minimize(
                    objective,
                    sp,
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'maxiter': 200,
                        'maxfev': min(N - self.calls, 150)
                    }
                )
                if result.fun < best_f:
                    update_best(result.x, result.fun)
            except Exception:
                pass

        # Phase 2: Adaptive coordinate descent with momentum
        # For ill-conditioned problems, use larger steps in well-conditioned directions
        if self.calls < N - 20 and time_check():
            x_current = best_x.copy()
            
            # Estimate condition number from function values
            # For Bent Cigar: f(x) = sum(10^(0.8*(i-1)) * x_i^2) + linear term
            # The dominant direction is the first coordinate
            # Use adaptive step sizes based on coordinate importance
            
            def adaptive_coordinate_descent(x_base, max_sweeps=15):
                nonlocal best_f, best_x
                
                # Start with larger steps, decrease over time
                step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
                
                for sweep in range(max_sweeps):
                    if not time_check() or self.calls >= N:
                        break
                    
                    step = step_sizes[min(sweep, len(step_sizes) - 1)]
                    improved = False
                    
                    for i in range(d):
                        if not time_check() or self.calls >= N:
                            break
                        
                        # Try positive and negative steps
                        for sign in [1, -1]:
                            x_trial = x_base.copy()
                            x_trial[i] += sign * step
                            x_trial = np.clip(x_trial, -5.0, 5.0)
                            
                            f_trial = safe_eval(x_trial)
                            update_best(x_trial, f_trial)
                            
                            if f_trial < best_f:
                                x_base[i] = x_trial[i]
                                improved = True
                                # Try to continue in the same direction
                                for _ in range(3):
                                    if not time_check() or self.calls >= N:
                                        break
                                    x_next = x_base.copy()
                                    x_next[i] += sign * step
                                    x_next = np.clip(x_next, -5.0, 5.0)
                                    f_next = safe_eval(x_next)
                                    update_best(x_next, f_next)
                                    if f_next < best_f:
                                        x_base[i] = x_next[i]
                                    else:
                                        break
                    
                    if not improved and step < 0.01:
                        break
                
                return x_base

            x_current = adaptive_coordinate_descent(x_current)
            update_best(x_current, safe_eval(x_current))

        return best_x, best_f
