import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        self.start_time = time.time()
        d = self.dim
        N = self.budget
        deadline = self.start_time + 280.0  # Safety margin

        def time_check():
            return time.time() < deadline

        def safe_eval(x):
            if self.calls >= N or not time_check():
                return np.inf
            self.calls += 1
            try:
                # Ensure x is a 1D array of floats
                x = np.asarray(x, dtype=float).flatten()
                # Clip to box [-5, 5] to be safe, though func should do it
                x = np.clip(x, -5.0, 5.0)
                val = float(func(x))
                return val
            except Exception:
                return np.inf

        def update_best(x, f):
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()

        # Initial evaluation
        x0 = np.zeros(d)
        f0 = safe_eval(x0)
        update_best(x0, f0)

        rng = np.random.default_rng(42)
        
        # Generate starting points
        start_points = [x0.copy()]
        
        # Add random points
        for _ in range(5):
            start_points.append(rng.uniform(-4, 4, d))
            
        # Add points along axes (helps if valley is aligned with an axis)
        for i in range(d):
            for sign in [-1, 1]:
                pt = np.zeros(d)
                pt[i] = 3.0 * sign
                start_points.append(pt)

        # Phase 1: L-BFGS-B runs from multiple starts to exploit curvature
        # Bent Cigar has anisotropic curvature; L-BFGS-B approximates the Hessian
        # and converges rapidly along the narrow valley direction.
        
        max_evals_per_run = 50
        
        for sp in start_points:
            if self.calls >= N or not time_check():
                break
            
            try:
                res = minimize(
                    safe_eval, 
                    sp, 
                    method='L-BFGS-B', 
                    bounds=[(-5.0, 5.0)] * d,
                    options={'maxiter': 100, 'maxfun': max_evals_per_run, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < self.best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        # Phase 2: Coordinate Descent with Golden Section Search
        # After L-BFGS-B finds the basin, coordinate descent with golden section
        # search can polish the solution to high precision, especially effective
        # for separable or nearly-separable functions like Bent Cigar.
        
        if self.calls < N - 10 and time_check():
            x_current = self.best_x.copy()
            
            def golden_section_search(x_base, coord_idx, x_lo, x_hi, max_evals=15):
                if not time_check() or self.calls >= N - max_evals:
                    return x_base[coord_idx], np.inf
                
                inv_phi = (1 + np.sqrt(5)) / 2
                inv_phi = 1 / inv_phi
                
                c = x_hi - inv_phi * (x_hi - x_lo)
                dd = x_lo + inv_phi * (x_hi - x_lo)
                
                x_c = x_base.copy()
                x_c[coord_idx] = c
                f_c = safe_eval(x_c)
                update_best(x_c, f_c)
                
                x_dd = x_base.copy()
                x_dd[coord_idx] = dd
                f_dd = safe_eval(x_dd)
                update_best(x_dd, f_dd)
                
                for _ in range(max_evals - 2):
                    if not time_check() or self.calls >= N:
                        break
                    
                    if f_c < f_dd:
                        x_hi = dd
                        dd = c
                        f_dd = f_c
                        c = x_hi - inv_phi * (x_hi - x_lo)
                        x_c = x_base.copy()
                        x_c[coord_idx] = c
                        f_c = safe_eval(x_c)
                        update_best(x_c, f_c)
                    else:
                        x_lo = c
                        c = dd
                        f_c = f_dd
                        dd = x_lo + inv_phi * (x_hi - x_lo)
                        x_dd = x_base.copy()
                        x_dd[coord_idx] = dd
                        f_dd = safe_eval(x_dd)
                        update_best(x_dd, f_dd)
                
                if f_c < f_dd:
                    return c, f_c
                else:
                    return dd, f_dd

            # Dynamic range coordinate descent
            range_sizes = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            
            max_sweeps = 30
            for sweep in range(max_sweeps):
                if not time_check() or self.calls >= N:
                    break
                
                if sweep < len(range_sizes):
                    r_size = range_sizes[sweep]
                else:
                    r_size = range_sizes[-1]
                
                improved = False
                for i in range(d):
                    if not time_check() or self.calls >= N:
                        break
                    
                    x_lo = max(x_current[i] - r_size, -5.0)
                    x_hi = min(x_current[i] + r_size, 5.0)
                    
                    if x_hi - x_lo < 1e-8:
                        continue

                    new_val, new_f = golden_section_search(x_current, i, x_lo, x_hi, max_evals=12)
                    
                    if new_f < self.best_f:
                        x_current[i] = new_val
                        improved = True
                
                if not improved:
                    if r_size <= 0.01:
                        break

        # Phase 3: Final polish with L-BFGS-B if budget remains
        if self.calls < N - 10 and time_check():
            try:
                res = minimize(
                    safe_eval, 
                    self.best_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5.0, 5.0)] * d,
                    options={'maxiter': 200, 'maxfun': N - self.calls - 5, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                if res.fun < self.best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        return self.best_x, self.best_f
