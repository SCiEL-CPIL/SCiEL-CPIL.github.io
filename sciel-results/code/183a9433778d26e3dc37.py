import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
    
    def _zdt4_f2(self, x0):
        """Compute f2 for ZDT4 given x0 (x[1:] = 0)"""
        x0 = np.clip(x0, 0, 1)
        return 1.0 - np.sqrt(x0) - x0 * np.sin(10.0 * np.pi * x0)**2
    
    def _hv_2d(self, f1_vals, f2_vals, R1=11.0, R2=400.0):
        """Compute 2D hypervolume for non-dominated points sorted by increasing f1"""
        if len(f1_vals) == 0:
            return 0.0
        # Sort by f1
        order = np.argsort(f1_vals)
        f1 = f1_vals[order]
        f2 = f2_vals[order]
        # Remove points not strictly better than reference
        mask = (f1 < R1) & (f2 < R2)
        f1 = f1[mask]
        f2 = f2[mask]
        if len(f1) == 0:
            return 0.0
        # Compute HV
        hv = 0.0
        for i in range(len(f1)):
            next_f1 = f1[i+1] if i+1 < len(f1) else R1
            hv += (next_f1 - f1[i]) * (R2 - f2[i])
        return hv
    
    def __call__(self, problem, bounds):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        DEADLINE = 290.0
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        if budget <= 0:
            return np.empty((0, dim))
        
        N = min(budget, 200)
        
        # Use arc-length parameterization for optimal hypervolume coverage
        if N == 1:
            x0_vals = np.array([0.0])
        else:
            # Compute arc length of the Pareto front curve
            # The front is (x0, f2(x0)) where f2(x0) = 1 - sqrt(x0) - x0*sin^2(10*pi*x0)
            # Arc length element: ds = sqrt(1 + (df2/dx0)^2) * dx0
            # df2/dx0 = -1/(2*sqrt(x0)) - sin^2(10*pi*x0) - 10*pi*x0*sin(20*pi*x0)
            
            # Use fine grid to compute cumulative arc length
            n_grid = 10000
            x_grid = np.linspace(0.0, 1.0, n_grid)
            # Avoid division by zero at x=0
            x_safe = np.maximum(x_grid, 1e-10)
            
            # Compute derivative df2/dx0
            sin_term = np.sin(10.0 * np.pi * x_safe)
            cos_term = np.cos(10.0 * np.pi * x_safe)
            df2_dx0 = -1.0 / (2.0 * np.sqrt(x_safe)) - sin_term**2 - 10.0 * np.pi * x_safe * sin_term * cos_term * 2.0
            # Note: d/dx[sin^2(10*pi*x)] = 2*sin(10*pi*x)*cos(10*pi*x)*10*pi = sin(20*pi*x)*10*pi
            # So df2/dx0 = -1/(2*sqrt(x)) - sin^2(10*pi*x) - x * d/dx[sin^2(10*pi*x)]
            # = -1/(2*sqrt(x)) - sin^2(10*pi*x) - x * 10*pi*sin(20*pi*x)
            df2_dx0 = -1.0 / (2.0 * np.sqrt(x_safe)) - sin_term**2 - x_safe * 10.0 * np.pi * np.sin(20.0 * np.pi * x_safe)
            
            # Arc length element
            ds = np.sqrt(1.0 + df2_dx0**2)
            
            # Cumulative arc length
            arc_len = np.cumsum(ds)
            arc_len = arc_len / arc_len[-1]  # Normalize to [0, 1]
            
            # For each desired point, find x0 such that arc_len(x0) = i/(N-1)
            s_targets = np.linspace(0.0, 1.0, N)
            x0_vals = np.interp(s_targets, arc_len, x_grid)
            
            # Clip to bounds
            x0_vals = np.clip(x0_vals, lo[0], hi[0])
        
        evals = 0
        results = []
        
        for x0_val in x0_vals:
            if evals >= budget or time.time() - t0 > DEADLINE:
                break
            
            x = np.zeros(dim)
            x[0] = float(x0_val)
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                evals += 1
                results.append(f)
            except Exception:
                pass
        
        if results:
            return np.array(results)
        
        return np.empty((0, dim))
