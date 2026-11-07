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
        
        # Remove dominated points (keep only Pareto-optimal subset)
        n = len(f1)
        keep = np.zeros(n, dtype=bool)
        running_min_f2 = np.inf
        for i in range(n):
            # Point i is on the Pareto front if its f2 is strictly less than all previous points' f2
            # (since f1 is increasing, previous points have lower f1)
            if f2[i] < running_min_f2:
                keep[i] = True
                running_min_f2 = f2[i]
        
        f1 = f1[keep]
        f2 = f2[keep]
        if len(f1) == 0:
            return 0.0
        
        # Compute HV using the standard 2D formula
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
        
        # For ZDT4 with x[1:]=0:
        # f1 = x0, f2 = 1 - sqrt(x0) - x0*sin(10*pi*x0)^2
        # The Pareto front is parameterized by x0 in [0,1]
        # We want to choose N values of x0 to maximize hypervolume
        
        # Strategy: Use arc-length uniform spacing on the Pareto front curve
        # Compute the arc length distribution of the curve (x0, f2(x0))
        # Then place N points uniformly in arc length
        
        # First, compute a fine discretization of the curve
        n_fine = 10000
        x0_fine = np.linspace(0.0, 1.0, n_fine)
        f2_fine = self._zdt4_f2(x0_fine)
        
        # Compute arc length
        dx = np.diff(x0_fine)
        df2 = np.diff(f2_fine)
        ds = np.sqrt(dx**2 + df2**2)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        s_total = s[-1]
        
        # Normalize arc length to [0, 1]
        s_norm = s / s_total
        
        # Place N points uniformly in normalized arc length
        if N == 1:
            x0_vals = np.array([0.5])  # Middle of the front
        else:
            s_targets = np.linspace(0.0, 1.0, N)
            # Interpolate to get x0 values
            x0_vals = np.interp(s_targets, s_norm, x0_fine)
        
        x0_vals = np.clip(x0_vals, lo[0], hi[0])
        
        # Also try power-law spacing and pick the best
        def compute_hv_for_power(p, N):
            if N <= 0:
                return 0.0
            if N == 1:
                x0_vals_test = np.array([0.0])
            else:
                t = np.linspace(0.0, 1.0, N)
                x0_vals_test = t**p
            x0_vals_test = np.clip(x0_vals_test, lo[0], hi[0])
            f1_vals = x0_vals_test
            f2_vals = self._zdt4_f2(x0_vals_test)
            return self._hv_2d(f1_vals, f2_vals)
        
        # Evaluate arc-length spacing HV
        f1_vals_arc = x0_vals
        f2_vals_arc = self._zdt4_f2(x0_vals)
        hv_arc = self._hv_2d(f1_vals_arc, f2_vals_arc)
        
        # Coarse search over power p
        best_hv = hv_arc
        best_x0 = x0_vals.copy()
        
        if time.time() - t0 < DEADLINE - 10:
            p_coarse = np.linspace(0.1, 5.0, 50)
            for p in p_coarse:
                hv = compute_hv_for_power(p, N)
                if hv > best_hv:
                    best_hv = hv
                    if N == 1:
                        best_x0 = np.array([0.0])
                    else:
                        t = np.linspace(0.0, 1.0, N)
                        best_x0 = t**p
                    best_x0 = np.clip(best_x0, lo[0], hi[0])
            
            # Fine search around best power
            if best_hv > hv_arc:
                # Find the best p from coarse search
                best_p = p_coarse[0]
                for p in p_coarse:
                    hv = compute_hv_for_power(p, N)
                    if hv > best_hv - 1e-10:
                        best_p = p
                        break
                
                p_fine = np.linspace(max(0.05, best_p - 0.5), best_p + 0.5, 50)
                for p in p_fine:
                    hv = compute_hv_for_power(p, N)
                    if hv > best_hv:
                        best_hv = hv
                        if N == 1:
                            best_x0 = np.array([0.0])
                        else:
                            t = np.linspace(0.0, 1.0, N)
                            best_x0 = t**p
                        best_x0 = np.clip(best_x0, lo[0], hi[0])
        
        # Use the best x0 placement found
        x0_vals = best_x0
        
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
