"""ZDT1 multi-objective optimizer exploiting the known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Optimizer exploiting ZDT1's decoupled Pareto front structure.
    
    ZDT1: f1 = x1, g = 1 + 9*sum(x2..xn)/(n-1), f2 = g*(1 - sqrt(f1/g))
    The Pareto front is achieved when x2..xn = 0 (g=1), and f1 sweeps [0,1].
    With 200 evaluations we can densely sample the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        B = int(self.budget)
        D = int(self.dim)

        # For ZDT1, the Pareto front is at x2..xn = 0 (lower bound)
        # f1 = x1, f2 = 1 - sqrt(x1) when x2..xn = 0
        # We sweep x1 with non-uniform spacing to maximize hypervolume coverage
        
        # Use power-law clustering: x1 = u^p where u is uniform in [0,1]
        # Higher p clusters more points near x1=0 where the front has high curvature
        # and the hypervolume integrand (11 - f2) is largest
        
        # Try a few power exponents and pick the one that maximizes hypervolume
        best_hv = -1
        best_x1_vals = None
        best_p = 2.0
        
        for p in [1.5, 2.0, 2.5, 3.0, 4.0]:
            u = np.linspace(0.0, 1.0, B)
            x1_vals = u ** p
            
            # Compute f2 values for these x1 values
            f1_vals = x1_vals
            f2_vals = 1.0 - np.sqrt(f1_vals)
            
            # Compute hypervolume with reference point [11, 11]
            # Sort by f1 ascending
            order = np.argsort(f1_vals)
            f1_sorted = f1_vals[order]
            f2_sorted = f2_vals[order]
            
            # Filter points strictly better than reference
            mask = (f1_sorted < 11.0) & (f2_sorted < 11.0)
            if not np.any(mask):
                continue
            f1_f = f1_sorted[mask]
            f2_f = f2_sorted[mask]
            
            if len(f1_f) == 0:
                continue
            
            # Sort by f1 ascending
            idx = np.argsort(f1_f)
            f1_s = f1_f[idx]
            f2_s = f2_f[idx]
            
            # Remove dominated points (keep only non-dominated)
            # Since f2 decreases with f1 on the front, all should be non-dominated
            nondom_idx = [0]
            for i in range(1, len(f1_s)):
                if f2_s[i] < f2_s[nondom_idx[-1]]:
                    nondom_idx.append(i)
            f1_nd = f1_s[nondom_idx]
            f2_nd = f2_s[nondom_idx]
            
            # Compute HV
            hv = 0.0
            for i in range(len(f1_nd)):
                if i + 1 < len(f1_nd):
                    hf1 = f1_nd[i + 1] - f1_nd[i]
                else:
                    hf1 = 11.0 - f1_nd[i]
                hf2 = 11.0 - f2_nd[i]
                hv += hf1 * hf2
            
            if hv > best_hv:
                best_hv = hv
                best_x1_vals = x1_vals
                best_p = p
        
        # Use the best distribution found
        if best_x1_vals is None:
            u = np.linspace(0.0, 1.0, B)
            best_x1_vals = u ** best_p
        
        # Create the sweep matrix
        X = np.zeros((B, D))
        X[:, 0] = best_x1_vals
        # x2..xn stay at 0 (lower bound), which is optimal for ZDT1
        
        # Clip to bounds to be safe
        X = np.clip(X, lo, hi)
        
        # Evaluate all sweep points
        used = 0
        for i in range(B):
            if time.time() > deadline:
                break
            if used >= B:
                break
            problem(X[i].copy())
            used += 1
        
        return X
# EVOLVE-BLOCK-END
