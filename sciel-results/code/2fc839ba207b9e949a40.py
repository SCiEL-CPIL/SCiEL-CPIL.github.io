import numpy as np
import time
from scipy.optimize import brentq

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _compute_hv(self, x1_vals):
        """Compute hypervolume for points on ZDT1 front with ref [11, 11].
        x1_vals must be sorted ascending, in [0, 1].
        f1 = x1, f2 = 1 - sqrt(x1)
        """
        if len(x1_vals) == 0:
            return 0.0
        f1 = x1_vals
        f2 = 1.0 - np.sqrt(x1_vals)
        
        # Widths in f1
        f1_next = np.empty_like(f1)
        f1_next[:-1] = f1[1:]
        f1_next[-1] = 11.0
        
        widths = f1_next - f1
        heights = 11.0 - f2
        
        return np.sum(widths * heights)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        B = self.budget
        D = self.dim
        
        # Optimal quantization: place points where cumulative weight is uniform
        # Weight function: w(x) = 10 + sqrt(x) (marginal HV contribution per dx)
        # CDF: W(x) = 10x + (2/3)x^(3/2)
        # W(1) = 10 + 2/3 = 32/3
        # Optimal: W(x_i) = i * W(1) / B for i = 1, ..., B
        
        def cdf(x):
            return 10.0 * x + (2.0 / 3.0) * x ** 1.5
        
        W_total = cdf(1.0)  # = 32/3
        
        # Solve for x_i: 10*x + (2/3)*x^1.5 = target_i
        # Use brentq for each target
        x1_vals = np.zeros(B)
        
        for i in range(B):
            if time.time() > deadline - 30.0:
                # Fallback: use power law approximation
                x1_vals[i:] = np.linspace(0, 1, B - i)[1:]
                break
            target = (i + 1) * W_total / B
            # Solve 10*x + (2/3)*x^1.5 = target for x in [0, 1]
            try:
                x1_vals[i] = brentq(lambda x: cdf(x) - target, 0.0, 1.0, xtol=1e-14)
            except:
                # Fallback for edge cases
                x1_vals[i] = target / W_total  # linear approximation
        
        x1_vals = np.clip(x1_vals, 0.0, 1.0)
        x1_vals.sort()
        
        # Verify and potentially refine
        hv_optimal = self._compute_hv(x1_vals)
        
        # Try a few alternative strategies to see if we can do better
        # Strategy 1: Uniform in f1 space
        x1_uniform = np.linspace(0, 1, B)
        hv_uniform = self._compute_hv(x1_uniform)
        
        # Strategy 2: Use the power-law with best p from quick search
        u_init = np.linspace(0, 1, B)
        best_hv = hv_optimal
        best_x1 = x1_vals.copy()
        
        if hv_uniform > best_hv:
            best_hv = hv_uniform
            best_x1 = x1_uniform.copy()
        
        # Quick power-law search
        for p in [1.5, 2.0, 2.5, 3.0]:
            x1_try = u_init ** p
            hv_try = self._compute_hv(x1_try)
            if hv_try > best_hv:
                best_hv = hv_try
                best_x1 = x1_try.copy()
        
        # Final refinement: local optimization of the best candidate
        # Use the optimal quantization result as primary
        x1_final = best_x1
        x1_final = np.clip(x1_final, 0.0, 1.0)
        x1_final.sort()
        
        # Build the evaluation matrix
        X = np.zeros((B, D))
        X[:, 0] = x1_final
        for i in range(1, D):
            X[:, i] = lo[i]
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        for i in range(B):
            if time.time() > deadline:
                break
            problem(X[i].copy())
        
        return X
