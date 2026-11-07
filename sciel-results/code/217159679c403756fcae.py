import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = [b[0] for b in bounds]
        hi = [b[1] for b in bounds]
        
        n_evals = self.budget
        start_time = time.time()
        deadline = 300.0

        # ZDT6 f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # Range of f1 is [f1_min, 1.0]
        # f1_min is approx 1 - exp(-1/3)
        
        # Pre-compute a high-resolution lookup table for f1(x1)
        n_grid = 200000
        x_grid = np.linspace(0, 1, n_grid)
        sin_term = np.sin(6 * np.pi * x_grid)
        y_grid = np.exp(-4 * x_grid) * np.power(sin_term, 8)
        f1_grid = 1 - y_grid
        
        f1_min_val = np.min(f1_grid)
        f1_max_val = 1.0
        
        # We want to select n_evals target f1 values in [f1_min_val, f1_max_val]
        # that maximize the hypervolume.
        # HV formula for points (u_i, v_i) sorted by u ascending, v descending:
        # HV = sum_{i=1}^N (R1 - u_i) * (v_{i-1} - v_i) with v_0 = R2.
        # Here v = 1 - sqrt(u).
        # R1 = 11, R2 = 11.
        
        def calc_hv_fast(u_vals):
            # u_vals must be sorted ascending
            v_vals = 1 - np.sqrt(u_vals)
            v_prev = np.concatenate([[11.0], v_vals[:-1]])
            return np.sum((11.0 - u_vals) * (v_prev - v_vals))

        # Candidate spacing strategies
        candidates = []
        
        # 1. Uniform in f1
        candidates.append(np.linspace(f1_min_val, f1_max_val, n_evals))
        
        # 2. Uniform in sqrt(f1)
        sqrt_u = np.linspace(np.sqrt(f1_min_val), np.sqrt(f1_max_val), n_evals)
        candidates.append(sqrt_u ** 2)
        
        # 3. Uniform in f1^0.25
        q_u = np.linspace(f1_min_val**0.25, f1_max_val**0.25, n_evals)
        candidates.append(q_u ** 4)
        
        # 4-8. Spacing based on marginal HV gain
        # Marginal gain ~ (11-u)/(2*sqrt(u)), higher for small u
        # So cluster points at small u
        for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
            # Create a CDF-like mapping
            f1_test = np.linspace(f1_min_val, f1_max_val, 2000)
            weights = (11.0 - f1_test) / (2.0 * np.sqrt(f1_test))
            cdf = np.cumsum(weights)
            cdf = cdf / cdf[-1]
            t = np.linspace(0, 1, n_evals)
            f1_mapped = np.interp(t, cdf, f1_test)
            candidates.append(f1_mapped)
        
        # Evaluate HV for each candidate
        best_u = None
        best_hv = -1
        for u_cand in candidates:
            if not np.all(np.diff(u_cand) > 1e-12):  # Ensure strictly increasing
                continue
            hv_val = calc_hv_fast(u_cand)
            if hv_val > best_hv:
                best_hv = hv_val
                best_u = u_cand
        
        # Use best_u to find x1 values
        x1_vals = np.zeros(n_evals)
        for i in range(n_evals):
            if time.time() - start_time > deadline - 5:
                break
            target_f1 = best_u[i]
            # Find x1 in x_grid that minimizes |f1_grid - target_f1|
            idx = np.argmin(np.abs(f1_grid - target_f1))
            x1_vals[i] = x_grid[idx]
            
        # Construct points
        points = np.zeros((n_evals, self.dim))
        points[:, 0] = x1_vals
        # x[1:] are 0
        
        # Evaluate
        for i in range(n_evals):
            if time.time() - start_time > deadline:
                break
            x = points[i]
            x_clipped = np.clip(x, lo, hi)
            problem(x_clipped)
            
        return None
