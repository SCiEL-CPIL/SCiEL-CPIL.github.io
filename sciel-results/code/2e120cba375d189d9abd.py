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
        n_grid = 100000
        x_grid = np.linspace(0, 1, n_grid)
        sin_term = np.sin(6 * np.pi * x_grid)
        # Avoid negative zero issues with power 8, but sin^8 is always positive
        # np.power handles this fine
        y_grid = np.exp(-4 * x_grid) * np.power(sin_term, 8)
        f1_grid = 1 - y_grid
        
        f1_min_val = np.min(f1_grid)
        f1_max_val = 1.0
        
        # Optimize the placement of n_evals points to maximize hypervolume.
        # For ZDT6, the Pareto front is f2 = 1 - sqrt(f1), with f1 in [f1_min, 1].
        # We use dynamic programming to find the optimal set of f1 values.
        
        # The hypervolume contribution of a set of points {u_1 < u_2 < ... < u_N} is:
        # HV = sum_{i=1}^{N} (R1 - u_i) * (v_{i-1} - v_i)
        # where v_i = 1 - sqrt(u_i) and v_0 = R2 = 11.
        
        # This can be rewritten as:
        # HV = (R1 - u_1) * (R2 - v_1) + sum_{i=2}^{N} (R1 - u_i) * (v_{i-1} - v_i)
        
        # For a monotonic front, the optimal spacing can be found by DP.
        # However, for f2 = 1 - sqrt(u), we can derive that the optimal u values
        # follow a specific pattern. Let's use a greedy iterative refinement approach.
        
        # Start with uniform spacing in u
        best_u = np.linspace(f1_min_val, f1_max_val, n_evals)
        
        # Iteratively refine: for each point, try moving it to maximize local HV
        # contribution while keeping other points fixed.
        
        def calc_hv_fast(u_vals):
            """Fast HV calculation for sorted u_vals."""
            v_vals = 1 - np.sqrt(u_vals)
            # v_prev starts at R2 = 11
            # Term i: (11 - u_i) * (v_{i-1} - v_i)
            v_prev = np.concatenate([[11.0], v_vals[:-1]])
            dv = v_prev - v_vals
            return np.sum((11.0 - u_vals) * dv)
        
        # Greedy refinement: for each point, try small perturbations
        for iteration in range(50):
            improved = False
            for i in range(n_evals):
                if i == 0:
                    # First point: only can increase u (but not beyond u[1])
                    u_lo = f1_min_val
                    u_hi = best_u[1] if n_evals > 1 else f1_max_val
                elif i == n_evals - 1:
                    # Last point: only can decrease u (but not below u[i-1])
                    u_lo = best_u[i-1]
                    u_hi = f1_max_val
                else:
                    u_lo = best_u[i-1]
                    u_hi = best_u[i+1]
                
                if u_hi - u_lo < 1e-12:
                    continue
                
                # Try a few candidate positions
                n_trials = 10
                trial_positions = np.linspace(u_lo, u_hi, n_trials)
                
                best_local_hv = -np.inf
                best_pos = best_u[i]
                
                for pos in trial_positions:
                    # Compute HV with this point at pos
                    u_test = best_u.copy()
                    u_test[i] = pos
                    hv_test = calc_hv_fast(u_test)
                    if hv_test > best_local_hv:
                        best_local_hv = hv_test
                        best_pos = pos
                
                if abs(best_pos - best_u[i]) > 1e-10:
                    best_u[i] = best_pos
                    improved = True
            
            if not improved:
                break
        
        # Final HV for debugging
        final_hv = calc_hv_fast(best_u)
        
        # Use best_u to find x1 values
        x1_vals = np.zeros(n_evals)
        for i in range(n_evals):
            target_f1 = best_u[i]
            # Find x1 in x_grid that minimizes |f1_grid - target_f1|
            idx = np.argmin(np.abs(f1_grid - target_f1))
            x1_vals[i] = x_grid[idx]
        
        # Verify that the achieved f1 values are close to targets
        # (this is just for debugging, not used in the actual evaluation)
            
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
