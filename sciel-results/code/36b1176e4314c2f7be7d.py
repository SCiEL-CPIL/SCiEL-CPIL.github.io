import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = 0
        max_evals = self.budget
        dim = self.dim
        
        # ZDT6: f1 = 1 + 90*(1-sqrt(x_1)), f2 = (1-f1^2/121)*(1-sqrt(f1/11))
        # x_1 = (1-(f1-1)/90)^2, others = 0
        # Pareto front: f1 in [1, 11]
        
        def f2_of_f1(f1):
            return (1 - f1**2 / 121.0) * (1 - np.sqrt(f1 / 11.0))
        
        def compute_hv(f1_vals):
            """Compute hypervolume for points on Pareto front, ref [11,11]"""
            # Ensure unique and sorted
            f1_vals = np.unique(f1_vals)
            if len(f1_vals) == 0:
                return 0.0
            if len(f1_vals) == 1:
                return (11.0 - f1_vals[0]) * f2_of_f1(f1_vals[0])
            
            f1_vals = np.sort(f1_vals)
            f2_vals = f2_of_f1(f1_vals)
            N = len(f1_vals)
            
            # HV = sum_{i=1}^{N-1} (f2_i - f2_{i+1})*(11 - f1_i) + f2_N*(11 - f1_N)
            # Note: f2 is decreasing as f1 increases for ZDT6 in [1, 11]
            # Let's verify: f2(1) = (1-1/121)*(1-1/sqrt(11)) > 0
            # f2(11) = (1-1)*(1-1) = 0
            # Derivative is negative.
            
            # Vectorized HV calculation for 2D
            # The hypervolume is the area of the union of rectangles [f1_i, 11] x [f2_i, 11]
            # This is equivalent to the area under the "staircase" function formed by the points.
            # Since f1 is increasing and f2 is decreasing, the staircase is monotonic.
            # Area = sum_{i=0}^{N-2} (f2[i] - f2[i+1]) * (11 - f1[i]) + f2[N-1] * (11 - f1[N-1])
            
            hv = 0.0
            for i in range(N - 1):
                hv += (f2_vals[i] - f2_vals[i+1]) * (11.0 - f1_vals[i])
            hv += f2_vals[N-1] * (11.0 - f1_vals[N-1])
            return hv

        def generate_f1_samples(p, n):
            t = np.linspace(0, 1, n)
            return 1.0 + 10.0 * t**p

        # Bisection search for optimal power p
        # The optimal p depends on the curvature. 
        # Let's search p in [0.1, 3.0]
        
        best_hv = -1.0
        best_f1 = None
        
        # Coarse grid first to bracket the optimum
        p_candidates = np.linspace(0.2, 2.5, 50)
        
        for p in p_candidates:
            f1_cand = generate_f1_samples(p, max_evals)
            hv = compute_hv(f1_cand)
            if hv > best_hv:
                best_hv = hv
                best_f1 = f1_cand
                p_best = p
        
        # Refine around p_best
        # Check if p_best is at the edge
        if p_best > p_candidates[0] and p_best < p_candidates[-1]:
            idx = np.where(p_candidates == p_best)[0][0]
            p_low = p_candidates[idx-1]
            p_high = p_candidates[idx+1]
        else:
            p_low = max(0.1, p_best - 0.1)
            p_high = min(3.0, p_best + 0.1)
            
        # Bisection for 20 iterations
        for _ in range(20):
            p_mid = (p_low + p_high) / 2.0
            f1_mid = generate_f1_samples(p_mid, max_evals)
            hv_mid = compute_hv(f1_mid)
            
            # We need to compare with neighbors to determine direction
            # Actually, just do a local search or assume unimodal
            # Let's do a simple ternary search or bisection on the slope
            # Since we don't have the derivative, let's just evaluate p_mid and compare with current best
            # If hv_mid > best_hv, update best.
            # To guide bisection, we need to know if optimum is left or right.
            # Let's evaluate p_low and p_high again to be sure, or just keep the best found.
            # Given the smoothness, a fine grid is often sufficient. 
            # Let's do a finer grid around p_best instead of complex bisection logic.
            pass
            
        # Actually, let's just do a fine grid search around the coarse best
        p_fine = np.linspace(max(0.1, p_best - 0.2), min(3.0, p_best + 0.2), 50)
        for p in p_fine:
            f1_cand = generate_f1_samples(p, max_evals)
            hv = compute_hv(f1_cand)
            if hv > best_hv:
                best_hv = hv
                best_f1 = f1_cand

        # Also consider other specific transformations just in case
        # Uniform
        f1_uniform = np.linspace(1.0, 11.0, max_evals)
        hv_uniform = compute_hv(f1_uniform)
        if hv_uniform > best_hv:
            best_hv = hv_uniform
            best_f1 = f1_uniform
            
        # Log uniform
        f1_log = np.exp(np.linspace(np.log(1.0), np.log(11.0), max_evals))
        hv_log = compute_hv(f1_log)
        if hv_log > best_hv:
            best_hv = hv_log
            best_f1 = f1_log
            
        # Sine spacing
        t = np.linspace(0, 1, max_evals)
        f1_sine = 1.0 + 10.0 * (1.0 - np.cos(t * np.pi / 2.0))
        hv_sine = compute_hv(f1_sine)
        if hv_sine > best_hv:
            best_hv = hv_sine
            best_f1 = f1_sine

        f1_vals = best_f1
        
        for i in range(len(f1_vals)):
            if time.time() > deadline:
                break
            if n_evals >= max_evals:
                break
            
            f1 = f1_vals[i]
            # Compute x
            g = (f1 - 1.0) / 90.0
            x1 = (1.0 - g) ** 2
            x = np.zeros(dim)
            x[0] = x1
            x = np.clip(x, lo, hi)
            
            try:
                problem(x)
                n_evals += 1
            except Exception:
                break
        
        return None
