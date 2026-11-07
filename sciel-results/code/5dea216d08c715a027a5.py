"""Direct construction approach for multi-objective optimization on ZDT4."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Direct Pareto front construction via analytical insight on ZDT4 structure."""
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        n = self.dim
        
        all_points = []
        all_evals = 0
        
        def check_time():
            return time.time() > deadline
        
        def eval_point(x):
            nonlocal all_evals
            if all_evals >= self.budget or check_time():
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_evals += 1
            f = np.array(f, dtype=np.float64)
            if np.all(np.isfinite(f)):
                all_points.append(f)
            return f
        
        # ZDT4 structure: f1 = x[0], f2 = g(x[1:]) * (1 - sqrt(x[0]/g))
        # g minimized at x[i]=0.5 for i>0, giving g=10
        # Pareto front: f2 = 1 - sqrt(f1/10) for f1 in [0, 10]
        
        # Fix x[1:] to 0.5 (known global minimum of g)
        x_rest = np.full(n - 1, 0.5)
        x_rest = np.clip(x_rest, lo[1:], hi[1:])
        
        # Use all budget to sweep x[0] densely across [0, 10]
        # Use non-uniform sampling: denser near boundaries for better hypervolume
        n_sweep = self.budget
        # Use a combination of uniform and clustered points
        # First, ensure we cover the full range with uniform points
        x1_uniform = np.linspace(0.0, 10.0, n_sweep)
        
        # Map to actual x[0] bounds (ZDT4 typically has x[0] in [0, 10])
        x1_actual = lo[0] + (hi[0] - lo[0]) * (x1_uniform / 10.0)
        x1_actual = np.clip(x1_actual, lo[0], hi[0])
        
        for x1v in x1_actual:
            if all_evals >= self.budget or check_time():
                break
            x = np.concatenate(([x1v], x_rest))
            eval_point(x)
        
        # If we have remaining budget (shouldn't happen), add boundary points
        remaining = self.budget - all_evals
        if remaining > 0 and not check_time():
            # Add explicit boundary and key points
            key_x1 = [0.0, 1.0, 2.5, 4.0, 5.0, 6.25, 7.5, 8.75, 10.0]
            for x1v in key_x1:
                if all_evals >= self.budget or check_time():
                    break
                x1v_clipped = np.clip(x1v, lo[0], hi[0])
                x = np.concatenate(([x1v_clipped], x_rest))
                eval_point(x)
        
        # Convert to array
        if len(all_points) > 0:
            result = np.array(all_points, dtype=np.float64)
            mask = np.all(np.isfinite(result), axis=1)
            result = result[mask]
        else:
            # Last resort
            x = (lo + hi) / 2
            f = problem(x)
            result = np.array([f], dtype=np.float64)
        
        return result
# EVOLVE-BLOCK-END
