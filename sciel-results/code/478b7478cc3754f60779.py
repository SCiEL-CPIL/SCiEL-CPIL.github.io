import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For ZDT4, the Pareto front is 1D (controlled by x1)
        # x2..xd affect f1 through sum(x_i^2 * sin(10*pi*x_i))
        # To minimize f1, we want sin terms to be 0 or negative
        # Optimal tail values: x_i = 0 (sin=0) or x_i = 0.5 (sin(5π)=0)
        # Actually sin(10π*0.5) = sin(5π) = 0, so both 0 and 0.5 give sin=0
        
        # Strategy: For each x1 value, try multiple tail configurations
        # and pick the one with lowest f1
        
        # Sample x1 values across [0, 1] to cover the Pareto front
        n_x1_samples = 150
        x1_vals = np.linspace(0.0, 1.0, n_x1_samples)
        
        # Pre-compute candidate tail configurations
        # For ZDT4, x_i in {0, 0.5} gives sin(10πx_i) = 0
        # Also try small non-zero values to see if they help
        tail_configs = [
            np.zeros(self.dim - 1),           # all zeros
            np.full(self.dim - 1, 0.5),       # all 0.5
            np.full(self.dim - 1, 1.0),       # all 1.0 (sin(10π)=0)
            np.full(self.dim - 1, 0.1),       # small value
        ]
        
        # Clip tail configs to bounds
        tail_lo = lo[1:]
        tail_hi = hi[1:]
        tail_configs = [np.clip(cfg, tail_lo, tail_hi) for cfg in tail_configs]
        
        # For each x1, evaluate all tail configs and pick the best
        best_points = []
        for x1 in x1_vals:
            x1_clipped = np.clip(x1, lo[0], hi[0])
            best_f = np.inf
            best_tail_idx = 0
            
            for tail_idx, tail in enumerate(tail_configs):
                x = np.concatenate(([x1_clipped], tail))
                x_clipped = np.clip(x, lo, hi)
                f = problem(x_clipped)
                f1_val = f[0]
                if f1_val < best_f:
                    best_f = f1_val
                    best_tail_idx = tail_idx
            
            # Store the best point for this x1
            best_x = np.concatenate(([x1_clipped], tail_configs[best_tail_idx]))
            best_points.append(best_x)
        
        # We've used n_x1_samples * len(tail_configs) evaluations
        # If we have budget left, add more x1 samples or refine
        used_evals = n_x1_samples * len(tail_configs)
        remaining_budget = self.budget - used_evals
        
        if remaining_budget > 0:
            # Add more x1 samples with the best-performing tail config
            # Or sample randomly to explore
            n_extra = min(remaining_budget, 10)
            extra_x1_vals = np.random.uniform(0.0, 1.0, n_extra)
            for x1 in extra_x1_vals:
                x1_clipped = np.clip(x1, lo[0], hi[0])
                # Use the most common best tail
                # For simplicity, use all zeros
                x = np.concatenate(([x1_clipped], tail_configs[0]))
                x_clipped = np.clip(x, lo, hi)
                problem(x_clipped)
        
        return None
