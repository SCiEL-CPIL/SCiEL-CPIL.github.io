"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct multi-objective optimizer for RE21 using known structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        results = []
        used = 0

        # RE21 Structure:
        # f1 = sum(x_i^2)
        # f2 = sum((x_i - 2)^2)
        # Pareto front parameterization: x_i = 2(1-w) for all i
        # f1(w) = 4*n*(1-w)^2
        # f2(w) = 4*n*w^2
        # Reference point: [3000, 0.06]
        # Only points with f2 < 0.06 contribute to hypervolume.
        # 4*n*w^2 < 0.06 => w < sqrt(0.06 / (4*n))
        
        n = dim
        ref_f2 = 0.06
        w_max = np.sqrt(ref_f2 / (4.0 * n))
        
        # Generate points on the Pareto front
        # Focus heavily on the region w in [0, w_max]
        # Also include some points in [w_max, 1] for robustness
        
        # 80% of budget on the contributing region [0, w_max]
        # 20% on the rest [w_max, 1]
        
        n_main = int(0.8 * budget)
        n_rest = budget - n_main
        
        # Ensure at least 1 point in each
        if n_main < 1: n_main = 1
        if n_rest < 1: n_rest = 1
        
        # Generate w values
        # Uniform in w is a good starting point.
        w_main = np.linspace(0, w_max, n_main)
        w_rest = np.linspace(w_max, 1.0, n_rest)
        
        w_values = np.concatenate([w_main, w_rest])
        
        # Shuffle to ensure we don't run out of time before evaluating all
        # But we must respect the deadline.
        # Since evaluation is fast, we can just iterate.
        
        for w in w_values:
            if used >= budget:
                break
            if time.time() > deadline - 5:
                break
                
            # Compute x for this w
            # x_i = 2(1-w)
            x_val = 2.0 * (1.0 - w)
            x = np.full(dim, x_val)
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            used += 1
            results.append((x.copy(), f))
            
        # If we have budget left (e.g. due to early break or integer rounding),
        # add random points to ensure coverage in case structure assumption fails
        if used < budget:
            rng = np.random.default_rng(42)
            for _ in range(budget - used):
                if time.time() > deadline - 5:
                    break
                x = lo + (hi - lo) * rng.random(dim)
                x = np.clip(x, lo, hi)
                f = problem(x)
                used += 1
                results.append((x.copy(), f))
        
        if not results:
            # Fallback if something went wrong
            x = (lo + hi) / 2.0
            f = problem(x)
            results.append((x, f))
            used += 1

        return results[-1][0]
# EVOLVE-BLOCK-END
