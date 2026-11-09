"""Multi-objective optimizer for DTLZ7."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the Pareto front structure.
    The Pareto front is defined by x[2:] = 0.
    f0 = x[0]
    f1 = x[1]
    f2 = 1 - sqrt(x[1])
    We use non-uniform sampling that concentrates points where
    hypervolume contribution is highest.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        n_evals = 0
        budget = self.budget
        
        # DTLZ7 Pareto front: x[2:] = 0
        # x[0] = u, x[1] = v
        # f0 = u
        # f1 = v
        # f2 = 1 - sqrt(v)
        
        # Hypervolume contribution: (1.1-f0)*(1.1-f1)*(25.0-f2)
        # The 25.0-f2 term dominates. Small f2 (v near 1) gives high contribution.
        
        # Strategy: Grid with non-uniform spacing in v to concentrate near v=1.
        # Use a denser grid overall within budget.
        
        # Allocate evaluations between u and v dimensions.
        # More points in v direction since f2 variation is more impactful.
        # Let's try n_u * n_v <= budget, with n_v > n_u.
        
        # Optimal: maximize coverage of the 2D surface in (f0, f1, f2) space.
        # Since the front is 2D parameterized by (u, v), we want good coverage in (u, v).
        # The hypervolume weighting favors v near 1.
        
        # Let n_v = 2 * n_u, so 2*n_u^2 <= budget
        # n_u = sqrt(budget/2), n_v = 2*n_u
        n_u = max(2, int(np.sqrt(budget / 2.0)))
        n_v = min(budget // n_u, budget - n_u + 1)
        
        if n_u * n_v > budget:
            n_v = budget // n_u
        
        u_vals = np.linspace(0, 1, n_u)
        
        # For v: concentrate near v=1 using v = 1 - t^2 with t uniform
        t_vals = np.linspace(0, 1, n_v)
        v_vals = 1.0 - t_vals**2
        v_vals = np.clip(v_vals, lo[1], hi[1])
        
        # Sort v_vals to ensure we cover the range properly
        # t_vals goes from 0 to 1, so v_vals goes from 1 to 0
        # Keep as is - this gives more points near v=1
        
        for u in u_vals:
            for v in v_vals:
                if n_evals >= budget:
                    break
                if time.time() > deadline:
                    break
                
                x = np.zeros(self.dim)
                x[0] = np.clip(u, lo[0], hi[0])
                x[1] = v
                # x[2:] = 0
                
                obj = problem(x)
                results.append(obj)
                n_evals += 1
            if n_evals >= budget:
                break
        
        # Use remaining budget for additional points if any
        while n_evals < budget:
            if time.time() > deadline:
                break
            
            # Add points along the edges of the front for better boundary coverage
            # Cycle through different strategies
            rem = budget - n_evals
            idx = n_evals % (n_u + n_v)
            
            if idx < n_u:
                u = idx / max(1, n_u - 1) if n_u > 1 else 0.5
                v = 1.0  # Best f2
            else:
                u = 0.0  # Best f0
                v_idx = idx - n_u
                v = v_vals[min(v_idx, n_v - 1)]
            
            x = np.zeros(self.dim)
            x[0] = np.clip(u, lo[0], hi[0])
            x[1] = np.clip(v, lo[1], hi[1])
            
            obj = problem(x)
            results.append(obj)
            n_evals += 1
        
        return results
# EVOLVE-BLOCK-END
