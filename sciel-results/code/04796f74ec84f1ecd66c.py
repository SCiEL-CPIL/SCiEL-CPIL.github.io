import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT6 Pareto front: x_2..x_n = 0, x_1 in [0, 1]
        # f1(x1) = 1 - (1 - exp(-4*x1)*sin(6*pi*x1)^6)^(1/10)
        # f2(x1) = 1 - sqrt(f1(x1))
        #
        # Parameterize by u = (1 - exp(-4*x1)*sin(6*pi*x1)^6)^(1/10)
        # so f1 = 1 - u, f2 = 1 - sqrt(1-u)
        # We need x1 such that u(x1) = target_u
        #
        # Strategy: build a fine grid of x1, compute u(x1), then for each target
        # u in [0,1] use bisection to find x1. But since u(x1) is not monotonic
        # (sin^6 oscillates), we instead directly sample x1 uniformly and evaluate.
        # However, the directive says to use a closed-form parametric representation
        # by sampling auxiliary variable u uniformly.
        #
        # Actually, let's re-read: "Use a closed-form parametric representation of the
        # ZDT6 Pareto front by directly sampling the auxiliary variable u (where f1 = 1-u)
        # uniformly in [0,1], bypassing the need for root-finding or grid searches to map
        # objective values to decision variables"
        #
        # The issue is that u(x1) is not injective on [0,1] because sin(6*pi*x1)^6 oscillates.
        # For a given u in [0,1], there are multiple x1 values. We want points ON the Pareto front.
        # Every x1 in [0,1] with x_2..x_n=0 is on the Pareto front.
        #
        # The hypervolume is maximized when we have a diverse set of points on the front.
        # Sampling x1 uniformly gives a uniform spread in x1, but f1 is not uniform in x1.
        # To maximize hypervolume with reference [11,11], we want points spread across the front.
        #
        # Let's just sample x1 uniformly in [0,1] and evaluate. This is simple and valid.
        # But the current program already does something similar with a lookup table.
        #
        # Let me try the approach: sample u uniformly in [0,1], and for each u, find an
        # x1 that achieves approximately that u value. Since u(x1) = (1-exp(-4*x1)*sin(6*pi*x1)^6)^(0.1),
        # and this is bounded in [0,1], we can use a precomputed table with interpolation.
        
        n_evals = self.budget
        dim = self.dim
        
        # Precompute a dense table of x1 -> u
        n_grid = 20000
        x1_grid = np.linspace(0, 1, n_grid)
        sin_term = np.sin(6 * np.pi * x1_grid)
        sin_pow6 = sin_term ** 6
        exp_term = np.exp(-4 * x1_grid)
        base = 1 - exp_term * sin_pow6
        base = np.clip(base, 0, 1)
        u_grid = base ** 0.1  # this is u where f1 = 1 - u
        
        # Sort by u
        sort_idx = np.argsort(u_grid)
        u_sorted = u_grid[sort_idx]
        x1_sorted = x1_grid[sort_idx]
        
        # Sample target u values uniformly in [0, 1]
        target_us = np.linspace(0, 1, n_evals)
        
        # For each target u, find x1 via searchsorted + linear interpolation
        idx = np.searchsorted(u_sorted, target_us)
        idx = np.clip(idx, 1, n_grid - 1)
        
        # Linear interpolation
        i_low = idx - 1
        i_high = idx
        
        u_low = u_sorted[i_low]
        u_high = u_sorted[i_high]
        
        # Avoid division by zero
        denom = u_high - u_low
        denom[denom < 1e-15] = 1e-15
        
        t = (target_us - u_low) / denom
        t = np.clip(t, 0, 1)
        
        x1_targets = x1_sorted[i_low] + t * (x1_sorted[i_high] - x1_sorted[i_low])
        x1_targets = np.clip(x1_targets, 0, 1)
        
        # Evaluate
        recorded = []
        for i in range(n_evals):
            x = np.zeros(dim)
            x[0] = x1_targets[i]
            obj = problem(x)
            recorded.append(obj)
        
        return recorded
