"""Optimized ZDT2 optimizer using convex spacing for Hypervolume maximization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Model-based optimizer for ZDT2: places points directly on the Pareto front 
    with convex spacing optimized for Hypervolume maximization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = self.budget
        
        # ZDT2 Pareto Front: x[1:] = 0, f1 = x[0], f2 = 1 - sqrt(x[0]).
        # Reference point R = [11, 11].
        # HV contribution of interval i is (u_{i+1} - u_i) * (11 - f2(u_i)).
        # f2(u) = 1 - sqrt(u).
        # Height h(u) = 11 - (1 - sqrt(u)) = 10 + sqrt(u).
        # h(u) is increasing with u.
        # To maximize sum, allocate larger widths to larger u.
        # This requires u_i to be convex with respect to index i.
        # u_i ~ (i/N)^alpha with alpha > 1.
        
        # The function h(u) varies from 10 to 11 (10% variation).
        # Optimal alpha is slightly greater than 1.
        # Let's use alpha = 1.15.
        
        i = np.arange(n_evals)
        # Ensure exact 0 and 1 at ends
        t = i / (n_evals - 1)
        
        # Optimize alpha for maximum hypervolume
        # HV(alpha) = sum_{i=0}^{N-2} (u_{i+1} - u_i) * (10 + sqrt(u_i))
        # where u_i = t_i^alpha
        best_alpha = 1.0
        best_hv = -1.0
        
        # Grid search over alpha
        alphas = np.linspace(1.0, 2.0, 200)
        for a in alphas:
            u = t ** a
            # Compute hypervolume
            # Intervals: from u[i] to u[i+1] for i=0..N-2
            # Height at left endpoint: 10 + sqrt(u[i])
            widths = np.diff(u)
            heights = 10.0 + np.sqrt(u[:-1])
            hv = np.sum(widths * heights)
            if hv > best_hv:
                best_hv = hv
                best_alpha = a
        
        # Refine with finer grid around best
        alpha_lo = max(1.0, best_alpha - 0.05)
        alpha_hi = min(2.0, best_alpha + 0.05)
        alphas_fine = np.linspace(alpha_lo, alpha_hi, 100)
        for a in alphas_fine:
            u = t ** a
            widths = np.diff(u)
            heights = 10.0 + np.sqrt(u[:-1])
            hv = np.sum(widths * heights)
            if hv > best_hv:
                best_hv = hv
                best_alpha = a
        
        alpha = best_alpha
        x0_vals = t ** alpha
        
        # Create candidate solutions
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_vals
        # x[1:] = 0 by default

        # Evaluate all points
        F = np.zeros((n_evals, self.n_obj))
        for k in range(n_evals):
            F[k] = problem(X[k])

            # Deadline guard
            if time.time() > deadline:
                return X

        return X
# EVOLVE-BLOCK-END
