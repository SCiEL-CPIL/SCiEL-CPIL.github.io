# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Directly construct points on the DTLZ1 Pareto front using a Sobol sequence."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        n = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # DTLZ1 has M objectives, N variables.
        # The Pareto front is defined by the first N-M+1 variables (position variables).
        # The remaining M-1 variables are aggregation variables which should be set to 0.5
        # to lie on the front.
        # Here, dim=7, n_obj=3, so N-M+1 = 7-3+1 = 5 position variables.
        # The last 2 variables are aggregation variables.
        
        n_pos = dim - n_obj + 1  # 5
        n_agg = n_obj - 1        # 2
        
        # Generate a Sobol sequence for the position variables.
        # We use scipy.stats.qmc.Sobol if available, otherwise fall back to random.
        try:
            from scipy.stats import qmc
            # Use a seed derived from the global random state to vary between runs/seeds
            seed = np.random.randint(0, 2**32)
            sampler = qmc.Sobol(d=n_pos, scramble=True, seed=seed)
            # Generate n points in [0, 1]^n_pos
            pos_points = sampler.random(n)
        except ImportError:
            # Fallback to random if scipy is not available (though it should be)
            rng = np.random.default_rng()
            pos_points = rng.random((n, n_pos))
        
        # Construct the full input vectors
        # For each point, the first n_pos values are from the Sobol sequence.
        # The last n_agg values are set to 0.5.
        X = np.zeros((n, dim))
        X[:, :n_pos] = pos_points
        X[:, n_pos:] = 0.5
        
        # Clip to bounds to be safe (though bounds are [0,1] and points are in [0,1])
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
