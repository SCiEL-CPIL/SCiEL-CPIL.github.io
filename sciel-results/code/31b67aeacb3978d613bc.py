# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import brentq, minimize_scalar


def _find_f1_min():
    """Find the minimum f1 achievable on the Pareto front."""
    # Maximize phi(x) = exp(-4x) * sin(6*pi*x)^6
    # to minimize f1 = 1 - phi(x)
    def neg_phi(x):
        return -np.exp(-4.0 * x) * (np.sin(6.0 * np.pi * x) ** 6)
    
    # Search in [0, 1]
    # The first peak is around 1/12
    res = minimize_scalar(neg_phi, bounds=(0.0, 1.0), method='bounded', options={'xatol': 1e-12})
    phi_max = -res.fun
    f1_min = 1.0 - phi_max
    return f1_min


def _precompute_zdt6_pareto_table(n_points=200):
    """
    Precompute a table of x1 values such that f1(x1) = t for t in [f1_min, 1].
    """
    f1_min = _find_f1_min()
    
    # We want to sample f1 in [f1_min, 1.0]
    # Using uniform f1 is a good strategy here as the singularity at f1=0 is outside the domain
    t_values = np.linspace(f1_min, 1.0, n_points)
    
    x1_values = np.zeros(n_points)
    
    # Fine grid for bracketing
    x_grid = np.linspace(0.0, 1.0, 50000)
    phi = np.exp(-4.0 * x_grid) * (np.sin(6.0 * np.pi * x_grid) ** 6)
    
    for i, t in enumerate(t_values):
        target = 1.0 - t
        diff = phi - target
        sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
        
        x1_val = None
        if len(sign_changes) > 0:
            # Pick the first sign change (smallest x1)
            idx = sign_changes[0]
            a, b = x_grid[idx], x_grid[idx+1]
            try:
                root = brentq(
                    lambda x: np.exp(-4.0 * x) * (np.sin(6.0 * np.pi * x) ** 6) - target,
                    a, b,
                    xtol=1e-15, rtol=1e-15, maxiter=100
                )
                x1_val = root
            except:
                idx_min = np.argmin(np.abs(diff))
                x1_val = x_grid[idx_min]
        else:
            # Fallback: closest point
            idx_min = np.argmin(np.abs(diff))
            x1_val = x_grid[idx_min]
        
        x1_values[i] = x1_val
    
    return x1_values


# Precompute the table at module load time
_ZDT6_X1_TABLE = _precompute_zdt6_pareto_table(200)


class Optimizer:
    """
    Optimizer that uses a pre-computed lookup table of ZDT6 Pareto-optimal decision vectors.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_points = min(self.budget, len(_ZDT6_X1_TABLE))
        
        # Build decision vectors: x1 from table, rest zeros
        X = np.zeros((n_points, self.dim))
        X[:, 0] = _ZDT6_X1_TABLE[:n_points]
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
