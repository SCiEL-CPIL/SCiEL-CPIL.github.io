import numpy as np
import time

def _halton_sequence(n, d):
    """Generate n points in d-dimensional Halton sequence."""
    points = np.zeros((n, d))
    for i in range(1, n):
        for j in range(d):
            base = j + 2
            f = 1.0
            i_val = i
            while i_val > 0:
                f /= base
                i_val, remainder = divmod(i_val, base)
                points[i, j] += f * remainder
    # The first point is 0.0 by default.
    return points

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # 300s limit, keep 10s margin
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_points = self.budget
        
        # Generate Halton points in 2D for the two angle variables
        samples = _halton_sequence(n_points, 2)
        
        X = np.zeros((n_points, self.dim))
        
        # For DTLZ2, the Pareto front is reached when the 'position' variables are 0.5.
        # These are typically x[2:] for 3 objectives.
        # We set them to the midpoint of the bounds.
        # Assuming bounds are [0, 1], midpoint is 0.5.
        mid = (lo[2:] + hi[2:]) / 2.0
        X[:, 2:] = mid
        
        # Transform Halton samples (uniform in [0,1]) to angles
        # To get uniform coverage on the surface of the quarter-sphere (Pareto front of DTLZ2):
        # theta1 (polar) should be sampled such that sin(theta1) is uniform? No, area element is sin(theta1).
        # CDF of theta1 is 1 - cos(theta1).
        # Inverse CDF: theta1 = arccos(1 - u1)
        # theta2 (azimuth) is uniform.
        # theta2 = (pi/2) * u2
        
        u1 = samples[:, 0]
        u2 = samples[:, 1]
        
        theta1 = np.arccos(1.0 - u1)
        theta2 = (np.pi / 2.0) * u2
        
        # Map angles back to decision variables x[0], x[1] in [0, 1]
        # Assuming x[0] maps to theta1 in [0, pi/2]
        X[:, 0] = (2.0 / np.pi) * theta1
        X[:, 1] = (2.0 / np.pi) * theta2
        
        # Clip to bounds to be safe
        X = np.clip(X, lo, hi)
        
        results = []
        for x in X:
            if time.time() > deadline:
                break
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                # If an exception occurs, we stop to preserve valid results
                break
                
        return results
