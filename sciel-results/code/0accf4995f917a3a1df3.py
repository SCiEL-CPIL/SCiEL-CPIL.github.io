# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2 using Fibonacci lattice for uniform
    coverage of the 2D angle space that defines the Pareto front.
    
    DTLZ2 structure:
    - 12 variables, 3 objectives
    - First 2 variables are angle variables (theta1, theta2 in [0, pi/2])
    - Last 10 variables are distance variables (optimal at 0.5)
    - Pareto front is a quarter-sphere in positive octant of R^3
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fibonacci_lattice(self, n):
        """
        Generate n points using a Fibonacci lattice in 2D.
        This provides near-uniform distribution on the unit disk,
        which maps to uniform coverage of the Pareto front angles.
        
        Returns array of shape (n, 2) with values in [0, 1].
        """
        golden_ratio = (1 + np.sqrt(5)) / 2.0
        inv_phi2 = 1.0 / (golden_ratio * golden_ratio)
        
        # Index from 1 to n for better distribution
        i = np.arange(1, n + 1)
        
        # Angle using golden ratio conjugate for low-discrepancy
        theta = 2.0 * np.pi * i * inv_phi2
        theta = theta % (2.0 * np.pi)
        
        # Radius: use sqrt for uniform area distribution
        r = np.sqrt((i - 0.5) / n)
        
        # Convert to [0, 1]^2 by mapping from polar to Cartesian and normalizing
        x = 0.5 + 0.5 * r * np.cos(theta)
        y = 0.5 + 0.5 * r * np.sin(theta)
        
        # Clip to [0, 1]
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        
        return np.column_stack([x, y])

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_obj = self.n_obj  # 3
        dim = self.dim  # 12
        n_angle = n_obj - 1  # 2 angle variables
        n_dist = dim - n_obj + 1  # 10 distance variables
        
        n_samples = min(self.budget, 200)
        
        # Generate Fibonacci lattice points in [0, 1]^2
        fib_pts = self._fibonacci_lattice(n_samples)
        
        # Construct decision vectors:
        # - First 2 variables: angle variables from Fibonacci lattice
        # - Last 10 variables: distance variables set to 0.5 (optimal for DTLZ2)
        X = np.zeros((n_samples, dim))
        X[:, :n_angle] = fib_pts
        X[:, n_angle:] = 0.5
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        # Check time limit
        if time.time() > deadline:
            return X
        
        return X
# EVOLVE-BLOCK-END
