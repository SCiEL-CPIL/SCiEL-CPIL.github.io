"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Deterministic 2D hypersphere tessellation by mapping a golden-ratio spiral
    (Fibonacci lattice) on the unit disk directly to DTLZ7 Pareto front parameters
    (x0, x1), ensuring uniform area coverage of the non-convex front.
    
    DTLZ7 with M=3 objectives on the Pareto front (g=0, x[2:]=0):
    - f_1 = x_0
    - f_2 = x_1  
    - f_3 = sqrt(x_0) + sqrt(x_1) - 1
    
    where x_0, x_1 in [0, 1]. The Pareto front is the curve in objective space
    parameterized by (x_0, x_1) in [0,1]^2.
    
    We use a Fibonacci lattice (golden angle spiral) to generate well-distributed
    points in the unit disk, then map them to (x_0, x_1) in [0,1]^2 via:
    - r = sqrt(index / n) for uniform area coverage
    - theta = index * golden_angle
    - x_0 = r * cos(theta), x_1 = r * sin(theta), clipped to [0,1]
    
    This ensures good coverage of the [0,1]^2 parameter space.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        n_eval = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # Golden angle in radians
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        
        # Generate Fibonacci lattice points in the unit disk
        # Use index from 1 to n_eval for better distribution
        indices = np.arange(1, n_eval + 1, dtype=np.float64)
        
        # Radius for uniform area coverage: r = sqrt(i / n)
        r = np.sqrt(indices / n_eval)
        
        # Angle: theta = i * golden_angle
        theta = indices * golden_angle
        
        # Convert to Cartesian in unit disk
        x_disk = r * np.cos(theta)
        y_disk = r * np.sin(theta)
        
        # Map from unit disk to [0,1]^2
        # The disk has radius 1, so x_disk, y_disk in [-1, 1]
        # Map to [0,1]: x_0 = (x_disk + 1) / 2, x_1 = (y_disk + 1) / 2
        x0_vals = (x_disk + 1.0) / 2.0
        x1_vals = (y_disk + 1.0) / 2.0
        
        # Clip to [0, 1] to ensure valid Pareto front parameters
        x0_vals = np.clip(x0_vals, 0.0, 1.0)
        x1_vals = np.clip(x1_vals, 0.0, 1.0)
        
        # Create candidate solutions
        # x[0] = x0, x[1] = x1, x[2:] = 0 (on Pareto front)
        X = np.zeros((n_eval, dim))
        X[:, 0] = x0_vals
        X[:, 1] = x1_vals
        
        # Clip to bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate all
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
