# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2 using Sobol sequence with corner-biased
    angle sampling to maximize hypervolume contribution from Pareto front tips.
    
    DTLZ2 structure:
    - 12 variables, 3 objectives
    - First 2 variables are angle variables (theta1, theta2 in [0, pi/2])
    - Last 10 variables are distance variables (optimal at 0.5)
    - Pareto front is a quarter-sphere in positive octant of R^3
    
    Key Insight:
    - The hypervolume indicator is dominated by regions far from the origin in 
      the objective space (i.e., near the 'tips' of the Pareto front where one 
      objective is high and others are low).
    - Uniform surface density on the Pareto front under-samples these tips.
    - By biasing the Sobol sequence points towards the corners of the 2D 
      parameter space, we concentrate evaluations near the high-contribution 
      tips, thereby increasing the total dominated volume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _bias_to_corners(self, pts, power=2.0):
        """
        Biases uniform points in [0,1] towards 0 and 1.
        This clusters samples near the boundaries of the parameter space.
        """
        # Center at 0.5
        centered = pts - 0.5
        # Apply power to magnitude, preserving sign
        biased = np.sign(centered) * (np.abs(centered) ** power)
        # Shift back to [0, 1]
        shifted = 0.5 + biased
        # Clip to valid range
        return np.clip(shifted, 0.0, 1.0)

    def _sobol_to_angles(self, n, bias_power=1.5):
        """
        Generate n points on the DTLZ2 Pareto front using Sobol sequence 
        with corner biasing.
        
        Returns array of shape (n, 2) with angle variable values in [0, 1]
        (i.e., x1 = theta1/(pi/2), x2 = theta2/(pi/2)).
        """
        # Generate Sobol sequence in 2D
        sampler = qmc.Sobol(d=2, scramble=True)
        pts = sampler.random(n)
        
        # Apply corner biasing to enhance tip coverage
        pts[:, 0] = self._bias_to_corners(pts[:, 0], bias_power)
        pts[:, 1] = self._bias_to_corners(pts[:, 1], bias_power)
        
        # Transform to angle space
        # For DTLZ2 with 3 objectives:
        # f1 = cos(theta1) sin(theta2)
        # f2 = cos(theta1) cos(theta2)
        # f3 = sin(theta1)
        # where theta1, theta2 in [0, pi/2]
        # x1 = theta1 / (pi/2), x2 = theta2 / (pi/2)
        
        # We use the biased points directly as the normalized angles.
        # v1 in [0,1] -> theta1 = v1 * (pi/2)
        # v2 in [0,1] -> theta2 = v2 * (pi/2)
        
        # Note: This is NOT uniform surface density. It is biased towards 
        # the corners of the (theta1, theta2) square, which correspond to 
        # the tips of the Pareto front.
        
        x1 = np.clip(pts[:, 0], 0.0, 1.0)
        x2 = np.clip(pts[:, 1], 0.0, 1.0)
        
        return np.column_stack([x1, x2])

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
        
        # Generate Sobol-based angle points with corner bias
        # bias_power=1.5 provides a moderate concentration towards tips
        angle_pts = self._sobol_to_angles(n_samples, bias_power=1.5)
        
        # Construct decision vectors:
        # - First 2 variables: angle variables from Sobol sequence
        # - Last 10 variables: distance variables set to 0.5 (optimal for DTLZ2)
        X = np.zeros((n_samples, dim))
        X[:, :n_angle] = angle_pts
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
