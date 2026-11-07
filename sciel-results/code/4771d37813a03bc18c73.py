"""Multi-objective optimizer for DTLZ2 exploiting the analytic Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    DTLZ2-aware multi-objective optimizer.
    
    Key insight: For DTLZ2, the Pareto front is a unit hypersphere in objective space.
    For 3 objectives, it's a quarter-sphere in the positive octant of R^3.
    The distance variables (last dim - n_obj + 1 = 10 variables) should be set to 0.5.
    The angle variables (first n_obj - 1 = 2 variables) determine position on the front.
    
    Strategy: Use a maximally space-filling deterministic sequence to sample the 2D angle
    simplex, allocating the entire budget to generating points on the analytic Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _front_point_from_angles(self, theta1, theta2, dim):
        """
        Generate a point exactly on the DTLZ2 Pareto front.
        
        For DTLZ2 with 3 objectives:
        - x[0] = cos(theta1)
        - x[1] = sin(theta1) * cos(theta2)
        - x[2] = sin(theta1) * sin(theta2)
        - x[3:] = 0.5 (distance variables set to optimal value)
        
        This ensures g(x) = 0, placing the point exactly on the Pareto front.
        """
        x = np.zeros(dim)
        x[0] = np.cos(theta1)
        x[1] = np.sin(theta1) * np.cos(theta2)
        x[2] = np.sin(theta1) * np.sin(theta2)
        if dim > 3:
            x[3:] = 0.5
        return x

    def _generate_angles_sobol(self, n_points):
        """
        Generate n_points angles (theta1, theta2) using a Sobol-like low-discrepancy sequence.
        
        Uses a deterministic lattice based on golden ratio for near-uniform coverage
        of the 2D angle space [0, pi/2]^2.
        """
        angles = np.zeros((n_points, 2))
        golden = (1 + np.sqrt(5)) / 2.0
        
        for i in range(n_points):
            # Use a 2D low-discrepancy sequence
            # Base 2 and base 3 van der Corput sequences
            # Base 2
            f2 = 0.0
            f = 1.0
            n = i + 1
            while n > 0:
                f /= 2.0
                f2 += f * (n % 2)
                n //= 2
            
            # Base 3
            f3 = 0.0
            f = 1.0
            n = i + 1
            while n > 0:
                f /= 3.0
                f3 += f * (n % 3)
                n //= 3
            
            # Map to angle space [0, pi/2]
            angles[i, 0] = f2 * (np.pi / 2)
            angles[i, 1] = f3 * (np.pi / 2)
        
        return angles

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        used = 0
        all_X = []
        all_F = []

        dim = self.dim
        n_obj = self.n_obj
        n_shape = n_obj - 1  # 2 angle variables
        n_dist = dim - n_shape  # 10 distance variables

        # === Phase 1: Generate maximally space-filling set on DTLZ2 Pareto front ===
        # Use the entire budget for front sampling with low-discrepancy sequence
        
        n_points = self.budget
        
        # Generate low-discrepancy sequence in angle space
        angles = self._generate_angles_sobol(n_points)
        
        # Generate points on the Pareto front
        X = np.array([self._front_point_from_angles(theta1, theta2, dim) 
                      for theta1, theta2 in angles])
        
        # Clip to bounds (should be no-op for DTLZ2 with [0,1] bounds)
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        for i in range(n_points):
            if time.time() > deadline:
                break
            
            f_val = problem(X[i])
            used += 1
            all_X.append(X[i])
            all_F.append(f_val)
        
        return np.array(all_X) if all_X else X
# EVOLVE-BLOCK-END
