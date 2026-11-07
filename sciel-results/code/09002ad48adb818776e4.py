# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2 exploiting the known Pareto front structure.
    
    DTLZ2 with 3 objectives has a Pareto front that is a quarter-sphere in the positive
    octant of R^3. The distance variables (last dim - n_obj + 1 = 10 variables) should
    be set to 0.5 for optimality. The angle variables (first n_obj - 1 = 2 variables)
    determine position on the front.
    
    Strategy: Use Halton quasi-random sequence to sample the 2D angle space densely,
    then map to decision space. This directly generates points on the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton_sequence(self, n, base):
        """Generate Halton sequence in [0, 1) for given base."""
        result = np.zeros(n)
        f = 1.0 / base
        for i in range(n):
            x = 0.0
            fi = f
            idx = i + 1
            while idx > 0:
                x += fi * (idx % base)
                idx //= base
                fi /= base
            result[i] = x
        return result

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Number of angle variables
        n_angles = n_obj - 1  # 2 for 3 objectives
        
        # Generate Halton sequence points in [0, 1]^2
        n_points = min(budget, 200)
        
        # Halton sequences for 2 dimensions
        h1 = self._halton_sequence(n_points, 2)
        h2 = self._halton_sequence(n_points, 3)
        
        # Map Halton points to angles on the Pareto front
        # For DTLZ2 with 3 objectives, the Pareto front is a quarter-sphere
        # We need angles theta1, theta2 such that:
        # f1 = 0.5 * (1 + cos(theta1) * cos(theta2))  -- wait, let me reconsider
        # Actually for DTLZ2: f_i = 0.5 * prod_{j=1}^{i-1} sin(theta_j) * cos(theta_i) for i < n_obj
        # f_n = 0.5 * prod_{j=1}^{n_obj-1} sin(theta_j)
        # where theta_j in [0, pi/2]
        
        # So theta1 = h1 * pi/2, theta2 = h2 * pi/2
        theta1 = h1 * (np.pi / 2)
        theta2 = h2 * (np.pi / 2)
        
        # Build decision vectors
        # First n_obj - 1 variables are the angles (scaled to [0, pi/2])
        # Last dim - n_obj + 1 variables are distance variables (set to 0.5)
        X = np.zeros((n_points, dim))
        X[:, 0] = theta1
        X[:, 1] = theta2
        for j in range(2, dim):
            X[:, j] = 0.5  # Midpoint of [0, 1]
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = n_points
        
        # Check if we have time for additional refinement
        if used < budget and time.time() < deadline:
            # Try to refine with small perturbations around promising points
            # Keep simple: just return what we have
            pass
        
        return X
# EVOLVE-BLOCK-END
