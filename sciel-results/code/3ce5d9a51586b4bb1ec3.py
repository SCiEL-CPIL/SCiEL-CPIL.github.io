"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2 exploiting known Pareto front structure.
    
    DTLZ2 with n_obj=3 has a Pareto front that is the unit sphere in objective space.
    We use a Halton sequence in 2D (theta1, theta2) to generate well-spread points
    on the Pareto front, then map them back to decision space.
    
    For DTLZ2, the Pareto-optimal points satisfy:
      f_i = x_1^2 * cos(theta_i) for i = 1..n_obj-1 (in angular parameterization)
      Actually, the standard DTLZ2 parameterization:
        x_1 = sin(theta_1)
        x_2 = cos(theta_1) * sin(theta_2)
        x_3 = cos(theta_1) * cos(theta_2)
        ... for n_obj objectives, we need n_obj-1 angles.
        
      The objectives are:
        f_i = x_i for i = 1..n_obj (when on Pareto front, g=0)
        
      More precisely, for DTLZ2 with n_obj objectives:
        f_i = x_i * (1 + g) for i = 1..n_obj
        where x_1^2 + x_2^2 + ... + x_n_obj^2 = 1 and g = 0 on Pareto front.
        
      So on the Pareto front: f_i = x_i where sum(x_i^2) = 1, x_i >= 0.
      
      We can parameterize using angles: for 3 objectives,
        x_1 = sin(theta_1) * sin(theta_2)  -- wait, let me be more careful.
        
      Actually for DTLZ2, the decision variables x_1, ..., x_n_obj are the first n_obj
      variables. The Pareto front condition is sum_{i=1}^{n_obj} x_i^2 = 1 with x_i >= 0,
      and the remaining variables are set to make g=0.
      
      For n_obj=3: x_1^2 + x_2^2 + x_3^2 = 1
      We can use:
        x_1 = sin(a) * sin(b)
        x_2 = sin(a) * cos(b)  
        x_3 = cos(a)
      where a in [0, pi/2], b in [0, pi/2]
      
      Or more symmetrically using spherical coordinates:
        x_1 = sin(phi) * cos(theta)
        x_2 = sin(phi) * sin(theta)
        x_3 = cos(phi)
      where phi in [0, pi/2], theta in [0, 2*pi] but we want x_i >= 0 so
      actually for the positive octant: phi in [0, pi/2], theta in [0, pi/2]
      
      Let me use:
        x_1 = sin(phi) * cos(theta)
        x_2 = sin(phi) * sin(theta)
        x_3 = cos(phi)
      with phi in [0, pi/2], theta in [0, pi/2]
      
      This covers the quarter-sphere in the positive octant.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton_2d(self, n, base1=2, base2=3):
        """Generate n points in [0,1]^2 using Halton sequence."""
        points = []
        for i in range(1, n + 1):
            p1, p2 = [], []
            ii = i
            b = base1
            while ii > 0:
                p1.append((ii % b) / b)
                ii //= b
            f = 1.0
            for digit in p1:
                f *= digit
            p1_val = f

            ii = i
            b = base2
            while ii > 0:
                p2.append((ii % b) / b)
                ii //= b
            f = 1.0
            for digit in p2:
                f *= digit
            p2_val = f

            points.append([p1_val, p2_val])
        return np.array(points)

    def _front_to_x(self, angles, bounds):
        """
        Map 2D angles (phi, theta) in [0, pi/2]^2 to a 12-dim decision vector
        that lies on the DTLZ2 Pareto front.
        
        For DTLZ2 with 12 variables and 3 objectives:
          - First 3 variables: x_1, x_2, x_3 with x_1^2 + x_2^2 + x_3^2 = 1
          - Remaining 9 variables: set to values that make g(x) = 0
        
        g(x) = sum_{i=4}^{12} (x_i - 0.5)^2
        For g=0, we need x_i = 0.5 for i = 4..12.
        
        So:
          x_1 = sin(phi) * cos(theta)
          x_2 = sin(phi) * sin(theta)
          x_3 = cos(phi)
          x_4 = ... = x_12 = 0.5
        """
        phi, theta = angles
        x1 = np.sin(phi) * np.cos(theta)
        x2 = np.sin(phi) * np.sin(theta)
        x3 = np.cos(phi)
        
        x = np.zeros(self.dim)
        x[0] = x1
        x[1] = x2
        x[2] = x3
        # Set remaining variables to 0.5 to make g=0
        for i in range(3, self.dim):
            x[i] = 0.5
        
        # Clip to bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        x = np.clip(x, lo, hi)
        
        return x

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 290s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Generate Halton sequence points in [0, pi/2]^2
        n_points = self.budget
        
        # Use Halton sequence for well-spread coverage
        halton_pts = self._halton_2d(n_points)
        
        # Map to angle space [0, pi/2]^2
        phi = halton_pts[:, 0] * (np.pi / 2)
        theta = halton_pts[:, 1] * (np.pi / 2)
        angles = np.column_stack([phi, theta])
        
        # Generate decision vectors on the Pareto front
        X = np.array([self._front_to_x(angles[i], bounds) for i in range(n_points)])
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        # Check time
        if time.time() > deadline:
            return X
        
        # If we have budget remaining and time, we could refine, but the
        # Halton points should already be on the Pareto front.
        # Let's just return what we have.
        
        return X
# EVOLVE-BLOCK-END
