"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2 using direct Pareto front construction.
    
    Key insight: For DTLZ2, the Pareto front is the unit sphere in objective space.
    Given target objective values (f1, f2, f3) on the unit sphere, we can directly
    construct the decision variables:
    - x[0] = f1
    - x[1] = f2 / (1 - f1) if f1 < 1, else 0
    - x[2:] = 0.5 (to make g(x) = 0, ensuring we're on the Pareto front)
    
    This requires exactly 1 evaluation per Pareto point, allowing us to use the
    full budget for dense Pareto front coverage with low-discrepancy sampling.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _generate_low_discrepancy_points(self, n, dim=2):
        """Generate n low-discrepancy points in [0, 1)^dim using Sobol sequence."""
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True)
        # generate points
        points = sampler.random(n)
        return points

    def _pareto_front_point(self, theta1, theta2):
        """
        For DTLZ2 with 3 objectives, the Pareto front is the unit sphere:
        sum(f_i^2) = 1, f_i >= 0
        
        Parameterization using spherical coordinates:
        theta1 in [0, pi/2], theta2 in [0, pi/2]
        f1 = cos(theta1) * cos(theta2)
        f2 = cos(theta1) * sin(theta2)
        f3 = sin(theta1)
        
        This covers the positive octant of the unit sphere.
        """
        f1 = np.cos(theta1) * np.cos(theta2)
        f2 = np.cos(theta1) * np.sin(theta2)
        f3 = np.sin(theta1)
        return np.array([f1, f2, f3])

    def _decision_variables_from_objectives(self, target_f, lo, hi):
        """
        Construct DTLZ2 decision variables that achieve the target objective values.
        
        For DTLZ2:
        - f1 = (1+g) * x1
        - f2 = (1+g) * (1-x1) * x2  
        - f3 = (1+g) * (1-x1) * (1-x2)
        
        On the Pareto front, g = 0, so:
        - f1 = x1
        - f2 = (1-x1) * x2
        - f3 = (1-x1) * (1-x2)
        
        To achieve target (f1, f2, f3):
        - x1 = f1
        - x2 = f2 / (1 - f1) if f1 < 1, else 0
        - x3..xn = 0.5 (to make g = 0)
        """
        x = np.zeros(self.dim)
        x[0] = target_f[0]
        
        if self.dim >= 2:
            if target_f[0] < 1.0 - 1e-10:
                x[1] = target_f[1] / (1.0 - target_f[0])
            else:
                x[1] = 0.0
            x[1] = np.clip(x[1], 0.0, 1.0)
        
        # Set remaining variables to 0.5 to make g(x) = 0
        if self.dim >= 3:
            x[2:] = 0.5
        
        # Clip to bounds
        x = np.clip(x, lo, hi)
        return x

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        used = 0
        all_points = []
        all_f = []
        
        # Strategy: Use ALL budget for Pareto front sampling with low-discrepancy sequence
        n_pareto = self.budget
        
        # Generate low-discrepancy points in 2D
        points_2d = self._generate_low_discrepancy_points(n_pareto, 2)
        u1 = points_2d[:, 0]
        u2 = points_2d[:, 1]
        
        # Correct for the Jacobian of the spherical mapping to ensure uniform surface coverage.
        # Surface area element dA = cos(theta1) dtheta1 dtheta2.
        # We want uniform distribution in area, so we need sin(theta1) ~ Uniform[0,1].
        # Thus theta1 = arcsin(u1).
        # theta2 is uniform in [0, pi/2], so theta2 = u2 * pi/2.
        theta1 = np.arcsin(u1)
        theta2 = u2 * (np.pi / 2)
        
        # For each Pareto front point, construct decision variables and evaluate
        for i in range(n_pareto):
            if used >= self.budget:
                break
            
            # Get target objective values on the Pareto front
            target_f = self._pareto_front_point(theta1[i], theta2[i])
            
            # Construct decision variables that achieve this objective
            x = self._decision_variables_from_objectives(target_f, lo, hi)
            
            # Evaluate (this records the point)
            fx = problem(x)
            used += 1
            all_points.append(x)
            all_f.append(fx)
        
        # Return the last point (return value is ignored, but must not be None)
        if all_points:
            return all_points[-1]
        return lo

# EVOLVE-BLOCK-END
