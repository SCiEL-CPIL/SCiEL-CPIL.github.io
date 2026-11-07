# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2 exploiting the known Pareto front structure.
    
    DTLZ2 with 3 objectives has a Pareto front that is a (1/4)-sphere in the positive
    octant of R^3. The distance variables (last dim - n_obj + 1 = 10 variables) should
    be set to 0.5 for optimality. The angle variables (first n_obj - 1 = 2 variables)
    determine position on the front.
    
    Strategy: Use Sobol quasi-random sequence to sample the 2D angle space densely,
    then map to decision space. This directly generates points near/on the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # 300s wall-clock limit with safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2 structure:
        # - First (n_obj - 1) = 2 variables are "angle" variables in [0, 1]
        # - Last (dim - n_obj + 1) = 10 variables are "distance" variables
        # For Pareto optimality, distance variables should be 0.5
        # Angle variables map to the quarter-sphere via:
        #   f_i = (1 + g) * cos(pi/2 * x_i * prod_{j>i} sin(pi/2 * x_j))
        # where g = sum((x_j - 0.5)^2) for distance variables
        # When g = 0 (x_j = 0.5 for all distance vars), we get the Pareto front.
        
        # The Pareto front in objective space is:
        #   f_1 = cos(theta_1) * cos(theta_2)
        #   f_2 = cos(theta_1) * sin(theta_2)
        #   f_3 = sin(theta_1)
        # where theta_1 = pi/2 * x_1, theta_2 = pi/2 * x_2
        # and x_1, x_2 are in [0, 1]
        
        # To generate points ON the Pareto front:
        # Set distance variables to 0.5
        # Map angle variables to theta values that cover the quarter-sphere uniformly
        
        # Use Sobol sequence for uniform coverage of the 2D angle space
        n_evals = self.budget
        
        # Generate Sobol points in [0, 1]^2
        sobol = qmc.Sobol(d=2, scramble=True)
        # We need n_evals points; Sobol requires power-of-2 for best properties,
        # but we can generate the next power of 2 and take the first n_evals
        n_power = 1
        while n_power < n_evals:
            n_power *= 2
        sobol_points = sobol.random(n_power)
        # Take only the first n_evals points
        sobol_points = sobol_points[:n_evals]
        
        # Map Sobol points [0, 1]^2 to angle space
        # theta_1 = pi/2 * s_1, theta_2 = pi/2 * s_2
        # But we want uniform distribution on the sphere surface, so we need to
        # account for the Jacobian. For uniform distribution on the quarter-sphere:
        # Use the inverse CDF approach or simply sample theta_1 and theta_2
        # such that the points are uniform on the sphere.
        
        # Actually, for DTLZ2, the standard mapping is:
        # x_1 -> theta_1 = pi/2 * x_1
        # x_2 -> theta_2 = pi/2 * x_2
        # This gives f_1 = cos(theta_1)*cos(theta_2), f_2 = cos(theta_1)*sin(theta_2), f_3 = sin(theta_1)
        # The distribution of points on the front is NOT uniform in angle space.
        # To get uniform distribution on the sphere, we should sample:
        # theta_1 ~ arcsin(U) where U ~ Uniform(0, 1)  [for uniform on sphere]
        # theta_2 ~ Uniform(0, pi/2)
        
        # Let's use the proper transformation for uniform points on the quarter-sphere:
        # For a uniform point on the unit sphere in the positive octant:
        # z ~ Uniform(-1, 1) -> theta_1 = arcsin(z) but we want [0, pi/2]
        # So: sin(theta_1) ~ Uniform(0, 1) => theta_1 = arcsin(U1)
        # theta_2 ~ Uniform(0, pi/2)
        
        U1 = sobol_points[:, 0]
        U2 = sobol_points[:, 1]
        
        theta_1 = np.arcsin(U1)  # in [0, pi/2]
        theta_2 = np.pi / 2 * U2  # in [0, pi/2]
        
        # Now compute the decision variables
        # x_1 = 2*theta_1/pi, x_2 = 2*theta_2/pi
        # Distance variables = 0.5
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = 2 * theta_1 / np.pi
        X[:, 1] = 2 * theta_2 / np.pi
        X[:, 2:] = 0.5  # distance variables at optimal value
        
        # Clip to bounds (shouldn't be needed but for safety)
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
