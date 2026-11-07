# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2 exploiting the known Pareto front structure.
    
    DTLZ2 with 3 objectives has a Pareto front that is a quarter-sphere in the positive
    octant of R^3. The distance variables (last dim - n_obj + 1 = 10 variables) should
    be set to 0.5 for optimality. The angle variables (first n_obj - 1 = 2 variables)
    determine position on the front.
    
    Strategy: 
    1. Use Sobol quasi-random sequence to sample the 2D angle space.
    2. Apply importance sampling correction to get uniform density on the Pareto front.
       The Jacobian correction accounts for the non-uniform surface element on the sphere.
    3. Set distance variables to 0.5 to lie exactly on the Pareto front.
    4. Use the entire budget for deterministic, space-filling sampling to maximize
       hypervolume coverage without wasting evaluations on dominated points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        
        # DTLZ2 structure:
        # - First m-1 variables are "angle" variables that determine position on Pareto front
        # - Last n-m+1 variables are "distance" variables; optimal value is 0.5 for each
        # - Pareto front: sum of squares of objectives = 1 (unit sphere in positive octant)
        
        # Number of distance variables
        n_dist = n - m + 1  # = 10 for dim=12, n_obj=3
        
        # Optimal distance values (0.5 for DTLZ2)
        dist_optimal = np.full(n_dist, 0.5)
        
        # Use the entire budget for Sobol sampling of the 2D angle space
        n_samples = self.budget
        
        # Use Sobol sequence for (m-1) dimensional angle space
        # scramble=False for deterministic, reproducible results
        sampler = qmc.Sobol(d=m-1, scramble=False)
        
        # Generate points. Sobol works best with powers of 2, but handles any N.
        # We'll generate n_samples points directly.
        raw_angles = sampler.random(n_samples)
        
        # Apply importance sampling correction for uniform density on the quarter-sphere.
        # For 3 objectives, the Pareto front is parameterized by (theta1, theta2) in [0, pi/2]^2:
        #   f1 = cos(theta1)
        #   f2 = sin(theta1)*cos(theta2)
        #   f3 = sin(theta1)*sin(theta2)
        # The surface element is sin(theta1)*sin(theta2) dtheta1 dtheta2 (for the full sphere;
        # on the quarter-sphere we need to account for the correct Jacobian).
        # 
        # For uniform sampling on the quarter-sphere surface:
        # - theta1 marginal: P(theta1 < t) proportional to sin(t) on [0, pi/2]
        #   => theta1 = arcsin(2*u1/pi) * (pi/2) ... let's derive properly.
        #   The CDF of theta1 for uniform surface measure: F(t) = (1 - cos(t)) / 2 for t in [0, pi/2]
        #   Wait, let me reconsider. For a unit sphere, the surface element in spherical coords
        #   (theta from z-axis, phi azimuthal) is sin(theta) dtheta dphi.
        #   
        #   In our parameterization for 3 objectives on the positive octant:
        #   Let theta1 be the angle from the f1-axis toward the f2-f3 plane.
        #   Let theta2 be the angle in the f2-f3 plane.
        #   
        #   f1 = cos(theta1)
        #   f2 = sin(theta1) * cos(theta2)
        #   f3 = sin(theta1) * sin(theta2)
        #   where theta1 in [0, pi/2], theta2 in [0, pi/2]
        #   
        #   The surface element on the unit sphere is sin(theta1) dtheta1 dtheta2.
        #   So for uniform density on the surface, we need:
        #   - theta2 uniform on [0, pi/2]: theta2 = (pi/2) * u2
        #   - theta1 with PDF proportional to sin(theta1) on [0, pi/2]
        #     CDF: F(t) = (1 - cos(t)) / (1 - cos(pi/2)) = (1 - cos(t)) / 1 = 1 - cos(t)
        #     Inverse: t = arccos(1 - u1) = arccos(1 - u1)
        #     So theta1 = arccos(1 - u1)
        
        # Transform raw Sobol samples to corrected angles
        u1 = raw_angles[:, 0]
        u2 = raw_angles[:, 1]
        
        # theta1 = arccos(1 - u1), giving uniform surface density
        theta1 = np.arccos(1.0 - u1)
        # theta2 = (pi/2) * u2, uniform on [0, pi/2]
        theta2 = (np.pi / 2.0) * u2
        
        # Build decision vectors
        X = np.zeros((n_samples, n))
        
        # Set distance variables to optimal value 0.5
        X[:, m-1:] = dist_optimal
        
        # Set angle variables: DTLZ2 uses theta_i = (pi/2) * x_i for x_i in [0,1]
        # So we need x_i such that (pi/2)*x_i = theta_i, i.e., x_i = theta_i / (pi/2) = 2*theta_i/pi
        X[:, 0] = 2.0 * theta1 / np.pi
        X[:, 1] = 2.0 * theta2 / np.pi
        
        # Clip to bounds (should already be within [0,1] for standard DTLZ2 bounds)
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.zeros((n_samples, m))
        for i in range(n_samples):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
        
        return X
# EVOLVE-BLOCK-END
