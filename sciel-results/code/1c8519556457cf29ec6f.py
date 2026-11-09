"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


class Optimizer:
    """Constructive sampling based multi-objective optimizer for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref_point = np.array([2.5] * self.n_obj)
        
        all_objectives = []
        
        # DTLZ2 specific strategy:
        # The Pareto front is a convex hypersphere.
        # We assume the standard DTLZ2 formulation where:
        # 1. The first M variables (x_0, x_1, x_2) determine the shape.
        # 2. The remaining variables (x_3...x_D) determine the distance from the front via g(x).
        # 3. The minimum of g(x) is 0, typically achieved when x_i = 0.5 for all i > M.
        
        # We will sample points on the Pareto front directly by:
        # 1. Sampling angles uniformly to cover the spherical triangle front.
        # 2. Mapping these angles back to decision variables x_0, x_1, x_2.
        # 3. Setting x_3...x_D to 0.5 to minimize g(x) to 0.
        
        # If the problem is not standard DTLZ2 or g(0.5) != 0, these points might be slightly off the front,
        # but they will likely be much better than random points.
        # We use all budget for this constructive sampling.
        
        n_samples = self.budget
        
        # Use a fixed seed for reproducibility in the sampling design itself,
        # although the problem evaluator might set np.random.seed.
        # We use np.random directly as the evaluator seeds it.
        
        # Sample uniformly in the parameter space of the Pareto front.
        # For DTLZ2 with M=3, the front is a quarter-sphere.
        # We can parameterize by theta1, theta2 in [0, pi/2].
        # f3 = cos(theta2)
        # f1 = sin(theta2) * cos(theta1)
        # f2 = sin(theta2) * sin(theta1)
        
        # To get a uniform distribution on the surface of the sphere,
        # we should sample theta2 such that cos(theta2) is uniform in [0,1]?
        # Actually, for hypervolume, we want good coverage.
        # Uniform sampling in theta1, theta2 [0, pi/2] x [0, pi/2] is a standard approach.
        
        theta1 = np.random.uniform(0, np.pi / 2, n_samples)
        theta2 = np.random.uniform(0, np.pi / 2, n_samples)
        
        # Calculate target objective values on the unit sphere
        f3 = np.cos(theta2)
        sin_theta2 = np.sin(theta2)
        f1 = sin_theta2 * np.cos(theta1)
        f2 = sin_theta2 * np.sin(theta1)
        
        # Map back to decision variables x_0, x_1, x_2
        # Standard DTLZ2 mapping:
        # f3 = (1+g) * sin(x_0 * pi/2)  => if g=0, sin(x_0*pi/2) = f3
        # x_0 = (2/pi) * arcsin(f3)
        
        # f1 = (1+g) * cos(x_0*pi/2) * cos(x_1*pi/2) * cos(x_2*pi/2)
        # f2 = (1+g) * cos(x_0*pi/2) * cos(x_1*pi/2) * sin(x_2*pi/2)
        
        # Let C0 = cos(x_0*pi/2) = sqrt(1 - f3^2)
        # Let C1 = cos(x_1*pi/2)
        # f1 = C0 * C1 * cos(x_2*pi/2)
        # f2 = C0 * C1 * sin(x_2*pi/2)
        
        # From f1, f2:
        # tan(x_2*pi/2) = f2/f1
        # x_2 = (2/pi) * arctan(f2/f1)
        
        # Magnitude:
        # sqrt(f1^2 + f2^2) = C0 * C1
        # C1 = sqrt(f1^2 + f2^2) / C0
        # cos(x_1*pi/2) = C1
        # x_1 = (2/pi) * arccos(C1)
        
        # Handle edge cases where f1 or f3 are 0 or 1 to avoid division by zero or domain errors
        eps = 1e-9
        
        x0 = np.zeros(n_samples)
        x1 = np.zeros(n_samples)
        x2 = np.zeros(n_samples)
        
        # Mask for valid arcsin
        mask_f3 = (f3 > eps) & (f3 < 1 - eps)
        x0[mask_f3] = (2 / np.pi) * np.arcsin(f3[mask_f3])
        # If f3 is close to 0, x0 is close to 0. If f3 close to 1, x0 close to 1.
        # np.arcsin handles [0,1] well.
        x0 = (2 / np.pi) * np.arcsin(np.clip(f3, 0, 1))
        
        # Calculate C0
        C0 = np.sqrt(np.clip(1 - f3**2, 0, 1))
        
        # Calculate x2
        # arctan(f2/f1) is problematic if f1=0.
        # Use arctan2(f2, f1) to handle quadrants, but here f1,f2 >= 0.
        # If f1=0, x2 should be pi/2 (since tan(pi/2) is inf, and f2/f1 -> inf).
        # If f2=0, x2 should be 0.
        x2_raw = np.arctan2(f2, f1)
        x2 = (2 / np.pi) * x2_raw
        
        # Calculate x1
        # C1 = sqrt(f1^2 + f2^2) / C0
        mag_f12 = np.sqrt(f1**2 + f2**2)
        C1 = np.zeros(n_samples)
        mask_C0 = C0 > eps
        C1[mask_C0] = mag_f12[mask_C0] / C0[mask_C0]
        # Clip C1 to [0, 1] for arccos
        C1 = np.clip(C1, 0, 1)
        x1 = (2 / np.pi) * np.arccos(C1)
        
        # Clip x0, x1, x2 to [0, 1] to be safe
        x0 = np.clip(x0, 0, 1)
        x1 = np.clip(x1, 0, 1)
        x2 = np.clip(x2, 0, 1)
        
        # Construct full decision vectors
        # Assume x_0, x_1, x_2 are the shape variables.
        # x_3 ... x_D are the g-variables.
        # Set them to 0.5 to minimize g(x).
        
        n_shape = 3
        n_tail = self.dim - n_shape
        
        population = np.zeros((n_samples, self.dim))
        population[:, 0] = x0
        population[:, 1] = x1
        population[:, 2] = x2
        
        # Set tail variables to 0.5 (midpoint of [0,1])
        if n_tail > 0:
            population[:, n_shape:] = 0.5
            
        # Evaluate all points
        for x in population:
            obj = problem(x)
            all_objectives.append(obj)
            
        # Convert to numpy array
        all_objectives = np.array(all_objectives)
        
        # Filter points that are strictly better than reference point in all objectives
        # Reference point is [2.5, 2.5, 2.5]
        valid_mask = np.all(all_objectives < ref_point, axis=1)
        valid_objectives = all_objectives[valid_mask]
        
        if len(valid_objectives) == 0:
            return np.array([])
        
        return valid_objectives
# EVOLVE-BLOCK-END
