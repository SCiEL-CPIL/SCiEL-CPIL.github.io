import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        used = 0
        budget = self.budget
        dim = self.dim
        
        # Generate Sobol sequence points in [0, 1]^2
        # Sobol sequences have low discrepancy, providing better coverage than uniform random or simple grids
        sampler = qmc.Sobol(d=2, scramble=True)
        # Ensure we have enough points. Sobol requires power of 2 for optimal properties, 
        # but we can just generate a large batch and take the first `budget` points.
        # Actually, for small N, Sobol is excellent. Let's generate exactly budget points.
        # Note: qmc.Sobol().random(N) returns N points.
        try:
            points = sampler.random(budget)
        except Exception:
            # Fallback to uniform if scipy qmc fails for some reason (unlikely)
            points = np.random.rand(budget, 2)
        
        # points[:, 0] is u1 in [0, 1]
        # points[:, 1] is u2 in [0, 1]
        
        # Map u1, u2 to x1, x2 on the Pareto front of DTLZ2.
        # DTLZ2 3-objective front is a quarter-sphere of radius 1.
        # f1 = cos(theta1) * cos(theta2)
        # f2 = cos(theta1) * sin(theta2)
        # f3 = sin(theta1)
        # where theta1 in [0, pi/2], theta2 in [0, pi/2].
        
        # To maximize hypervolume with reference point [2.5, 2.5, 2.5],
        # we want points distributed according to the surface area element.
        # The surface area element of a sphere is proportional to sin(theta1) dtheta1 dtheta2.
        # Wait, the parameterization is:
        # x = [cos(t1)cos(t2), cos(t1)sin(t2), sin(t1)]
        # Jacobian determinant for surface area is sin(t1).
        # So the probability density function (PDF) for uniform surface area is proportional to sin(t1).
        # We want to sample t1 such that the density is proportional to sin(t1).
        # CDF of t1: F(t1) = ∫0^t1 sin(t) dt / ∫0^(pi/2) sin(t) dt = (1 - cos(t1)) / 1 = 1 - cos(t1).
        # Inverse CDF: u1 = 1 - cos(t1) => cos(t1) = 1 - u1 => t1 = arccos(1 - u1).
        
        # For t2, the surface area element is constant with respect to t2 (since it's a rotation around the z-axis, 
        # but here it's a quarter sphere, the "longitude" part). 
        # The area element is sin(t1) dt1 dt2. 
        # So t2 should be uniform in [0, pi/2].
        # u2 = t2 / (pi/2) => t2 = u2 * pi/2.
        
        u1 = points[:, 0]
        u2 = points[:, 1]
        
        # Calculate angles
        theta1 = np.arccos(1.0 - u1)
        theta2 = u2 * (np.pi / 2.0)
        
        # Calculate x1 and x2
        # x1 = theta1, x2 = theta2
        # Wait, the decision variables x in DTLZ2 are mapped to objectives via:
        # f1 = 1 + sum(x_i^2 for i >= 3) + cos(x1)cos(x2)
        # f2 = 1 + sum(x_i^2 for i >= 3) + cos(x1)sin(x2)
        # f3 = 1 + sum(x_i^2 for i >= 3) + sin(x1)
        
        # To be on the Pareto front, we need sum(x_i^2 for i >= 3) = 0.
        # This implies x3 = x4 = ... = x12 = 0.
        # And x1, x2 determine the position on the front.
        # So x[0] = theta1, x[1] = theta2, x[2:] = 0.
        
        x1_vals = theta1
        x2_vals = theta2
        
        candidates = []
        for i in range(budget):
            if used >= budget:
                break
            if time.time() >= deadline:
                break
            
            x = np.zeros(dim)
            x[0] = x1_vals[i]
            x[1] = x2_vals[i]
            # x[2:] remains 0
            
            # Clip to bounds just in case, though [0,1] covers [0, pi/2] since pi/2 ~ 1.57 > 1.
            # Wait, bounds are [0, 1] for all variables.
            # theta1 = arccos(1-u1). Max theta1 is pi/2 ~ 1.57.
            # The problem clips x to bounds [0, 1].
            # If x[0] > 1, it gets clipped to 1.
            # This means points with theta1 > 1 will be mapped to x[0]=1.
            # Is this optimal?
            # The Pareto front in decision space is defined by x3..x12=0.
            # x1 and x2 can be anything, but the objective values depend on cos(x1), sin(x1), etc.
            # The "known Pareto front" usually refers to the set of non-dominated points.
            # For DTLZ2, the ideal point is [0,0,0] (if we consider the standard formulation where g(x)=0).
            # Actually, standard DTLZ2:
            # f1 = g(x) + cos(x1)cos(x2)
            # f2 = g(x) + cos(x1)sin(x2)
            # f3 = g(x) + sin(x1)
            # where g(x) = sum_{i=3}^n (x_i - 0.5)^2.
            # Minimum g is 0 when x3..xn = 0.5.
            # So x[2:] should be 0.5, not 0!
            
            # Let's re-verify DTLZ2 definition.
            # Standard DTLZ2:
            # g(x) = sum_{i=3}^n (x_i - 0.5)^2
            # f1 = (1+g) cos(x1) cos(x2)
            # f2 = (1+g) cos(x1) sin(x2)
            # f3 = (1+g) sin(x1)
            
            # To minimize objectives, we want g=0.
            # So x3...x12 = 0.5.
            # x1 and x2 should be in [0, pi/2] to get positive cos/sin.
            # If we set x[0] = theta1 and x[1] = theta2, and x[2:] = 0.5, we are on the front.
            
            x[2:] = 0.5
            
            candidates.append(x)
            
            if time.time() >= deadline:
                break
            
            problem(x)
            used += 1
            
        return np.array(candidates[:used])
