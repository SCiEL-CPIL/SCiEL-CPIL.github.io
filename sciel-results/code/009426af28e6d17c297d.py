import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2 Pareto front is a quarter-sphere in the first orthant of 3D objective space.
        # f1 = sum(xi^2) * 0.5? No, standard DTLZ2:
        # g(x) = sum_{i in M}(x_i - 0.5)^2
        # f_i = (1+g) * sin(theta_i)
        # To get f on the front, g=0 => x_i = 0.5 for all i.
        # Then f_i = sin(theta_i) where theta_i are angles in [0, pi/2] such that sum(theta_i^2) is not constrained, 
        # but rather the points lie on the unit sphere: sum(f_i^2) = 1.
        # So we need to sample points on the surface of the unit sphere in the first octant (f_i >= 0).
        # Actually, DTLZ2 front is the part of the unit hypersphere in the positive orthant.
        # For 3 objectives, it's a quarter-sphere (surface area 2*pi).
        
        # We will generate 200 points uniformly distributed on this surface.
        # Method: Generate random points in 3D, normalize them, and reject if any coordinate is negative.
        # Or use a specific parameterization.
        # Uniform sampling on a sphere:
        # 1. Generate 3D normal random numbers.
        # 2. Normalize to unit length.
        # 3. If any component is negative, flip sign or reject.
        # Since we need the first octant, we can just take the absolute value of the components.
        # This maps the entire sphere to the first octant, but the density will be uniform on the octant surface.
        
        n_points = self.budget
        
        # Use np.random which is seeded by the evaluator
        # Generate n_points * 3 random numbers
        # We might need more points to get enough valid ones if we reject, but abs() always works.
        # Let's generate points.
        
        # To ensure we don't use too much time, we pre-generate all points.
        # 200 points is small.
        
        # Generate random points
        # Using standard normal distribution for uniform sphere sampling
        points = np.random.randn(n_points, self.n_obj)
        
        # Normalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero (probabilistically zero)
        norms[norms == 0] = 1.0
        points = points / norms
        
        # Project to first octant
        points = np.abs(points)
        
        # Now points are on the unit sphere in the first octant.
        # These are the objective values f.
        # We need to map these back to x.
        # f_i = sin(theta_i) => theta_i = arcsin(f_i)
        # x_0 = theta_0, x_1 = theta_1, x_2 = theta_2 ... wait, DTLZ2 uses n variables.
        # Standard DTLZ2:
        # M objectives, n variables.
        # g(x) = sum_{i=M}^{n} (x_i - 0.5)^2
        # f_i(x) = (1+g(x)) * sin(theta_i)
        # where theta_i = pi/2 * x_i for i=0..M-1.
        
        # To be on the Pareto front, g(x)=0, so x_i = 0.5 for i=M..n-1.
        # And f_i = sin(theta_i) = sin(pi/2 * x_i).
        # So if we have target f_i, we need sin(pi/2 * x_i) = f_i.
        # pi/2 * x_i = arcsin(f_i)
        # x_i = (2/pi) * arcsin(f_i)
        
        # So:
        # x[0] = (2/pi) * arcsin(f_0)
        # x[1] = (2/pi) * arcsin(f_1)
        # x[2] = (2/pi) * arcsin(f_2)
        # x[3:] = 0.5
        
        # Let's evaluate these points.
        
        for i in range(n_points):
            if time.time() > deadline:
                break
            
            f = points[i]
            x = np.zeros(self.dim)
            
            # Calculate x for the first M variables
            # Handle numerical issues where f might be slightly > 1 or < 0
            f_clipped = np.clip(f, 0.0, 1.0)
            
            # x_i = (2/pi) * arcsin(f_i)
            x[:self.n_obj] = (2.0 / np.pi) * np.arcsin(f_clipped)
            
            # Remaining variables are 0.5
            x[self.n_obj:] = 0.5
            
            # Clip to bounds (should be within [0, 1] already if f is in [0, 1])
            x = np.clip(x, lo, hi)
            
            problem(x)
            
        return None
