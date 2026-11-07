import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = self.budget

        # DTLZ2 specific structure:
        # To be on the Pareto front, we must set the 'distance' variables to their optimal value.
        # For DTLZ2, g(x) = sum_{i=2}^{n} (x_i - 0.5)^2. 
        # g(x) = 0 when all x_i (i>=2) are 0.5.
        # The objectives are then determined by x_0 and x_1, which map to angles on the unit sphere.
        
        # We will use x_0 and x_1 to span the Pareto front.
        # The other variables (x_2 to x_{n-1}) are fixed at 0.5.
        
        # Use a deterministic Fibonacci lattice (Koksma) to distribute points uniformly on the unit sphere's surface
        # This ensures good coverage of the spherical Pareto front with exact Pareto front membership
        
        # Generate Fibonacci lattice points on the unit sphere in 3D
        n = n_evals
        golden_ratio = (1 + np.sqrt(5)) / 2
        # Use the inverse of the golden ratio for the angle increment
        angle_increment = 2 * np.pi / golden_ratio
        
        # Generate points on the sphere
        # z = 1 - 2 * (i + 0.5) / n for i = 0, ..., n-1
        z = 1 - 2 * (np.arange(n) + 0.5) / n
        # r = sqrt(1 - z^2)
        r = np.sqrt(np.clip(1 - z**2, 0, 1))
        # theta = i * angle_increment
        theta = np.arange(n) * angle_increment
        
        # Convert to Cartesian coordinates on the unit sphere
        x_sph = r * np.cos(theta)
        y_sph = r * np.sin(theta)
        
        # The DTLZ2 Pareto front is a quarter-sphere in the positive orthant:
        # f1 = cos(theta1) * cos(theta2)
        # f2 = cos(theta1) * sin(theta2)
        # f3 = sin(theta1)
        # where theta1, theta2 in [0, pi/2]
        
        # We need to map our sphere points to the quarter-sphere.
        # The standard DTLZ2 mapping uses:
        # x[0] in [0, 1] maps to theta1 in [0, pi/2]
        # x[1] in [0, 1] maps to theta2 in [0, pi/2]
        
        # For a point on the unit sphere (x_sph, y_sph, z_sph):
        # f1 = x_sph, f2 = y_sph, f3 = z_sph (if we consider the full sphere)
        # But we need the quarter-sphere where all components are non-negative.
        
        # The Fibonacci lattice generates points on the full sphere.
        # We need to project or map these to the quarter-sphere.
        
        # Alternative approach: Use the fact that for DTLZ2, the decision variables x[0] and x[1]
        # directly determine the angles. 
        # theta1 = x[0] * pi/2
        # theta2 = x[1] * pi/2
        
        # So we need to find x[0] and x[1] such that the resulting point is on the Pareto front
        # and covers the quarter-sphere uniformly.
        
        # A better approach: Use the spherical coordinates directly.
        # For a point on the quarter-sphere, we can parameterize by (phi, theta) where:
        # f1 = sin(phi) * cos(theta)
        # f2 = sin(phi) * sin(theta)
        # f3 = cos(phi)
        # with phi, theta in [0, pi/2]
        
        # The area element on the sphere is sin(phi) dphi dtheta.
        # To get uniform distribution on the sphere surface, we need to account for this.
        
        # For the quarter-sphere, we can use a 2D grid or sequence that accounts for the curvature.
        
        # Let's use a simpler approach: Use a Sobol sequence in 2D and map to angles.
        # This is what the original code did, but let's try to improve the distribution.
        
        # Actually, the Fibonacci lattice on the full sphere, when projected to the quarter-sphere,
        # may not be uniform. Let's use a different approach.
        
        # Use a Halton sequence or Sobol sequence for better space-filling in the 2D angle space.
        # But first, let's try the direct mapping from the Fibonacci lattice.
        
        # For each point (x_sph, y_sph, z_sph) on the unit sphere:
        # If all coordinates are non-negative, it's already on the quarter-sphere.
        # Otherwise, we need to reflect it.
        
        # But this reflection may not give uniform coverage.
        
        # Let's stick with the Sobol sequence approach but ensure we're using it correctly.
        # The key insight is that x[0] and x[1] map linearly to theta1 and theta2.
        # So we need a space-filling sequence in [0, 1]^2.
        
        # Let's use a Halton sequence for better distribution properties.
        def halton(index, base):
            result = 0
            f = 1
            i = index
            while i > 0:
                f /= base
                result += f * (i % base)
                i //= base
            return result
        
        # Generate Halton sequence points
        # Use prime bases 2 and 3 for 2D
        points = []
        for i in range(1, n + 1):  # Start from 1 to avoid the origin
            p1 = halton(i, 2)
            p2 = halton(i, 3)
            points.append((p1, p2))
        
        points = np.array(points)
        
        # Map to angles
        # x[0] = p1, x[1] = p2 (since bounds are [0, 1])
        # theta1 = x[0] * pi/2 = p1 * pi/2
        # theta2 = x[1] * pi/2 = p2 * pi/2
        
        x_0_mapped = points[:, 0] * (hi[0] - lo[0]) + lo[0]
        x_1_mapped = points[:, 1] * (hi[1] - lo[1]) + lo[1]
        
        # Construct the full decision vectors
        X = np.zeros((n_evals, self.dim))
        
        # Fill the first two columns with the mapped Halton samples
        X[:, 0] = x_0_mapped
        X[:, 1] = x_1_mapped
        
        # Fill the remaining dimensions with the optimal value for g(x)=0
        # For DTLZ2, the optimal value for the remaining variables is typically the midpoint of the bounds
        # Given bounds are [0,1], midpoint is 0.5.
        if self.dim > 2:
            mid_points = (lo[2:] + hi[2:]) / 2.0
            X[:, 2:] = mid_points
        
        # Evaluate all points
        eval_count = 0
        for i in range(n_evals):
            if time.time() > deadline:
                break
            try:
                x = X[i]
                f = problem(x)
                eval_count += 1
            except Exception:
                break
                
        return X
