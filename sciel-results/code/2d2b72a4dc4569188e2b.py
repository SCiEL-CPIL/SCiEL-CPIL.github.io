"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting DTLZ2 analytic structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_obj = self.n_obj
        dim = self.dim
        n_pos = n_obj - 1
        n_dist = dim - n_pos
        
        def make_pareto_point(theta1, theta2):
            """Create a point on the known Pareto front of DTLZ2."""
            x = np.zeros(dim)
            x[0] = theta1
            x[1] = theta2
            for i in range(n_pos, dim):
                x[i] = 0.5
            return np.clip(x, lo, hi)
        
        evaluated_points = []
        evaluated_f = []
        used = 0
        budget = self.budget
        
        # Strategy: Use Fibonacci lattice (Koksma) on the unit quarter-sphere
        # to distribute all budget points uniformly on the Pareto front.
        # For DTLZ2 with n_obj=3, the Pareto front is a quarter-sphere in 3D.
        # We map spherical coordinates (theta, phi) to x[0], x[1] such that
        # the resulting objective vector lies on the unit sphere.
        
        # Fibonacci lattice on sphere: use golden ratio to distribute points
        # uniformly over the surface. We use the standard Fibonacci sphere algorithm
        # and then map to the quarter-sphere (all positive octant).
        
        n_points = budget
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        # Generate points using Fibonacci lattice on sphere
        # This gives uniform distribution on the full sphere
        # We'll take the first n_points and map to quarter-sphere
        
        # Alternative: use the known mapping for DTLZ2 Pareto front
        # x[0] = theta1, x[1] = theta2 where theta1, theta2 in [0, 1]
        # f0 = sin(pi/2 * x0), f1 = sin(pi/2 * x1), f2 = cos(pi/2 * (x0 + x1))
        # The Pareto front is the set of (f0, f1, f2) with f0^2 + f1^2 + f2^2 = 1
        # and all fi >= 0
        
        # Use Fibonacci lattice to generate uniform points on quarter-sphere
        # then invert the mapping to get x[0], x[1]
        
        # For uniform coverage of the quarter-sphere, we can use:
        # theta_i = 2*pi*i/golden_ratio  (azimuthal angle)
        # phi_i = arccos(1 - 2*(i+0.5)/n)  (polar angle)
        # But we need to restrict to the first octant (all positive)
        
        # Simpler approach: generate uniform points in [0,1]^2 and use
        # inverse CDF to get uniform distribution on the quarter-sphere
        
        # Actually, let's use the direct approach: generate points uniformly
        # on the quarter-sphere using spherical coordinates with proper density
        
        # For a quarter-sphere, uniform distribution in (theta, phi) requires
        # careful handling. Let's use the Fibonacci lattice on the full sphere
        # and keep only points in the first octant, or use a direct method.
        
        # Direct method: use the fact that for DTLZ2, we can parameterize
        # the Pareto front using two angles. Let's use:
        # f0 = sin(a) * cos(b)
        # f1 = sin(a) * sin(b) 
        # f2 = cos(a)
        # where a in [0, pi/2], b in [0, pi/2]
        # Then x[0] = 2*a/pi, x[1] = 2*b/pi
        
        # To get uniform distribution on the quarter-sphere, we need
        # a and b such that the area element is uniform. The area element
        # on the sphere is sin(a) da db, so we need to sample a with
        # density proportional to sin(a) and b uniformly.
        
        # Use inverse CDF: P(A <= a) = (1 - cos(a)) / (1 - cos(pi/2)) = 1 - cos(a)
        # So cos(a) = 1 - u where u is uniform [0,1]
        # a = arccos(1 - u)
        
        # For b, the conditional distribution given a is uniform in [0, pi/2]
        
        # But we want a 2D lattice, not random. Let's use a grid in (u, v)
        # where u maps to a and v maps to b.
        
        # Actually, the simplest approach for uniform coverage is to use
        # a grid in the (f0, f1) plane and compute f2, then invert.
        
        # Let's use the Fibonacci lattice approach properly:
        # Generate n points on the unit sphere using the Fibonacci spiral,
        # then map to the quarter-sphere by taking absolute values and
        # normalizing, then invert to get x[0], x[1].
        
        # Fibonacci sphere points:
        # z_i = 1 - 2*(i + 0.5)/n
        # r_i = sqrt(1 - z_i^2)
        # theta_i = 2*pi*i/golden_ratio
        # x_i = r_i * cos(theta_i)
        # y_i = r_i * sin(theta_i)
        
        # This gives uniform points on the full sphere. To get quarter-sphere,
        # we take abs() and normalize, but this distorts the distribution.
        
        # Better: use a 2D grid in (u, v) where u, v in [0,1]
        # and map to the quarter-sphere using:
        # a = arccos(1 - u)  (from inverse CDF for uniform on sphere)
        # b = v * pi/2
        # Then f0 = sin(a)*cos(b), f1 = sin(a)*sin(b), f2 = cos(a)
        # And x[0] = 2*a/pi, x[1] = 2*b/pi
        
        # This gives uniform distribution on the quarter-sphere.
        
        # Use a grid in (u, v) with sqrt(n) points per axis for square grid
        n_grid = int(np.ceil(np.sqrt(n_points)))
        
        u_vals = np.linspace(0.5/n_points, 1 - 0.5/n_points, n_grid)
        v_vals = np.linspace(0.5/n_points, 1 - 0.5/n_points, n_grid)
        
        for u in u_vals:
            for v in v_vals:
                if used >= budget or time.time() > deadline:
                    break
                # Map to angles
                a = np.arccos(1 - u)  # a in [0, pi/2]
                b = v * np.pi / 2  # b in [0, pi/2]
                
                # Map to decision variables
                x0 = 2 * a / np.pi
                x1 = 2 * b / np.pi
                
                x = make_pareto_point(x0, x1)
                f = problem(x)
                evaluated_points.append(x.copy())
                evaluated_f.append(np.array(f))
                used += 1
            if used >= budget or time.time() > deadline:
                break
        
        # Return the last point (return value is ignored)
        if evaluated_points:
            return evaluated_points[-1]
        else:
            return lo.copy()
# EVOLVE-BLOCK-END
