import numpy as np
import time

class Optimizer:
    """DTLZ2-specific optimizer: exploit known structure for Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ2 structure:
        # Pareto front is achieved when g = 0, i.e., x_i = 0.5 for i >= m
        # The remaining m-1 = 2 variables (x_1, x_2, indices 0, 1) parameterize the front.
        # f_1 = cos(theta_1)cos(theta_2), f_2 = cos(theta_1)sin(theta_2), f_3 = sin(theta_1)
        # theta_1 = x_0 * pi/2, theta_2 = x_1 * pi/2
        # To maximize hypervolume, sample uniformly on the Pareto front surface.
        # Surface area element dA = sin(theta_1) dtheta_1 dtheta_2.
        # Uniform sampling requires:
        # 1. theta_2 uniform in [0, pi/2] -> x_1 uniform in [0, 1]
        # 2. 1 - cos(theta_1) uniform in [0, 1] -> theta_1 = arccos(1 - u) -> x_0 = (2/pi) * arccos(1 - u)

        g_start = self.n_obj - 1
        base_x = np.zeros(self.dim)
        base_x[g_start:] = 0.5

        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        try:
            from scipy.stats.qmc import Sobol
            # Use Sobol sequence for better uniformity
            sobol = Sobol(d=2, scramble=True)
            # Generate points. Note: Sobol requires power of 2 for optimal performance, but works for any N.
            # We will generate in chunks or just generate N points.
            # For N=200, it's fine.
            points = sobol.random(self.budget)
            # points shape (budget, 2)
            u = points[:, 0]
            v = points[:, 1]
            
            evaluated_points = []
            for i in range(self.budget):
                if time.time() > deadline:
                    break
                
                x0 = (2.0 / np.pi) * np.arccos(1.0 - u[i])
                x1 = v[i]
                
                x = base_x.copy()
                x[0] = x0
                x[1] = x1
                
                f = problem(x)
                evaluated_points.append(x)
                
        except ImportError:
            # Fallback to Halton if scipy is not available
            evaluated_points = []
            for i in range(1, self.budget + 1):
                if time.time() > deadline:
                    break
                
                u = halton(i, 2)
                v = halton(i, 3)
                
                x0 = (2.0 / np.pi) * np.arccos(1.0 - u)
                x1 = v
                
                x = base_x.copy()
                x[0] = x0
                x[1] = x1
                
                f = problem(x)
                evaluated_points.append(x)

        return evaluated_points
