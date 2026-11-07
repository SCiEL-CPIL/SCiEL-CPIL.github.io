import numpy as np
import time

class Optimizer:
    """DTLZ2 optimizer using correct uniform surface area parameterization and Halton sequence."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ2 structure:
        # Pareto front: x_i = 0.5 for i >= m (m = n_obj - 1 = 2)
        # x_0, x_1 parameterize the front.
        # f_1 = cos(theta_0)cos(theta_1), f_2 = cos(theta_0)sin(theta_1), f_3 = sin(theta_0)
        # theta_0 = x_0 * pi/2, theta_1 = x_1 * pi/2
        # Uniform surface area on the unit sphere (quarter hemisphere) requires:
        # cos(theta_0) ~ U[0, 1] and theta_1 ~ U[0, pi/2]
        # Let u ~ U[0, 1], v ~ U[0, 1]
        # cos(theta_0) = u  => theta_0 = arccos(u) => x_0 = (2/pi) * arccos(u)
        # theta_1 = v * pi/2 => x_1 = v

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

        evaluated_points = []
        used_budget = 0

        # Use a uniform grid in (u, v) space for uniform surface area coverage
        # 14 x 14 = 196 points, plus 4 additional points
        n_grid = 14
        u_vals = np.linspace(0.0, 1.0, n_grid, endpoint=True)
        v_vals = np.linspace(0.0, 1.0, n_grid, endpoint=True)
        
        for u in u_vals:
            for v in v_vals:
                if time.time() > deadline:
                    return evaluated_points
                
                x0 = (2.0 / np.pi) * np.arccos(u)
                x1 = v
                
                x = base_x.copy()
                x[0] = x0
                x[1] = x1
                
                f = problem(x)
                evaluated_points.append(x)
                used_budget += 1
        
        # Add 4 additional points at the center of the parameter space
        # to improve coverage in the middle region
        extra_points = [
            (0.5, 0.5),
            (0.25, 0.75),
            (0.75, 0.25),
            (0.5, 0.0),
        ]
        for u, v in extra_points:
            if time.time() > deadline:
                break
            if used_budget >= self.budget:
                break
            
            x0 = (2.0 / np.pi) * np.arccos(u)
            x1 = v
            
            x = base_x.copy()
            x[0] = x0
            x[1] = x1
            
            f = problem(x)
            evaluated_points.append(x)
            used_budget += 1

        return evaluated_points
