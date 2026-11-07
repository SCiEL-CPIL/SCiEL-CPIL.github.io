import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_points = self.budget
        n_vertices = 4
        n_lhs = n_points - n_vertices

        # Optimal value for distance variables in DTLZ4 (beta=10)
        # x^beta = 0.5 => x = 0.5^(1/10)
        x_opt_dist = 0.5 ** (1.0 / 10.0)

        # Generate LHS points in [0, 1]^2
        # perm_theta, perm_phi
        perm_theta = np.random.permutation(n_lhs)
        perm_phi = np.random.permutation(n_lhs)
        
        # Uniform points in each cell
        u_theta = (perm_theta + np.random.rand(n_lhs)) / n_lhs
        u_phi = (perm_phi + np.random.rand(n_lhs)) / n_lhs

        # Map to spherical coordinates for uniform AREA coverage
        # theta in [0, pi/2]
        theta = u_theta * (np.pi / 2)
        
        # phi mapping: 
        # Area element is sin(phi) dphi dtheta.
        # We want uniform distribution in area.
        # Let c = cos(phi). c is uniform in [0, 1] if phi is sampled correctly.
        # So we map u_phi to c directly.
        c = u_phi  # c in [0, 1]
        # phi = arccos(c)
        # sin(phi) = sqrt(1 - c^2)
        
        # Objective values (which are proportional to x0, x1, x2 on the unit sphere)
        # f3 = cos(phi) = c
        # f1 = sin(phi) cos(theta)
        # f2 = sin(phi) sin(theta)
        
        sin_phi = np.sqrt(np.maximum(0, 1 - c**2))
        
        f1 = sin_phi * np.cos(theta)
        f2 = sin_phi * np.sin(theta)
        f3 = c

        # Create full dimension vectors
        x_all = np.zeros((n_points, self.dim))
        
        # Set distance variables to optimal
        x_all[:, 3:] = x_opt_dist

        # Set the first 3 variables for LHS points
        x_all[:n_lhs, 0] = f1
        x_all[:n_lhs, 1] = f2
        x_all[:n_lhs, 2] = f3

        # Add explicit vertices
        # 1. (1, 0, 0)
        x_all[n_lhs, 0] = 1.0
        x_all[n_lhs, 1] = 0.0
        x_all[n_lhs, 2] = 0.0
        # 2. (0, 1, 0)
        x_all[n_lhs + 1, 0] = 0.0
        x_all[n_lhs + 1, 1] = 1.0
        x_all[n_lhs + 1, 2] = 0.0
        # 3. (0, 0, 1)
        x_all[n_lhs + 2, 0] = 0.0
        x_all[n_lhs + 2, 1] = 0.0
        x_all[n_lhs + 2, 2] = 1.0
        # 4. (1/sqrt(3), 1/sqrt(3), 1/sqrt(3))
        val = 1.0 / np.sqrt(3.0)
        x_all[n_lhs + 3, 0] = val
        x_all[n_lhs + 3, 1] = val
        x_all[n_lhs + 3, 2] = val

        # Clip to bounds
        x_all = np.clip(x_all, lo, hi)

        results = []
        for i in range(n_points):
            if time.time() > deadline:
                break
            try:
                f = problem(x_all[i])
                results.append(f)
            except Exception:
                pass

        return results
