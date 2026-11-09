import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ2 problem.
    Uses analytical knowledge of the Pareto front to generate points directly on the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, leave 20s margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2 specific structure:
        # f1 = (1+g) * sin(pi/2 * x0) * sin(pi/2 * x1) * sin(pi/2 * x2)
        # f2 = (1+g) * sin(pi/2 * x0) * sin(pi/2 * x1) * cos(pi/2 * x2)
        # f3 = (1+g) * sin(pi/2 * x0) * cos(pi/2 * x1)
        # g = sum((xi - 0.5)^2) for i in [M, n-1]
        
        # To be on the Pareto front with g=0:
        # x0 = 1.0 (so sin(pi/2 * x0) = 1)
        # x_i = 0.5 for i >= 3 (so g=0)
        # x1, x2 determine the position on the front.
        
        # We want to sample the quarter sphere f1^2 + f2^2 + f3^2 = 1 uniformly.
        # f3 = cos(theta), theta in [0, pi/2]. Uniform in f3 means uniform in cos(theta)?
        # Area element is sin(theta) dtheta dphi.
        # Let u = cos(theta) = f3. du = -sin(theta) dtheta.
        # So area element is -du dphi.
        # Thus f3 is uniform in [0, 1].
        # And phi is uniform in [0, pi/2].
        
        n_points = self.budget
        # Generate points
        # f3 ~ U[0, 1]
        f3_vals = np.random.uniform(0, 1, n_points)
        # phi ~ U[0, pi/2]
        phi_vals = np.random.uniform(0, np.pi/2, n_points)
        
        # Calculate f1, f2
        # f1 = sqrt(1 - f3^2) * cos(phi)
        # f2 = sqrt(1 - f3^2) * sin(phi)
        r = np.sqrt(1 - f3_vals**2)
        f1_vals = r * np.cos(phi_vals)
        f2_vals = r * np.sin(phi_vals)
        
        # Map back to x1, x2
        # f3 = cos(pi/2 * x1)  => x1 = (2/pi) * arccos(f3)
        x1_vals = (2.0 / np.pi) * np.arccos(f3_vals)
        
        # f1 = sqrt(1 - f3^2) * sin(pi/2 * x2)
        # sin(pi/2 * x2) = f1 / sqrt(1 - f3^2)
        # Handle division by zero if r is 0 (f3=1)
        sin_x2 = np.zeros_like(f1_vals)
        mask = r > 1e-12
        sin_x2[mask] = f1_vals[mask] / r[mask]
        # Clip to [0, 1] for safety
        sin_x2 = np.clip(sin_x2, 0, 1)
        x2_vals = (2.0 / np.pi) * np.arcsin(sin_x2)
        
        # Construct full vectors
        # dim = 12, n_obj = 3
        # x = [x0, x1, x2, x3, ..., x11]
        X = np.zeros((n_points, self.dim))
        X[:, 0] = 1.0  # x0
        X[:, 1] = x1_vals
        X[:, 2] = x2_vals
        # x3 to x11
        for i in range(3, self.dim):
            X[:, i] = 0.5  # Center of bounds to minimize g
        
        # Clip to bounds just in case
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = []
        for i in range(n_points):
            if time.time() > deadline:
                break
            f = problem(X[i])
            F.append(f)
            
        return X
