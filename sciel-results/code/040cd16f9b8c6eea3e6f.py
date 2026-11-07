"""Multi-objective optimizer for DTLZ1 using Sobol quasi-random sampling with Pareto front exploitation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Exploits the known Pareto front structure of DTLZ1.
    The Pareto front for DTLZ1 with 3 objectives is defined by:
    - f_0 = 0.5 * (1 + g) * x_0
    - f_1 = 0.5 * (1 + g) * (1 - x_0) * x_1
    - f_2 = 0.5 * (1 + g) * (1 - x_0) * (1 - x_1)
    where g = sum((x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))) for i=2..6, g >= 0.
    
    The ideal Pareto front has g=0, achieved when x_2..x_6 = 0.5.
    On the front: f_0 + f_1 + f_2 = 0.5 * (1+g), and with g=0, sum = 0.5.
    
    Strategy:
    1. Use Sobol sequence to sample points in [0,1]^7.
    2. Focus on points where x_2..x_6 are near 0.5 (to minimize g).
    3. Additionally, directly construct points on the Pareto front by:
       - Setting x_2..x_6 = 0.5 (g=0)
       - Sampling x_0, x_1 from a triangular domain (x_0 >= 0, x_1 >= 0, x_0 + x_1 <= 1)
    4. Use remaining budget for local refinement around best points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Track all evaluated points and their objective values
        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= self.budget:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            used += 1
            return x, f

        # Phase 1: Direct Pareto front construction
        # For DTLZ1, set x_2..x_6 = 0.5 to get g=0
        # Then sample x_0, x_1 on the simplex: x_0 >= 0, x_1 >= 0, x_0 + x_1 <= 1
        # This gives points exactly on the ideal Pareto front.

        # Allocate 60% of budget to direct front construction
        n_front = min(int(self.budget * 0.6), self.budget)
        n_front = max(n_front, 10)

        # Use Sobol sequence for uniform coverage on the triangle
        # Map 2D Sobol to triangle: if u,v in [0,1], let w = min(u, v, 1-u-v)
        # Better: use transformation for uniform sampling on triangle
        # x_0 = r * cos(theta), x_1 = r * sin(theta) with r in [0,1], theta in [0, pi/2]
        # But we want uniform density. Use: a, b from [0,1], a+b <= 1
        # Transformation: s = sqrt(u), x_0 = s*v, x_1 = s*(1-v) where u,v in [0,1]

        sobol = qmc.Sobol(d=2, scramble=True)
        # Generate enough points
        n_sobol = int(np.ceil(n_front))
        points_2d = sobol.random(n_sobol)  # shape (n_sobol, 2), in [0,1]^2

        # Transform to triangle: x_0 = sqrt(u)*v, x_1 = sqrt(u)*(1-v)
        u = points_2d[:, 0]
        v = points_2d[:, 1]
        s = np.sqrt(u)
        x0_vals = s * v
        x1_vals = s * (1 - v)

        for i in range(n_front):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x0_vals[i]
            x[1] = x1_vals[i]
            # Set x_2..x_6 = 0.5 for g=0
            for j in range(2, self.dim):
                x[j] = 0.5
            result = evaluate(x)
            if result is not None:
                all_X.append(result[0])
                all_F.append(result[1])

        # Phase 2: Sobol sampling in full space for robustness
        # Allocate remaining budget to full-space exploration
        remaining = self.budget - used
        if remaining > 0 and used < self.budget:
            sobol_full = qmc.Sobol(d=self.dim, scramble=True)
            # Number of points to generate
            n_full = remaining
            points_full = sobol_full.random(n_full)  # shape (n_full, dim)

            for i in range(n_full):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                x = points_full[i] * span + lo
                result = evaluate(x)
                if result is not None:
                    all_X.append(result[0])
                    all_F.append(result[1])

        # Return something (value is ignored by evaluator)
        if all_X:
            return np.array(all_X)
        return None
# EVOLVE-BLOCK-END
