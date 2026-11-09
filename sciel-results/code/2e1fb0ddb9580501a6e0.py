"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead optimizer on the 2D Pareto-front parameter space of DTLZ4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.t0 = time.time()
        # Safety margin: 300s limit, keep 20s buffer
        self.deadline = self.t0 + 280

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 with M=3 has a Pareto front that is a quarter-sphere in 3D.
        # We parameterize the front using 2 angles: theta, phi in [0, pi/2].
        # f1 = sin(theta) * sin(phi)
        # f2 = sin(theta) * cos(phi)
        # f3 = cos(theta)
        # To map back to decision variables x (dim=12):
        # x[0] = f1
        # x[1] = f2
        # x[2:] = 0.5 (to make g=0, assuming standard DTLZ4 g function)
        
        # We will use a multi-start strategy.
        # Total budget: 200 evaluations.
        # Let's do 100 starts, with 2 evaluations per start (1 initial + 1 improvement via NM? 
        # Actually NM takes many steps. Let's be smarter.
        # NM in 2D is very fast. We can afford ~5-10 starts with full NM optimization, 
        # or many starts with limited steps.
        
        # Strategy:
        # 1. Generate a coarse grid of starting points (e.g., 10x10 = 100 points).
        # 2. For each starting point, run Nelder-Mead with a small max_iter (e.g., 10-20).
        # 3. This uses ~100 * (1 + 10) = 1100 evaluations? No, budget is 200.
        # So we can only afford ~20-50 evaluations per start if we do 4-10 starts, 
        # or ~2 evaluations per start if we do 100 starts.
        
        # Better strategy:
        # Use a small number of starts (e.g., 10) and let NM run for ~20 iterations each.
        # 10 starts * 20 evals = 200 evals.
        
        n_starts = 10
        max_iter_per_start = 10  # NM in 2D converges quickly
        
        # Generate starting points: Halton sequence or grid
        # Let's use a stratified random approach for starts
        starts = []
        for i in range(n_starts):
            # Spread starts across the [0, pi/2]^2 domain
            theta0 = (i + 0.5) / n_starts * (np.pi / 2)
            # Vary phi
            phi0 = (np.random.rand() * np.pi) / 2
            starts.append([theta0, phi0])
            
        # To ensure coverage, let's use a grid of starts instead
        grid_size = int(np.sqrt(n_starts))
        if grid_size * grid_size < n_starts:
            grid_size += 1
            
        starts = []
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= n_starts:
                    break
                theta0 = (i + 0.5) / grid_size * (np.pi / 2)
                phi0 = (j + 0.5) / grid_size * (np.pi / 2)
                starts.append([theta0, phi0])
                count += 1
            if count >= n_starts:
                break

        evaluated_points = []
        
        for start in starts:
            if time.time() > self.deadline:
                break
                
            def objective(params):
                theta, phi = params
                # Clip to valid range
                theta = np.clip(theta, 0, np.pi/2)
                phi = np.clip(phi, 0, np.pi/2)
                
                f1 = np.sin(theta) * np.sin(phi)
                f2 = np.sin(theta) * np.cos(phi)
                f3 = np.cos(theta)
                
                x = np.zeros(self.dim)
                x[0] = f1
                x[1] = f2
                for k in range(2, self.dim):
                    x[k] = 0.5
                
                # Clip to bounds
                x = np.clip(x, lo, hi)
                
                # Evaluate
                f = problem(x)
                evaluated_points.append(x.copy())
                
                # Scalarize for Nelder-Mead: minimize the sum of squares or similar
                # Since we are on the front, the sum of squares is constant (1).
                # We need to check if we are actually on the front.
                # If g > 0, the point is dominated.
                # So we want to minimize the "distance" to the front or the g value.
                # But we don't have g directly. We have f1, f2, f3.
                # For DTLZ4, if we are on the front, f1^2 + f2^2 + f3^2 = 1.
                # If we are off the front, the sum will be > 1 (since f_i >= 0).
                # So we can minimize (f1^2 + f2^2 + f3^2 - 1)^2 to find the front.
                
                dist_sq = f1**2 + f2**2 + f3**2 - 1.0
                return dist_sq**2

            try:
                res = minimize(objective, start, method='Nelder-Mead', 
                               options={'maxiter': max_iter_per_start, 'xatol': 1e-4, 'fatol': 1e-6})
            except Exception:
                pass

        return evaluated_points
# EVOLVE-BLOCK-END
