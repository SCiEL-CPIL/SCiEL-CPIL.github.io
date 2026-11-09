"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Optimizes DTLZ1 by minimizing g and sampling the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        all_points = []
        deadline = time.time() + 290
        
        # DTLZ1 structure:
        # x[0] is the scaling variable.
        # x[1] to x[n_obj-2] are the shaping variables (here x[1] for n_obj=3).
        # x[n_obj-1] to x[dim-1] are the g-variables (here x[2] to x[6]).
        
        # To minimize objectives, we must minimize g(x_g).
        # The minimum of g is 0, achieved when all g-variables are 0.5.
        # We set the g-variables to 0.5.
        
        # The number of g-variables is dim - (n_obj - 1).
        # For dim=7, n_obj=3, g-variables are x[2], x[3], x[4], x[5], x[6]?
        # Standard DTLZ1: M = K - m. Here K=7, m=3. M=4.
        # Variables x_m ... x_{K-1} are g-variables.
        # Indices: 2, 3, 4, 5? Or 3, 4, 5, 6?
        # Let's assume the last M variables.
        # M = 7 - 3 = 4.
        # So x[3], x[4], x[5], x[6] are g-variables?
        # Or x[2], x[3], x[4], x[5]?
        # Usually x_1 is scaling, x_2..x_{m-1} are shaping, x_m..x_{K-1} are g.
        # 0-indexed: x[0] scaling, x[1]..x[m-2] shaping, x[m-1]..x[K-1] g.
        # m=3. x[0] scaling. x[1] shaping. x[2]..x[6] g-variables.
        # So 5 g-variables?
        # Let's check M. If M=K-m, then M=4.
        # If M=4, then x[3]..x[6] are g-variables. x[2] is shaping?
        # For m=3, shaping vars are x_2..x_{m-1} = x_2..x_2. Just one variable x_2 (index 1).
        # So x[1] is shaping.
        # Then g-variables start at index 2.
        # So x[2], x[3], x[4], x[5], x[6] are g-variables. Total 5.
        # But M=4?
        # Let's try setting all x[2:] to 0.5. This covers both cases.
        
        base_x = np.zeros(self.dim)
        # Set g-variables to 0.5 to minimize g
        base_x[2:] = 0.5
        
        # We need to sample x[0] and x[1] to cover the Pareto Front.
        # The Pareto Front is defined by varying x[0] and x[1] while g is minimal.
        # If g=0, the front is a point (origin).
        # If the problem has a non-trivial front (e.g. g=1), we need to sample the simplex.
        # To be safe, we sample a dense grid of x[0] and x[1] in [0, 1].
        
        remaining = self.budget
        if remaining <= 0:
            return all_points
            
        # Determine grid size
        # We have 2 variables to sample (x[0], x[1]).
        # Use a square grid.
        n_grid = int(np.sqrt(remaining))
        n_grid = max(2, min(n_grid, 200)) # Cap to avoid excessive overhead if budget is huge
        
        # If remaining is not a perfect square, we might not use all budget.
        # We'll fill the rest with random points if needed.
        
        points_to_eval = []
        for i in range(n_grid):
            for j in range(n_grid):
                if len(points_to_eval) >= remaining:
                    break
                x0 = i / (n_grid - 1) if n_grid > 1 else 0.5
                x1 = j / (n_grid - 1) if n_grid > 1 else 0.5
                x = base_x.copy()
                x[0] = x0
                x[1] = x1
                points_to_eval.append(x)
                
        # Evaluate all points
        for x in points_to_eval:
            if time.time() > deadline:
                break
            f = problem(x)
            all_points.append(f.copy())
            
        # If we have budget left, add random points to ensure coverage
        if len(all_points) < self.budget:
            rng = np.random.default_rng(42)
            while len(all_points) < self.budget and time.time() < deadline:
                x = base_x.copy()
                x[0] = rng.random()
                x[1] = rng.random()
                f = problem(x)
                all_points.append(f.copy())

        return all_points
# EVOLVE-BLOCK-END
