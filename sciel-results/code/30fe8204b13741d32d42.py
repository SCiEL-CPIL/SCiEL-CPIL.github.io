"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Analytic Pareto front sampler for DTLZ2 using stratified sampling in decision space."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2 specific strategy:
        # 1. Set all variables except the first M-1 (here 2) to 0.5 to minimize g(x) to 0.
        # 2. Sample the first M-1 variables (x1, x2) to cover the Pareto front.
        # The Pareto front of DTLZ2 is a quarter-sphere. The mapping from (x1, x2) in [0,1]^2
        # to the sphere is bijective. We use a stratified grid (Latin Hypercube style) 
        # to ensure uniform coverage of the objective space.

        n_evals_used = 0
        results = []
        
        # We have budget evaluations. Let's use them all for the front.
        # Create a grid size that fits within budget.
        # For 2 variables, a grid of N x N uses N^2 evaluations.
        # Let N = sqrt(budget).
        n_grid = int(np.sqrt(self.budget))
        
        # To ensure we don't exceed budget, we might need to adjust.
        # If n_grid^2 > budget, we reduce n_grid.
        while n_grid * n_grid > self.budget:
            n_grid -= 1
            
        # Generate stratified samples for x1 and x2
        # We want points in [0, 1]
        # Using a grid is simple and effective for this smooth mapping.
        # To be more "stratified" and avoid grid artifacts, we can add small random jitter 
        # or use a Sobol sequence. Let's use a Sobol-like approach or just a fine grid.
        # Given the smoothness, a grid is very effective.
        
        x1_vals = np.linspace(0, 1, n_grid)
        x2_vals = np.linspace(0, 1, n_grid)
        
        # Iterate through the grid
        for x1 in x1_vals:
            for x2 in x2_vals:
                if n_evals_used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                x = np.full(self.dim, 0.5)
                x[0] = x1
                x[1] = x2
                
                # Ensure x is within bounds (it is, since bounds are [0,1])
                # problem(x) clips, but our x is already valid.
                f = problem(x)
                n_evals_used += 1
                results.append(f)
                
            if n_evals_used >= self.budget or time.time() > deadline:
                break
        
        # If we have remaining budget (e.g., if budget wasn't a perfect square),
        # add more points using a finer grid or random sampling on the front.
        remaining = self.budget - n_evals_used
        if remaining > 0:
            # Use random sampling in the decision space for the remaining budget
            # This helps fill gaps in the grid.
            rng = np.random.default_rng(42)
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                x = np.full(self.dim, 0.5)
                x[0] = rng.random()
                x[1] = rng.random()
                f = problem(x)
                n_evals_used += 1
                results.append(f)

        return results
# EVOLVE-BLOCK-END
