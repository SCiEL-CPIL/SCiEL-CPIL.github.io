"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coarse-to-fine deterministic grid search with local refinement.
    Uses a systematic halving grid to ensure early global coverage and
    avoid local minima traps, followed by a local (1+1)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # Safety margin before 300s limit
        calls = 0

        def can_continue():
            return calls < budget and time.time() < deadline

        def remaining():
            return budget - calls

        def make_func_counter():
            nonlocal calls
            def f(x):
                nonlocal calls
                calls += 1
                return func(x)
            return f

        f_count = make_func_counter()

        best_x = None
        best_f = float('inf')

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_x = x.copy()
                best_f = fx

        rng = np.random.default_rng(0)

        # Phase 1: Coarse-to-Fine Grid Search
        # Start with a coarse grid to find the general region of the minimum.
        # The grid is centered at the origin initially, but we can also
        # try a few different offsets if we have budget.
        
        # We will perform 3 levels of grid refinement.
        # Level 1: 3 points per dim -> 3^5 = 243 points
        # Level 2: 3 points per dim around best -> 3^5 = 243 points
        # Level 3: 3 points per dim around best -> 3^5 = 243 points
        # Total: 729 points. We have 500 budget.
        # So we need to be smarter.
        
        # Let's use a 2-point per dim grid for the first pass? 2^5=32.
        # Or a 3-point per dim grid but only evaluate a subset?
        
        # Strategy:
        # 1. Evaluate a coarse 3x3x3x3x3 grid centered at 0. (243 evals)
        # 2. If budget remains, evaluate a finer 3x3x3x3x3 grid around the best point. (243 evals)
        # 3. Use remaining budget for local refinement.
        
        # However, 243 + 243 = 486. Leaves 14 for refinement.
        # This is a very structured approach.
        
        # Let's try a slightly different allocation to ensure we don't run out.
        # Level 1: 2 points per dim? No, 3 is better for capturing peaks/valleys.
        # Maybe start with a 3-point grid but limit to 100 points?
        # No, systematic is key.
        
        # Let's do:
        # Level 1: 3 points per dim, centered at 0. Range [-4, 4] step 4.
        # Points: -4, 0, 4.
        # 3^5 = 243 points.
        
        # If we have 500 budget:
        # 243 for Level 1.
        # 243 for Level 2 (around best of L1).
        # 14 for Level 3 (local refinement).
        
        # This leaves little room for refinement.
        # Alternative:
        # Level 1: 2 points per dim? 2^5=32. Range [-5,5] -> -5, 5? No, -4, 4?
        # Let's use 3 points per dim but only 2 levels of grid, then more refinement.
        
        # Revised Strategy:
        # Level 1: 3 points per dim, centered at 0. Range [-4, 4]. (243 evals)
        # Level 2: 3 points per dim, centered at best_x1. Range [best-2, best+2]. (243 evals)
        # Level 3: Local (1+1)-ES from best_x2. (Remaining ~14 evals)
        
        # This seems too rigid. What if the minimum is at the edge?
        # The problem says minimizer lies in [-4, 4]^5.
        # So a grid covering [-4, 4] is safe.
        
        # Let's implement this.
        
        # Level 1 Grid
        # 3 points per dimension
        # Center 0, spread 4. Points: -4, 0, 4
        # This covers the required domain [-4, 4].
        
        # Generate grid points
        def generate_grid(center, spread, points_per_dim):
            """Generate grid points around center with given spread."""
            # Create 1D points
            # If points_per_dim is odd, center is the middle point
            # Spread is the distance from center to the outermost point
            if points_per_dim == 3:
                # -spread, 0, +spread
                pts_1d = np.array([-spread, 0.0, spread])
            elif points_per_dim == 2:
                # -spread/2, +spread/2 ? Or -spread, +spread?
                # Let's stick to 3 for simplicity and symmetry.
                pts_1d = np.array([-spread, spread])
            else:
                # Linear space
                pts_1d = np.linspace(-spread, spread, points_per_dim)
            
            # Generate all combinations
            # Use itertools.product equivalent via meshgrid
            grids = np.meshgrid(*([pts_1d] * dim), indexing='ij')
            # Stack and reshape
            grid_points = np.stack([g.ravel() for g in grids], axis=1)
            # Center the grid
            grid_points += center
            # Clip to bounds
            grid_points = np.clip(grid_points, -5, 5)
            return grid_points

        # Phase 1: Coarse Grid
        # 3 points per dim, spread 4.0. Covers [-4, 4].
        # 3^5 = 243 points.
        if can_continue():
            grid1 = generate_grid(np.zeros(dim), 4.0, 3)
            # Shuffle to get early progress? No, systematic is better for AOCC?
            # Actually, random order might hit the minimum earlier if it's not at the center.
            # But systematic ensures coverage.
            # Let's evaluate in a random order to maximize chance of early hit?
            # The prompt says "systematically samples".
            # However, AOCC rewards early progress.
            # If the minimum is at [4,4,4,4,4], a systematic order starting from [-4,-4...] will take 243 steps to find it.
            # A random order might find it in 1 step.
            # Let's shuffle the grid points to randomize the search order while maintaining coverage.
            idx = rng.permutation(len(grid1))
            grid1_shuffled = grid1[idx]
            
            for x in grid1_shuffled:
                if not can_continue():
                    break
                fx = f_count(x)
                update_best(x, fx)
                
        if not can_continue():
            return best_x, best_f

        # Phase 2: Fine Grid around best
        # 3 points per dim, spread 2.0.
        # This zooms in on the best point found in Phase 1.
        # 3^5 = 243 points.
        if can_continue():
            grid2 = generate_grid(best_x, 2.0, 3)
            idx = rng.permutation(len(grid2))
            grid2_shuffled = grid2[idx]
            
            for x in grid2_shuffled:
                if not can_continue():
                    break
                fx = f_count(x)
                update_best(x, fx)

        if not can_continue():
            return best_x, best_f

        # Phase 3: Local Refinement (1+1)-ES
        # Use remaining budget.
        if can_continue() and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            n_refine = remaining()
            
            for i in range(n_refine):
                if not can_continue():
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = f_count(y)
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2
                    update_best(x, fx)
                else:
                    sigma *= 0.9

        return best_x, best_f

# EVOLVE-BLOCK-END
