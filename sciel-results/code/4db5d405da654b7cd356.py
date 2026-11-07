"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Grid-based coarse-to-fine deterministic scan for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        low, high = -5.0, 5.0
        side = high - low  # 10.0

        # We will perform a series of grid scans.
        # Stage 1: Coarse grid
        # Stage 2: Medium grid around best region
        # Stage 3: Fine grid
        # Stage 4: Very fine grid
        # Stage 5: Local refinement (1+1-ES style or Nelder-Mead)

        # Total budget is 500.
        # Let's allocate:
        # Coarse: 5^dim points? No, 5^5 = 3125 > 500.
        # So we need to be smarter.
        
        # Rastrigin is highly multimodal. Global min is at 0 (shifted).
        # The function is f(x) = 10*d + sum(x_i^2 - 10*cos(2*pi*x_i)).
        # The global minimum is at the shifted zero.
        
        # A simple grid won't find the global min if the grid is too coarse
        # relative to the period (1.0) and the shift.
        # However, the task says "grid-based coarse-to-fine deterministic scan".
        
        # Let's try a hierarchical grid approach.
        # We can't afford a full 5^5 grid.
        # Maybe we can do a 3^5 grid? 3^5 = 243 points.
        # Then take the best region and do a finer grid there.
        
        # Strategy:
        # 1. Evaluate a 3x3x3x3x3 grid over [-5, 5]^5. (243 calls)
        #    Centers: -5, 0, 5? Or -5, 0, 5 is too coarse.
        #    Step size 5.0. Centers: -5, 0, 5.
        #    This might miss the minimum if it's shifted.
        
        # Better: 4^5 = 1024 points. Too many.
        # 3^5 = 243 points. Step size 5.0.
        # Let's use 3 points per dimension: -5, 0, 5.
        # If the minimum is near 0, we'll find it.
        # But the shift is in [-4, 4]. So the minimum could be anywhere.
        
        # What if we use a random grid? The directive says "deterministic".
        # But "deterministic" might just mean no RNG? Or fixed seed?
        # The seed program uses rng = np.random.default_rng(0).
        # Let's stick to a structured grid.
        
        # Alternative:
        # Use a 2x2x2x2x2 grid (8 points) at corners and center?
        # No, let's try to be more systematic.
        
        # Since Rastrigin has period 1, and we have 500 budget,
        # we can evaluate 100 points per dimension? No, 100^5 is huge.
        
        # Let's look at the structure of Rastrigin.
        # f(x) = 10*5 + sum(x_i^2 - 10*cos(2*pi*x_i))
        # The term x_i^2 pushes towards 0.
        # The cos term has minima at integers.
        # So the global minimum is near the integer closest to the shift.
        
        # If we scan a grid with step 1.0, we will hit the integers.
        # Grid points: -5, -4, ..., 5. That's 11 points per dimension.
        # 11^5 = 161051 points. Way too many.
        
        # We need to reduce the dimensionality or use a smarter search.
        
        # How about a coordinate descent on a grid?
        # Or a local search from multiple grid points?
        
        # Let's try:
        # 1. Sample 50 points uniformly at random (deterministic seed).
        # 2. For each point, do a few steps of local refinement (e.g., Nelder-Mead or simple gradient-free).
        # 3. Keep the best.
        
        # But the directive says "grid-based coarse-to-fine deterministic scan".
        # Maybe it means:
        # - Divide the space into large cells.
        # - Evaluate the center of each cell.
        # - Select the best cell.
        # - Subdivide that cell.
        # - Repeat.
        
        # Let's implement this.
        
        # Stage 1: 2^5 = 32 cells. Centers of each cell.
        # Cell size 5.0.
        # Centers: -2.5, 2.5 for each dimension.
        # 32 evaluations.
        
        # Stage 2: Take the best cell, subdivide into 2^5 = 32 sub-cells.
        # Cell size 2.5.
        # 32 evaluations.
        
        # Stage 3: Take the best sub-cell, subdivide into 2^5 = 32 sub-sub-cells.
        # Cell size 1.25.
        # 32 evaluations.
        
        # Stage 4: Take the best, subdivide into 2^5 = 32.
        # Cell size 0.625.
        # 32 evaluations.
        
        # Total so far: 32 * 4 = 128 evaluations.
        # We have 500 - 128 = 372 left.
        
        # Stage 5: Continue subdividing?
        # 32 * 4 more = 128. Total 256.
        # 32 * 4 more = 128. Total 384.
        # 32 * 3 = 96. Total 480.
        
        # This approach is very systematic.
        # It will converge to the global minimum if the initial grid captures it.
        # With 2^5 cells of size 5, the centers are at +/- 2.5.
        # If the minimum is at 0, it's in the center cell? No, the center cell is [-2.5, 2.5].
        # The center of that cell is 0. So it will be evaluated in Stage 1.
        # Then it will be refined.
        
        # What if the minimum is at 4.5?
        # It's in the cell [2.5, 5]. Center is 3.75.
        # Stage 1 evaluates 3.75.
        # Stage 2 subdivides [2.5, 5] into [2.5, 3.75] and [3.75, 5].
        # Centers: 3.125, 4.375.
        # 4.375 is close to 4.5.
        # Stage 3 subdivides [3.75, 5] into [3.75, 4.375] and [4.375, 5].
        # Centers: 4.0625, 4.6875.
        # 4.6875 is close to 4.5.
        # It will converge.
        
        # This seems promising.
        
        def grid_scan(center, size, func, dim, rng=None):
            """
            Evaluate the centers of 2^dim sub-cells of a hypercube.
            Returns the best point and its value.
            """
            # 2^dim sub-cells
            # Centers are at center +/- size/2 in each dimension?
            # No, if we subdivide a cell of size `size` into 2 per dimension,
            # the new cells have size size/2.
            # The centers of the 2^dim sub-cells are:
            # For each dimension, the center is at center - size/4 or center + size/4?
            # Let's say the cell is [c - s/2, c + s/2].
            # Sub-cells: [c - s/2, c], [c, c + s/2].
            # Centers: c - s/4, c + s/4.
            
            best_x = None
            best_f = float('inf')
            
            # Generate all 2^dim combinations of -1 and +1
            # offsets = size / 4
            offset = size / 4.0
            for mask in range(2 ** dim):
                x = np.array([center[i] + (offset if (mask >> i) & 1 else -offset) for i in range(dim)])
                x = np.clip(x, -5.0, 5.0)
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            return best_x, best_f, size / 2.0

        # Initial cell: center 0, size 10
        center = np.zeros(dim)
        size = 10.0
        best_x, best_f, size = grid_scan(center, size, func, dim)
        center = best_x
        calls = 2 ** dim
        
        # Continue subdividing
        while calls + 2 ** dim <= budget:
            best_x, best_f, size = grid_scan(center, size, func, dim)
            center = best_x
            calls += 2 ** dim
        
        # Now we have a very small cell around the best point.
        # Use the remaining budget for a local search (e.g., 1+1-ES or Nelder-Mead).
        # Let's use a simple 1+1-ES with adaptive sigma.
        
        remaining = budget - calls
        if remaining > 0:
            x = best_x.copy()
            fx = best_f
            sigma = size / 2.0  # Start with the current cell size
            for _ in range(remaining):
                y = x + sigma * np.random.default_rng(0).standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9
            best_x, best_f = x, fx

        return best_x, best_f
# EVOLVE-BLOCK-END
