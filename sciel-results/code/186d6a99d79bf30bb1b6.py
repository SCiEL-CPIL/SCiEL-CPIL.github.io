"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic Multi-Resolution Grid Descent with stochastic escape for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(12345)
        n = self.dim
        budget = self.budget

        # Track best solution found
        best_x = None
        best_fx = np.inf

        # Phase 1: Coarse random sampling to find a promising region
        # Use a subset of budget for initial exploration
        coarse_budget = min(50, budget // 5)
        for i in range(coarse_budget):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # If we didn't find anything reasonable, use center
        if best_x is None:
            best_x = np.zeros(n)
            best_fx = func(best_x)

        # Phase 2: Multi-resolution grid descent
        # Start with a large hypercube around the best point and progressively refine
        # The step ellipsoidal function has plateaus, so we need to be careful
        
        # Adaptive step size for grid search
        # Start with a moderate radius
        radius = 3.0
        min_radius = 0.01
        max_iterations = budget - coarse_budget - 10  # reserve some budget for final refinement
        
        # Grid resolution: we'll evaluate points on a grid within the hypercube
        # For 5D, a full grid of 3^5 = 243 points per level is reasonable
        # But we can be smarter: only evaluate points that are "promising"
        
        iterations = 0
        stagnant_rounds = 0
        
        while iterations < max_iterations and time.time() < deadline:
            # Generate grid points around best_x within radius
            # Use a 3-point grid per dimension (center, -offset, +offset)
            # This gives 3^n points, which for n=5 is 243
            # But we can reduce this by using a subset or adaptive resolution
            
            # For efficiency, use a smaller grid: 2 points per dimension offset from center
            # Plus the center itself: this gives 2^n + 1 points
            # For n=5, that's 33 points per round
            
            grid_points = []
            # Add center
            grid_points.append(best_x.copy())
            
            # Add axis-aligned neighbors at radius/2
            for d in range(n):
                for sign in [-1, 1]:
                    pt = best_x.copy()
                    pt[d] += sign * radius * 0.5
                    pt = np.clip(pt, -5, 5)
                    grid_points.append(pt)
            
            # Add diagonal neighbors (corners of the hypercube)
            # For 5D, all 2^5 = 32 corners might be too many, so sample a subset
            # Actually, let's use a smaller set: just the axis-aligned ones plus center
            # That's 2*5 + 1 = 11 points per round
            
            # Evaluate all grid points
            improved = False
            for pt in grid_points:
                if time.time() > deadline:
                    break
                fx = func(pt)
                if fx < best_fx:
                    best_fx = fx
                    best_x = pt.copy()
                    improved = True
            
            iterations += 1
            
            if improved:
                stagnant_rounds = 0
                # Slightly shrink radius to refine
                radius *= 0.85
            else:
                stagnant_rounds += 1
                if stagnant_rounds >= 3:
                    # Stagnant: try a larger jump to escape plateau
                    # Use a random step in a random direction
                    direction = rng.standard_normal(n)
                    direction /= np.linalg.norm(direction) + 1e-10
                    jump_size = radius * 2.0
                    new_x = best_x + jump_size * direction
                    new_x = np.clip(new_x, -5, 5)
                    new_fx = func(new_x)
                    if new_fx < best_fx:
                        best_fx = new_fx
                        best_x = new_x.copy()
                        stagnant_rounds = 0
                        radius = 2.0  # Reset to moderate radius
                    else:
                        # Try another random point
                        new_x = rng.uniform(-5, 5, n)
                        new_fx = func(new_x)
                        if new_fx < best_fx:
                            best_fx = new_fx
                            best_x = new_x.copy()
                            stagnant_rounds = 0
                            radius = 3.0
                        else:
                            # Shrink radius even more to try finer resolution
                            radius *= 0.7
                            if radius < min_radius:
                                break
            
            # Safety: if radius is too small, break
            if radius < min_radius:
                break
        
        # Phase 3: Final local refinement with small random steps
        remaining_budget = budget - coarse_budget - iterations
        if remaining_budget > 0 and time.time() < deadline:
            sigma = max(radius, 0.005)
            for _ in range(min(remaining_budget, 50)):
                if time.time() > deadline:
                    break
                y = best_x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = func(y)
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.9
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
