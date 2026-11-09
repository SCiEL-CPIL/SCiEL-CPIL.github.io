"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive sigma, stall detection, and final polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls
            if calls >= self.budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            return fx

        # Strategy: Multi-restart (1+1)-ES with aggressive restarts and final polish.
        # Büche-Rastrigin is multimodal; fast restarts help escape local minima.
        
        # Phase 1: Coarse grid scan to find promising basins (~400 calls)
        # Use a stratified grid over [-4, 4]^5 (since minimizer is in [-4,4]^5)
        # With 400 calls, we can do about 400 grid points
        # Use a Latin hypercube or stratified sampling for better coverage
        
        # Generate stratified sample points
        n_grid = min(350, self.budget - 150)  # leave 150 for local search
        if n_grid < 10:
            n_grid = self.budget
        
        # Create grid points using stratified sampling
        grid_points = []
        # Use a simple approach: divide each dimension into k bins, sample one point per bin
        # For 5D, with ~350 points, we can't do a full grid, so use random stratified
        for _ in range(n_grid):
            if time.time() > deadline:
                break
            x = rng.uniform(-4, 4, self.dim)
            grid_points.append(x)
        
        # Evaluate all grid points
        for x in grid_points:
            if calls >= self.budget or time.time() > deadline:
                break
            fx = evaluate(x)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Local search with CMA-ES-like behavior from the best point
        if best_x is not None and calls < self.budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5  # Start with moderate sigma
            sigma_min = 0.005
            sigma_max = 3.0
            stall_threshold = 30
            max_local_steps = self.budget - calls - 20  # Reserve some calls for restarts
            
            # Local (1+1)-ES with aggressive adaptation
            stall_count = 0
            for step in range(max_local_steps):
                if calls >= self.budget or time.time() > deadline:
                    break
                
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                if fy is None:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.15, sigma_max)
                    stall_count = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.85, sigma_min)
                    stall_count += 1
                    if stall_count >= stall_threshold:
                        break
            
            # Phase 3: Final polish with very small steps if we have budget left
            if calls < self.budget and time.time() < deadline and best_x is not None:
                x = best_x.copy()
                fx = best_f
                sigma = 0.02  # Very small sigma for fine-tuning
                polish_steps = min(30, self.budget - calls)
                
                for step in range(polish_steps):
                    if calls >= self.budget or time.time() > deadline:
                        break
                    
                    y = x + sigma * rng.standard_normal(self.dim)
                    y = np.clip(y, -5.0, 5.0)
                    fy = evaluate(y)
                    if fy is None:
                        break
                    
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        sigma = max(sigma * 0.9, 0.001)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
