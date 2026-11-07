"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


class Optimizer:
    """Hybrid grid-init + CMA-ES-style local refinement for Rastrigin rotated."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, 10s safety margin
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        calls = 0

        # Generate candidate grid points: integer coordinates in [-4, 4]
        # For Rastrigin, global minima are at integer points (before rotation/shift)
        # After rotation+shift, the minima are shifted but we can still sample integers
        # and their neighbors to find good starting points
        
        # First, evaluate a sparse set of integer grid points to find good basins
        # Use a subset to save budget: sample integers in [-3, 3] for 5D
        grid_range = range(-3, 4)  # 7^5 = 16807, too many; use subset
        # Actually, let's be smarter: evaluate a few random integer points first
        # Then do local refinement from the best ones
        
        # Strategy: 
        # 1. Evaluate ~50 random integer points in [-4, 4]^5
        # 2. Pick the best 3-5
        # 3. For each, do a local CMA-ES-like refinement with remaining budget
        
        # Phase 1: Sample integer points
        n_grid = 40
        grid_points = []
        for _ in range(n_grid):
            pt = rng.integers(-4, 5, size=dim).astype(float)
            grid_points.append(pt)
        
        # Evaluate grid points
        grid_values = []
        for pt in grid_points:
            if calls >= budget or time.time() >= deadline:
                break
            val = func(pt)
            grid_values.append((val, pt))
            calls += 1
        
        if not grid_values:
            # Fallback: just do random search
            x = rng.uniform(-5, 5, size=dim)
            fx = func(x)
            calls += 1
            sigma = 2.0
            for _ in range(budget - 1):
                if time.time() >= deadline or calls >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.85
            return x, fx
        
        # Sort by value, take top 3
        grid_values.sort(key=lambda t: t[0])
        top_candidates = grid_values[:min(3, len(grid_values))]
        
        # Phase 2: Local refinement from each top candidate
        remaining_budget = budget - calls
        budget_per_candidate = max(10, remaining_budget // len(top_candidates))
        
        best_x = top_candidates[0][1]
        best_fx = top_candidates[0][0]
        
        for val, start_pt in top_candidates:
            if calls >= budget or time.time() >= deadline:
                break
            
            # CMA-ES-like local search
            x = start_pt.copy()
            fx = val
            sigma = 0.5  # Start with small step for local refinement
            n_evals = 0
            max_local = min(budget_per_candidate, budget - calls)
            
            # Simple (1+1)-ES with adaptive sigma
            for _ in range(max_local):
                if time.time() >= deadline or calls >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                n_evals += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    if sigma < 0.01:
                        break  # Converged
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
