"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Lattice search + (1+1)-ES with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        total_budget = self.budget
        dim = self.dim

        best_x = None
        best_f = float('inf')
        calls_made = 0

        # Phase 1: Lattice search with step size 0.5 (coarse grid)
        # This exploits Rastrigin's periodicity of 1.0
        # Grid points: -5, -4.5, -4, ..., 4.5, 5 -> 21 points per dim
        # But we only have 500 calls, so use a coarser grid
        # Use step 1.0: -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5 -> 11 points per dim
        # Total: 11^5 = 161051, too many
        # Use step 2.0: -5, -3, -1, 1, 3, 5 -> 6 points per dim
        # Total: 6^5 = 7776, still too many
        # Use a random subset of the lattice
        lattice_step = 2.0
        lattice_points = []
        for d in range(dim):
            coords = np.arange(-5, 5.1, lattice_step)
            lattice_points.append(coords)
        
        # Create all combinations but sample from them
        from itertools import product
        all_lattice = list(product(*lattice_points))
        
        # Shuffle and take first N
        np.random.shuffle(all_lattice)
        lattice_budget = min(100, total_budget // 5)
        
        for point in all_lattice[:lattice_budget]:
            if calls_made >= total_budget:
                break
            if time.time() > deadline - 5.0:
                break
            
            x = np.array(point, dtype=float)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: (1+1)-ES from best lattice point
        if best_x is not None and calls_made < total_budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            
            while calls_made < total_budget and time.time() < deadline - 2.0:
                y = np.clip(x + sigma * np.random.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

                sigma = np.clip(sigma, 0.01, 10.0)

        # Phase 3: Final local polish with smaller sigma
        if best_x is not None and calls_made < total_budget and time.time() < deadline - 5.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            while calls_made < total_budget and time.time() < deadline - 2.0:
                y = np.clip(x + sigma * np.random.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.8
                sigma = np.clip(sigma, 0.001, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
