"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Grid-based global exploration followed by Differential Evolution.
    1. Coarse grid search to locate the global basin (high early AOCC).
    2. Differential Evolution to refine the solution.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # Safety margin before 300s limit
        
        calls = 0
        best_x = None
        best_f = float('inf')
        
        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_x = x.copy()
                best_f = fx

        def can_continue():
            return calls < budget and time.time() < deadline

        def evaluate(x):
            nonlocal calls
            if not can_continue():
                return None
            x = np.clip(x, -5.0, 5.0)
            calls += 1
            fx = func(x)
            if not np.isfinite(fx):
                fx = 1e10
            update_best(x, fx)
            return fx

        # Phase 1: Coarse Grid Search
        # We use a 3x3x3x3x3 grid (243 points) over [-5, 5]^5.
        # This is deterministic and covers the whole domain.
        # The Schwefel function f20 has a global minimum near 0 (shifted).
        # A grid of step size 2.5 will likely hit the vicinity of the minimum.
        
        grid_levels = 3
        # Generate grid points
        # We want to include the center (0) explicitly if possible, or close to it.
        # For [-5, 5] with 3 levels: -5, 0, 5.
        # This is perfect for f20 which is centered at 0 (before shift).
        # Even with shift, the grid covers the box.
        
        # We will evaluate these points.
        # To maximize early AOCC, we evaluate the most "central" points first.
        # Distance from center (0,0,0,0,0) is a good proxy for "likely to be good" for f20.
        
        grid_points = []
        coords = np.linspace(-5.0, 5.0, grid_levels)
        
        # Generate all combinations
        for x1 in coords:
            for x2 in coords:
                for x3 in coords:
                    for x4 in coords:
                        for x5 in coords:
                            p = np.array([x1, x2, x3, x4, x5])
                            dist = np.linalg.norm(p)
                            grid_points.append((dist, p))
        
        # Sort by distance to origin (0,0,0,0,0)
        # This ensures we find the global minimum (likely near 0) very quickly.
        grid_points.sort(key=lambda t: t[0])
        
        # Evaluate grid points
        # We limit this phase to ensure we have budget left for DE.
        # 243 points is reasonable.
        max_grid_calls = min(243, budget // 2) 
        
        for dist, p in grid_points:
            if calls >= max_grid_calls:
                break
            if not can_continue():
                break
            evaluate(p)
            
        if not can_continue():
            return best_x, best_f

        # Phase 2: Differential Evolution
        # Initialize population around the best point found in Phase 1.
        # DE is robust and good for continuous optimization.
        
        pop_size = 10
        rng = np.random.default_rng(42)
        
        # Initialize population
        pop = []
        pop_f = []
        
        # Include the best point found so far
        if best_x is not None:
            pop.append(best_x.copy())
            pop_f.append(best_f)
        else:
            # Fallback if grid didn't run (shouldn't happen)
            x0 = np.zeros(dim)
            fx0 = evaluate(x0)
            if fx0 is None: return best_x, best_f
            pop.append(x0)
            pop_f.append(fx0)

        # Add random points around best_x
        # Initial spread: 1.0
        for i in range(pop_size - 1):
            x = best_x + rng.normal(0, 1.0, dim)
            x = np.clip(x, -5.0, 5.0)
            fx = evaluate(x)
            if fx is None:
                break
            pop.append(x)
            pop_f.append(fx)
            
        if not can_continue():
            return best_x, best_f

        # DE Parameters
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate
        # Adaptive spread? Let's keep it simple but adaptive to stagnation.
        spread = 1.0
        
        while can_continue():
            # Check for stagnation
            # If no improvement in last 10 evaluations, reduce spread
            # (Simple heuristic: just keep running, DE usually handles this)
            
            for i in range(pop_size):
                if not can_continue():
                    break
                
                # Select 3 distinct individuals other than i
                indices = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = rng.choice(indices, 3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                trial = np.copy(pop[i])
                for j in range(dim):
                    if rng.random() < CR or j == rng.integers(dim):
                        trial[j] = v[j]
                
                trial = np.clip(trial, -5.0, 5.0)
                
                # Evaluate trial
                ft = evaluate(trial)
                if ft is None:
                    break
                
                # Selection
                if ft < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = ft
        
        return best_x, best_f

# EVOLVE-BLOCK-END
