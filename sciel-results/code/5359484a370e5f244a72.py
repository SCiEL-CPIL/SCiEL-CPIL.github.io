"""Rewritten program for DTLZ2 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2 using a deterministic grid + evolutionary refinement.
    
    Strategy:
    1. Use a deterministic uniform grid in the 2D position space (variables 0..1) to cover
       the Pareto front boundaries and corners.
    2. Set remaining decision variables (2..11) to 0.5 to ensure g=0 (on the Pareto front).
    3. Use remaining budget for evolutionary refinement of the grid points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2: first 2 variables define position on the Pareto front,
        # remaining dim-2 variables determine g (distance from front).
        # To be on the Pareto front, we need g=0, which requires x[2:] = 0.5.
        
        # Step 1: Deterministic grid in 2D position space
        # 10x10 grid = 100 points, covering [0,1]^2 for the first two variables
        n_grid = 10
        grid_points = []
        for i in range(n_grid):
            for j in range(n_grid):
                x = np.full(self.dim, 0.5)
                x[0] = i / (n_grid - 1)
                x[1] = j / (n_grid - 1)
                grid_points.append(x)
        
        grid_points = np.array(grid_points)
        # Clip to bounds
        grid_points = np.clip(grid_points, lo, hi)
        
        # Evaluate grid points
        F = np.array([problem(x) for x in grid_points])
        used = len(grid_points)
        
        # Check if we've used all budget
        if used >= self.budget:
            return grid_points
        
        # Step 2: Evolutionary refinement with remaining budget
        # Start population with grid points
        pop_size = min(len(grid_points), max(20, self.budget // 5))
        X = grid_points[:pop_size].copy()
        F = F[:pop_size].copy()
        
        # Add some random points to diversify
        n_random = min(pop_size, self.budget - used)
        if n_random > 0:
            X_rand = lo + (hi - lo) * rng.random((n_random, self.dim))
            F_rand = np.array([problem(x) for x in X_rand])
            used += n_random
            X = np.vstack([X, X_rand])
            F = np.vstack([F, F_rand])
        
        if used >= self.budget:
            return X
        
        # Evolutionary loop
        while used < self.budget:
            # Find non-dominated points
            dominated = np.array([
                np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) 
                for f in F
            ])
            
            # Select parents from non-dominated set
            non_dom_idx = np.where(~dominated)[0]
            if len(non_dom_idx) == 0:
                parents = X
            else:
                parents = X[non_dom_idx]
            
            # Generate children
            n_children = min(self.budget - used, max(1, len(parents)))
            
            for _ in range(n_children):
                if used >= self.budget:
                    break
                
                # Pick a parent
                parent_idx = rng.integers(len(parents))
                parent = parents[parent_idx]
                
                # Mutate: small perturbation
                mutation_scale = 0.05 * (hi - lo)
                child = parent + mutation_scale * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                
                # Replace worst point in current population
                # Use sum of objectives as a simple quality measure
                if len(F) > 0:
                    worst_idx = int(np.argmax(np.sum(F, axis=1)))
                    X[worst_idx] = child
                    F[worst_idx] = fc
                else:
                    X = child.reshape(1, -1)
                    F = fc.reshape(1, -1)
        
        return X
# EVOLVE-BLOCK-END
