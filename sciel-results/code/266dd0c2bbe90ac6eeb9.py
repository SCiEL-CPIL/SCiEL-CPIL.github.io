"""Direct search pattern search with adaptive step size and random restarts for BBOB f18."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Pattern search with adaptive step size and sparse random restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        calls = [0]

        def call_func(x):
            calls[0] += 1
            return func(x)

        # Phase 1: Exploration
        # Sample random points to find a good starting region
        n_explore = min(50, budget // 5)
        
        best_x = np.zeros(dim)
        best_fx = float('inf')
        
        # Evaluate center first
        x0 = np.zeros(dim)
        fx0 = call_func(x0)
        best_x = x0.copy()
        best_fx = fx0

        for _ in range(n_explore - 1):
            if calls[0] >= budget:
                break
            x_try = rng.uniform(-5, 5, dim)
            fx_try = call_func(x_try)
            if fx_try < best_fx:
                best_fx = fx_try
                best_x = x_try.copy()

        # Phase 2: Pattern Search
        x = best_x.copy()
        fx = best_fx
        
        # Directions: Orthogonal + Random
        # Orthogonal handles axis-aligned features, random handles rotation
        directions = [np.eye(dim)[i] for i in range(dim)]
        for _ in range(dim):
            d = rng.normal(size=dim)
            norm = np.linalg.norm(d)
            if norm > 0:
                d /= norm
            directions.append(d)
            
        directions = np.array(directions)
        n_dirs = len(directions)
        
        # Step size management
        sigma = 1.0
        sigma_min = 1e-8
        sigma_max = 10.0
        
        # In each iteration, for each direction, we try a few step sizes
        # and update x to the best point found in that direction.
        # This is a coordinate-descent-like pattern search.
        
        while calls[0] < budget:
            improved_in_cycle = False
            best_step_found = 0.0
            
            for d in directions:
                if calls[0] >= budget:
                    break
                
                # Try a set of step sizes relative to current sigma
                # Factors: 1.0, 2.0, 0.5, -1.0, -2.0, -0.5
                best_x_dir = x
                best_fx_dir = fx
                best_step_factor = 0.0
                
                for factor in [1.0, 2.0, 0.5, -1.0, -2.0, -0.5]:
                    if calls[0] >= budget:
                        break
                    y = np.clip(x + factor * sigma * d, -5, 5)
                    fy = call_func(y)
                    if fy < best_fx_dir:
                        best_fx_dir = fy
                        best_x_dir = y
                        best_step_factor = factor
                
                if best_fx_dir < fx:
                    x = best_x_dir
                    fx = best_fx_dir
                    improved_in_cycle = True
                    best_step_found = abs(best_step_factor) * sigma
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
            
            # Adapt sigma
            if improved_in_cycle:
                # We made progress, increase sigma slightly to explore further
                sigma = min(sigma * 1.2, sigma_max)
            else:
                # No progress, decrease sigma
                sigma *= 0.5
                if sigma < sigma_min:
                    # Random restart
                    if calls[0] < budget:
                        x_try = rng.uniform(-5, 5, dim)
                        fx_try = call_func(x_try)
                        if fx_try < fx:
                            x = x_try
                            fx = fx_try
                            best_x = x.copy()
                            best_fx = fx
                    sigma = 1.0

        return best_x, best_fx
# EVOLVE-BLOCK-END
