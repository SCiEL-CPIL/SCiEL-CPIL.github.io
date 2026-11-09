import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return np.inf
            if calls >= budget:
                return np.inf
            calls += 1
            try:
                f_val = float(func(x))
            except:
                f_val = np.inf
            if np.isfinite(f_val) and f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Coarse global grid - 2^5 = 32 points on [-5,5]^5
        # This gives a rough location of the global minimum basin
        grid_coarse = np.array(np.meshgrid(
            np.linspace(-5, 5, 2),
            np.linspace(-5, 5, 2),
            np.linspace(-5, 5, 2),
            np.linspace(-5, 5, 2),
            np.linspace(-5, 5, 2)
        )).T.reshape(-1, dim)
        
        for x in grid_coarse:
            if calls >= budget or time.time() > deadline:
                break
            evaluate(x)
        
        # Add origin as a common minimum location
        if calls < budget:
            evaluate(np.zeros(dim))
        
        if best_f == np.inf:
            evaluate(np.zeros(dim))
        
        # Phase 2: Multi-resolution refinement
        # For each resolution level, create a grid centered on best_x
        # and progressively narrow the search space
        
        resolutions = [3, 4, 5, 6, 8, 10, 12, 16]
        
        for res in resolutions:
            if calls >= budget or time.time() > deadline:
                break
            
            # Determine search radius based on resolution
            # Start with larger radius for coarse, smaller for fine
            if res == 3:
                radius = 5.0
            elif res == 4:
                radius = 4.0
            elif res == 5:
                radius = 3.0
            elif res == 6:
                radius = 2.0
            elif res == 8:
                radius = 1.5
            elif res == 10:
                radius = 1.0
            elif res == 12:
                radius = 0.7
            else:
                radius = 0.5
            
            # Check if we can afford this resolution
            # Number of points = res^dim, but we'll use a smarter approach
            # For dim=5, res^5 can be large, so use a hybrid approach
            
            if res ** dim > 100:
                # Too many points, use a local refinement instead
                # Do a (1+1)-ES style local search
                local_budget = min(20, budget - calls)
                if local_budget <= 0:
                    break
                
                step_size = radius / res
                for _ in range(local_budget):
                    if calls >= budget or time.time() > deadline:
                        break
                    # Try a few random perturbations
                    for _ in range(3):
                        if calls >= budget or time.time() > deadline:
                            break
                        perturbation = step_size * np.random.randn(dim)
                        candidate = best_x + perturbation
                        candidate = np.clip(candidate, -5.0, 5.0)
                        evaluate(candidate)
                
                # Reduce step size for next iteration
                step_size *= 0.5
            else:
                # Create full grid for this resolution
                grid = np.array(np.meshgrid(
                    *[np.linspace(best_x[d] - radius, best_x[d] + radius, res) for d in range(dim)]
                )).T.reshape(-1, dim)
                
                # Clip to valid range
                grid = np.clip(grid, -5.0, 5.0)
                
                for x in grid:
                    if calls >= budget or time.time() > deadline:
                        break
                    evaluate(x)
        
        # Phase 3: Fine local search around the best point found
        remaining = budget - calls
        if remaining > 0:
            # Adaptive step size local search
            step_sizes = [0.3, 0.15, 0.075, 0.0375, 0.01875, 0.009375]
            
            for ss in step_sizes:
                if calls >= budget or time.time() > deadline:
                    break
                
                # Try coordinate-wise moves
                for coord in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    for direction in [-1, 1]:
                        if calls >= budget or time.time() > deadline:
                            break
                        
                        candidate = best_x.copy()
                        candidate[coord] += direction * ss
                        candidate = np.clip(candidate, -5.0, 5.0)
                        evaluate(candidate)
                
                # Also try diagonal moves
                for _ in range(5):
                    if calls >= budget or time.time() > deadline:
                        break
                    perturbation = ss * np.random.randn(dim)
                    candidate = best_x + perturbation
                    candidate = np.clip(candidate, -5.0, 5.0)
                    evaluate(candidate)
        
        return best_x, best_f
