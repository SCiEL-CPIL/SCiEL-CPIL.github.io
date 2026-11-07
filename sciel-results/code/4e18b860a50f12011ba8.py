import numpy as np
import scipy.optimize as so

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        def eval_point(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget:
                return float('inf')
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse Grid Exploration
        # Use a grid with 5 points per dimension = 5^5 = 3125 points (too many)
        # Use 3 points per dimension = 3^5 = 243 points (fits in ~200 budget)
        # Actually, let's use adaptive: start with center, then expand
        
        # Strategy: 
        # 1. Evaluate center (1 call)
        # 2. Evaluate corners of [-4,4]^5 (2^5 = 32 calls) - likely where optimum is
        # 3. Evaluate midpoints of edges between center and corners (5*2^5 = 160 calls)
        # Total so far: 1 + 32 + 160 = 193 calls
        
        # But we need to be smart. Let's do a stratified sampling.
        # Use a 3-level grid: -4, 0, 4 for each dimension
        # That's 3^5 = 243 points.
        
        grid_points = []
        # Generate all combinations of {-4, 0, 4}
        for i in range(3 ** dim):
            x = np.zeros(dim)
            for j in range(dim):
                level = (i // (3 ** j)) % 3
                x[j] = -4.0 + level * 4.0  # -4, 0, 4
            grid_points.append(x)
        
        # Shuffle to decorrelate (optional, but good practice)
        np.random.seed(42)
        np.random.shuffle(grid_points)
        
        # Evaluate grid points
        for x in grid_points:
            if calls_made >= budget:
                break
            eval_point(x)
        
        # Phase 2: Local refinement from the best point
        remaining = budget - calls_made
        if remaining <= 0 or best_x is None:
            return best_x, best_f
        
        # Use Nelder-Mead from the best grid point
        calls_left = remaining
        
        def wrapped_func(x):
            nonlocal calls_left
            if calls_left <= 0:
                return float('inf')
            calls_left -= 1
            return func(x)
        
        try:
            res = so.minimize(wrapped_func, best_x, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10,
                                      'adaptive': True})
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
            remaining = calls_left
        except Exception:
            pass
        
        # Phase 3: If we have budget left, do a finer local search
        # around the current best using a small grid
        if remaining > 10 and best_x is not None:
            # Create a small local grid around best_x
            # Use a box of size 0.5 around best_x
            local_size = 0.5
            local_grid = []
            n_local = 3  # 3 points per dimension
            for i in range(n_local ** dim):
                x = np.zeros(dim)
                for j in range(dim):
                    level = (i // (n_local ** j)) % n_local
                    # Map level 0,1,2 to -local_size, 0, +local_size
                    x[j] = best_x[j] + (level - 1) * local_size
                local_grid.append(x)
            
            # Shuffle
            np.random.shuffle(local_grid)
            
            for x in local_grid:
                if calls_made >= budget:
                    break
                fx = eval_point(x)
                if fx < best_f:
                    # If we found something better, run NM again
                    if budget - calls_made > 10:
                        calls_left = budget - calls_made
                        try:
                            res = so.minimize(wrapped_func, x, method='Nelder-Mead',
                                             options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10,
                                                      'adaptive': True})
                            if res.fun < best_f:
                                best_f = res.fun
                                best_x = res.x.copy()
                            calls_made = budget - calls_left
                        except Exception:
                            pass
            
        return best_x, best_f
