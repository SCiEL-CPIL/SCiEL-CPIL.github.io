# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Nelder-Mead Simplex Optimizer with Global Grid Initialization.
    
    Strategy:
    1. Initialize with a coarse grid of starting points to ensure coverage of the search space.
    2. For each starting point, run Nelder-Mead simplex method.
    3. Nelder-Mead is particularly effective for smooth, convex functions like the step ellipsoidal function f7.
    4. This avoids the flat-region stagnation of pure CMA-ES and exploits local convexity for rapid convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        self._call_count = 0
        self._best_f = np.inf
        self._best_x = None

        def wrapped(x):
            self._call_count += 1
            fx = float(func(x))
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = np.array(x, dtype=float)
            return fx

        def nelder_mead(func, x0, max_iter=100, tol=1e-8):
            """
            Implement Nelder-Mead simplex method.
            """
            dim = len(x0)
            # Initialize simplex
            simplex = [np.array(x0, dtype=float)]
            f_vals = [func(x0)]
            
            # Create simplex vertices
            for i in range(dim):
                x_new = x0.copy()
                x_new[i] += 0.1 if x_new[i] < 5 else -0.1
                x_new = np.clip(x_new, -5, 5)
                simplex.append(x_new)
                f_vals.append(func(x_new))
            
            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrinkage
            
            for _ in range(max_iter):
                # Sort simplex by function values
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]
                
                # Check convergence
                if abs(f_vals[0] - f_vals[-1]) < tol:
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                
                if f_vals[0] <= f_r < f_vals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                elif f_r < f_vals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    if f_c < f_vals[-1]:
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # Shrinkage
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_vals[i] = func(simplex[i])
            
            # Return best point
            order = np.argsort(f_vals)
            return simplex[order[0]], f_vals[order[0]]

        # Generate diverse starting points using a grid
        # For dim=5, we can use a 2^5 = 32 point grid or similar
        # Let's use a stratified sampling approach
        n_starts = 10  # Number of starting points
        rng = np.random.default_rng(42)
        
        # Create grid points
        grid_points = []
        if dim <= 3:
            # Dense grid for low dimensions
            step = 10.0 / (2 ** (dim // 2 + 1))
            points = np.linspace(-4.5, 4.5, int(10 / step) + 1)
            for p1 in points:
                for p2 in points:
                    for p3 in points:
                        if dim == 1:
                            grid_points.append([p1])
                        elif dim == 2:
                            grid_points.append([p1, p2])
                        elif dim == 3:
                            grid_points.append([p1, p2, p3])
        else:
            # For higher dimensions, use random points with some structure
            for _ in range(n_starts):
                x = rng.uniform(-4.5, 4.5, dim)
                grid_points.append(x)
        
        # Shuffle grid points
        rng.shuffle(grid_points)
        
        # Run Nelder-Mead from each starting point
        for x_start in grid_points:
            if self._call_count >= budget:
                break
            if time.time() > deadline:
                break
            
            # Limit iterations based on remaining budget
            remaining = budget - self._call_count
            max_iter = min(50, remaining // (dim + 1))
            
            if max_iter < 5:
                break
                
            x_best, f_best = nelder_mead(wrapped, np.array(x_start), max_iter=max_iter)
            
            # Update global best
            if f_best < self._best_f:
                self._best_f = f_best
                self._best_x = x_best
        
        # If we have remaining budget, do a final polish around the best point
        if self._best_x is not None and self._call_count < budget:
            remaining = budget - self._call_count
            if remaining > 10:
                # Run one more Nelder-Mead from the best point with more iterations
                max_iter = min(100, remaining // (dim + 1))
                if max_iter > 10:
                    x_best, f_best = nelder_mead(wrapped, self._best_x, max_iter=max_iter)
                    if f_best < self._best_f:
                        self._best_f = f_best
                        self._best_x = x_best

        if self._best_x is None:
            self._best_x = np.zeros(dim)
            self._best_f = func(self._best_x)

        return self._best_x, self._best_f

# EVOLVE-BLOCK-END
