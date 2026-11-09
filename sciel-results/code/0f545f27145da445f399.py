import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Multi-start Powell + (1+1)-ES for Rosenbrock valley exploitation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 290.0

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")
            
            x = np.asarray(x, dtype=np.float64).flatten()
            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = float(func(x))
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        rng = np.random.default_rng(42)
        
        # Phase 1: Multi-start Powell with adaptive valley following
        # Powell's method is excellent for Rosenbrock because it builds
        # conjugate directions that follow the valley curvature.
        
        # Generate diverse starting points
        start_points = []
        
        # Grid-based starts in the known minimum region [-4, 4]^5
        # Use a sparse grid to cover the space efficiently
        grid_vals = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
        
        # Select a subset of grid points (can't evaluate all 5^5=3125)
        # Pick points with mixed coordinates for diversity
        num_grid_starts = 10
        for _ in range(num_grid_starts):
            idx = rng.choice(len(grid_vals), self.dim, replace=True)
            x0 = grid_vals[idx]
            start_points.append(x0)
        
        # Add random points in the known region
        num_random_starts = 10
        for _ in range(num_random_starts):
            x0 = rng.uniform(-4, 4, self.dim)
            start_points.append(x0)
        
        # Add a few points near the origin and shifted positions
        start_points.append(np.zeros(self.dim))
        start_points.append(np.ones(self.dim) * 0.5)
        start_points.append(np.ones(self.dim) * -0.5)
        
        # Shuffle for diversity
        rng.shuffle(start_points)
        
        # Run Powell from each start point with limited budget
        # Powell typically needs ~50-100 calls for 5D Rosenbrock
        # We'll give each start ~25 calls, then refine the best
        
        powell_results = []
        calls_per_start = 25
        
        for i, x0 in enumerate(start_points):
            if self.calls_made >= self.budget - 50:  # Keep 50 for final refinement
                break
            if time.time() > self.deadline - 10:
                break
            
            try:
                res = scipy.optimize.minimize(
                    wrapped_func,
                    x0,
                    method='Powell',
                    options={
                        'maxiter': 50,
                        'maxfev': calls_per_start,
                        'xtol': 1e-6,
                        'ftol': 1e-6
                    }
                )
                powell_results.append((res.fun, res.x.copy()))
            except StopIteration:
                break
            except Exception:
                continue
        
        # Phase 2: Refine the best candidates with (1+1)-ES
        # (1+1)-ES is simple and effective for final polishing
        
        # Sort by function value
        powell_results.sort(key=lambda x: x[0])
        
        # Take top 3 distinct candidates
        candidates = []
        for f_val, x_val in powell_results[:5]:
            is_distinct = True
            for cx in candidates:
                if np.linalg.norm(x_val - cx) < 0.05:
                    is_distinct = False
                    break
            if is_distinct:
                candidates.append(x_val)
            if len(candidates) >= 3:
                break
        
        if not candidates and self.best_x is not None:
            candidates.append(self.best_x)
        
        # (1+1)-ES refinement for each candidate
        sigma_initial = 0.5
        sigma_min = 1e-8
        max_es_calls_per_candidate = 15
        
        for x0 in candidates:
            if self.calls_made >= self.budget - 5:
                break
            if time.time() > self.deadline - 3:
                break
            
            x = x0.copy()
            f = self.best_f if np.allclose(x, self.best_x) else wrapped_func(x)
            sigma = sigma_initial
            stagnation = 0
            
            for _ in range(max_es_calls_per_candidate):
                if self.calls_made >= self.budget - 5:
                    break
                if time.time() > self.deadline - 2:
                    break
                
                # Generate mutant
                mutation = rng.normal(0, sigma, self.dim)
                x_mut = np.clip(x + mutation, -5, 5)
                
                try:
                    f_mut = wrapped_func(x_mut)
                except StopIteration:
                    break
                
                if f_mut < f:
                    x = x_mut
                    f = f_mut
                    stagnation = 0
                    # Slightly increase sigma on success (optional)
                else:
                    stagnation += 1
                    # Decrease sigma on failure
                    sigma *= 0.85
                    if stagnation > 5:
                        break
                
                if sigma < sigma_min:
                    break
        
        # Final check
        if self.best_x is None:
            x0 = np.zeros(self.dim)
            self.best_x = x0
            self.best_f = wrapped_func(x0)

        return self.best_x, self.best_f
