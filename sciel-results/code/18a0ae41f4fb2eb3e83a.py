"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Powell's method for Sharp Ridge (f13)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        best_x = None
        best_f = np.inf
        calls = 0

        def time_ok():
            return time.time() < deadline

        # Multi-start initialization to find good basins
        # For Sharp Ridge, multiple starts help escape local flat regions
        n_starts = 10
        starts = []
        # Include origin and random points
        starts.append(np.zeros(self.dim))
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, self.dim))

        # Evaluate all starts first to find the best initial point
        for x0 in starts:
            if not time_ok():
                break
            fx = func(x0)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

        if best_x is None:
            # Fallback: use origin
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls += 1

        # Powell's method from the best start
        x = best_x.copy()
        fx = best_f
        
        # Initialize directions: standard basis + one random direction
        directions = [np.eye(self.dim, self.dim)[:, i] for i in range(self.dim)]
        directions.append(rng.normal(size=self.dim))
        
        # Normalize directions
        for i in range(len(directions)):
            norm = np.linalg.norm(directions[i])
            if norm > 0:
                directions[i] = directions[i] / norm

        max_powell_iter = 50  # Limit iterations to stay within budget
        
        for powell_iter in range(max_powell_iter):
            if not time_ok():
                break
            if calls >= self.budget - 5:  # Leave buffer
                break
                
            x_old = x.copy()
            f_old = fx
            min_f = fx
            min_idx = -1
            min_dir = None
            min_x = x.copy()
            
            # Line search along each direction
            for i in range(len(directions)):
                if not time_ok():
                    break
                if calls >= self.budget - 5:
                    break
                    
                d = directions[i]
                
                # Simple line search: try steps in both directions
                # For Powell, we typically minimize f(x + t*d)
                # Use a simple pattern: try several step sizes
                best_t = 0
                best_val = fx
                
                # Try positive and negative steps with adaptive sizing
                step_sizes = [0.5, 1.0, 2.0, 4.0, 0.25, 0.1]
                
                for t in step_sizes:
                    for sign in [1, -1]:
                        if calls >= self.budget - 5 or not time_ok():
                            break
                        y = np.clip(x + sign * t * d, -5, 5)
                        fy = func(y)
                        calls += 1
                        if fy < best_val:
                            best_val = fy
                            best_t = sign * t
                            best_x_candidate = y.copy()
                
                if best_t != 0:
                    x = best_x_candidate
                    fx = best_val
                    if fx < min_f:
                        min_f = fx
                        min_idx = i
                        min_dir = d.copy()
                        min_x = x.copy()

            # Update best solution
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Check convergence
            if abs(fx - f_old) < 1e-10:
                break

            # Powell's direction update:
            # 1. Add direction to x_old that gave minimum
            # 2. Check if new minimum is better than old minimum
            # 3. If so, replace the worst direction with the new one
            
            # Compute new point: x_old + (x - x_old) in the direction of min
            # Actually, standard Powell: 
            # x_new = x_old + 2*(x - x_old) - x_prev_min  (reflection)
            # But simpler: just add the direction that gave min to the set
            
            if min_idx >= 0:
                # Remove the direction that gave the previous worst (highest f)
                # For simplicity, just replace the last direction with min_dir
                # More robust: keep the best directions
                directions[-1] = min_dir
                # Normalize
                norm = np.linalg.norm(directions[-1])
                if norm > 0:
                    directions[-1] = directions[-1] / norm

            # If we made progress, continue; else break
            if fx >= f_old:
                # No progress, try a smaller step or break
                if powell_iter > 5:
                    break

        # Final refinement: a few (1+1)-ES steps from best_x
        sigma = 0.1
        x = best_x.copy()
        fx = best_f
        for _ in range(min(20, self.budget - calls - 2)):
            if not time_ok() or calls >= self.budget - 1:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
