import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0

    def __call__(self, func):
        # Track calls
        def safe_func(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            self.calls += 1
            return func(x)

        # 1. Global Grid Search
        # Points: -4, 0, 4
        grid_points = [-4.0, 0.0, 4.0]
        best_x = None
        best_f = float('inf')
        
        # Generate all combinations
        # 3^5 = 243 points
        for p0 in grid_points:
            for p1 in grid_points:
                for p2 in grid_points:
                    for p3 in grid_points:
                        for p4 in grid_points:
                            x = np.array([p0, p1, p2, p3, p4])
                            f = safe_func(x)
                            if f < best_f:
                                best_f = f
                                best_x = x

        x = best_x
        f_x = best_f

        # 2. Local Multi-Scale Coordinate Descent
        delta = 2.0
        shrink_factor = 0.7
        min_delta = 1e-4
        
        while self.calls < self.budget:
            moved = False
            for i in range(self.dim):
                if self.calls >= self.budget:
                    break
                
                # Try positive step
                x_plus = x.copy()
                x_plus[i] += delta
                if x_plus[i] > 5.0:
                    x_plus[i] = 5.0
                # Only evaluate if different from current
                if not np.allclose(x_plus, x):
                    f_plus = safe_func(x_plus)
                    if f_plus < f_x:
                        x = x_plus
                        f_x = f_plus
                        moved = True
                        continue # Move to next dimension, don't try negative yet? 
                                 # Or should we check negative too? 
                                 # If we moved, the direction was good. 
                                 # But maybe the other direction was better? 
                                 # Usually, if we move, we stick with it for this step size.
                                 # Let's just do one move per dim per pass.
                                 # This saves calls.

                # Try negative step (only if we didn't move in positive)
                x_minus = x.copy()
                x_minus[i] -= delta
                if x_minus[i] < -5.0:
                    x_minus[i] = -5.0
                if not np.allclose(x_minus, x):
                    f_minus = safe_func(x_minus)
                    if f_minus < f_x:
                        x = x_minus
                        f_x = f_minus
                        moved = True

            if not moved:
                delta *= shrink_factor
                if delta < min_delta:
                    break
            else:
                # If we moved, we might want to keep the same delta or slightly reduce?
                # Standard coordinate descent keeps delta same until no move.
                # This is fine.
                pass

        return x, f_x
