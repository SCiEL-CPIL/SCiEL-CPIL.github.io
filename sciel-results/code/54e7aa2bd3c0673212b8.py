import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # 1. Deterministic Grid Search (Vertices + Zero)
        # Generate vertices of [-5, 5]^dim
        # 2^dim vertices
        vertices = np.array(np.meshgrid(*[[-5.0, 5.0]] * self.dim)).T.reshape(-1, self.dim)
        zero_vec = np.zeros(self.dim)
        
        # Combine
        points = np.vstack([vertices, zero_vec])
        
        # Evaluate
        best_x = None
        best_f = float('inf')
        calls = 0
        
        for p in points:
            f_val = func(p)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = p
        
        x = best_x
        fx = best_f
        
        # 2. Local Hill Climb (Coordinate Descent with Decaying Step)
        remaining = self.budget - calls
        if remaining <= 0:
            return x, fx
            
        step = 1.0
        min_step = 1e-6
        
        while remaining > 0 and step > min_step:
            improved = False
            # Try moving in each coordinate direction
            for i in range(self.dim):
                # Positive direction
                if remaining > 0:
                    y = x.copy()
                    y[i] += step
                    if np.any(y < -5) or np.any(y > 5):
                         # Clip? The func clips, but we should be careful.
                         # The task says func clips. So we can just pass y.
                         pass
                    fy = func(y)
                    remaining -= 1
                    if fy < fx:
                        x, fx = y, fy
                        improved = True
                        
                # Negative direction
                if remaining > 0:
                    y = x.copy()
                    y[i] -= step
                    fy = func(y)
                    remaining -= 1
                    if fy < fx:
                        x, fx = y, fy
                        improved = True
                        
            if not improved:
                step *= 0.5
            else:
                # If we improved, we might want to continue with same step or reset?
                # Standard coordinate descent often keeps the step.
                # But if we are far, we might want to take bigger steps?
                # The directive says "fixed-step". I'll interpret this as the step size is fixed for the current "phase" of search, 
                # and we reduce it when stuck.
                pass

        return x, fx
