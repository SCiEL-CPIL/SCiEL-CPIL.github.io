import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # 1. Deterministic Grid Search: Vertices + Center
        # 2^5 vertices + 1 center = 33 points
        vertices = []
        for i in range(2 ** self.dim):
            v = []
            for j in range(self.dim):
                if (i >> j) & 1:
                    v.append(5.0)
                else:
                    v.append(-5.0)
            vertices.append(v)
        
        # Add center
        vertices.append([0.0] * self.dim)
        
        points = np.array(vertices)
        
        best_x = None
        best_f = float('inf')
        calls = 0
        
        for p in points:
            f_val = func(p)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = p.copy()
        
        x = best_x
        fx = best_f
        
        # 2. Local Hill Climb (Pattern Search with Decaying Step)
        remaining = self.budget - calls
        if remaining <= 0:
            return x, fx
            
        # Smaller initial step to allow finer search in more iterations
        step = 1.0 
        min_step = 1e-7
        shrink_factor = 0.5
        
        while remaining > 0 and step > min_step:
            best_neighbor_x = None
            best_neighbor_val = fx
            
            # Try all 2*dim directions
            for i in range(self.dim):
                # Positive direction
                if remaining > 0:
                    y = x.copy()
                    y[i] += step
                    fy = func(y)
                    remaining -= 1
                    if fy < best_neighbor_val:
                        best_neighbor_val = fy
                        best_neighbor_x = y.copy()
                        
                # Negative direction
                if remaining > 0:
                    y = x.copy()
                    y[i] -= step
                    fy = func(y)
                    remaining -= 1
                    if fy < best_neighbor_val:
                        best_neighbor_val = fy
                        best_neighbor_x = y.copy()
                        
            if best_neighbor_x is not None:
                x = best_neighbor_x
                fx = best_neighbor_val
                # Optional: If improvement is tiny, shrink step to avoid oscillation
                # But standard pattern search keeps step. Let's stick to standard first.
            else:
                step *= shrink_factor

        return x, fx
