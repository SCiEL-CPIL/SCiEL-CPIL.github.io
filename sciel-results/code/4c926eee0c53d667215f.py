import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        # Track state
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        calls = 0
        dim = self.dim
        
        # Helper to safely call func and update best
        def try_eval(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                fx = func(x)
            except Exception:
                return float('inf')
            
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # 1. Initial evaluation at origin
        try_eval(np.zeros(dim))
        
        # 2. Greedy Coordinate Descent to find the global vertex
        # For a linear function f(x) = c^T x + b, the minimum over a box is at a vertex.
        # We can determine the sign of each gradient component c_i by comparing f at
        # x + delta * e_i and x - delta * e_i.
        # If f(x + d e_i) < f(x - d e_i), the function decreases in the +i direction, so the min is at x_i = 5.
        # If f(x - d e_i) < f(x + d e_i), the function decreases in the -i direction, so the min is at x_i = -5.
        
        # We start from origin. We will determine the optimal bound for each coordinate.
        # To be robust against the "shift" mentioned in the prompt (though linear slope implies the
        # gradient is constant), we assume the standard linear form.
        # The prompt says "Linear slope", which usually means f(x) = w^T x + b.
        # In this case, the minimum is indeed at a vertex determined by the signs of w.
        
        # Let's probe each axis from the current best point (origin initially).
        # We use a small step for probing to avoid numerical issues, but since it's linear,
        # any step works. Let's use step = 1.0.
        probe_step = 1.0
        
        current_x = np.zeros(dim)
        
        # Phase 1: Determine the "direction" (sign) for each coordinate
        # We evaluate pairs of points for each coordinate to determine the slope sign.
        # This takes 2 * dim calls.
        # However, we can be smarter. We just need to find the vertex.
        # Let's just evaluate the two vertices for each coordinate relative to the origin?
        # No, that's 2*dim calls.
        
        # Better strategy:
        # 1. Evaluate origin.
        # 2. For each coordinate i, evaluate x_plus_i (origin + step * e_i) and x_minus_i (origin - step * e_i).
        # 3. If f(x_plus) < f(x_minus), set optimal_i = 5. Else set optimal_i = -5.
        # 4. Evaluate the resulting vertex.
        
        # This uses 1 + 2*dim calls. For dim=5, that's 11 calls.
        # This is extremely cheap and should find the global optimum exactly if the function is linear.
        
        optimal_vertex = np.zeros(dim)
        
        for i in range(dim):
            x_plus = current_x.copy()
            x_plus[i] += probe_step
            
            x_minus = current_x.copy()
            x_minus[i] -= probe_step
            
            f_plus = try_eval(x_plus)
            f_minus = try_eval(x_minus)
            
            # Determine which direction is better
            if f_plus < f_minus:
                optimal_vertex[i] = 5.0
            else:
                optimal_vertex[i] = -5.0
                
        # Evaluate the predicted global optimum (vertex)
        try_eval(optimal_vertex)
        
        # 3. Refinement: In case the function is not perfectly linear or there are numerical quirks,
        #    or if the "linear slope" description is slightly misleading (e.g., shifted linear),
        #    we perform a short local search from the best point found so far.
        #    Since we likely found the global min, this is just a safety net.
        
        if calls < self.budget:
            # Simple (1+1)-ES refinement
            sigma = 0.1
            current_x = best_x.copy()
            current_f = best_f
            rng = np.random.default_rng(42)
            
            # Run for a fixed number of steps or until budget/time
            # We have plenty of budget left (500 - 11 = 489 calls).
            # But 300s limit is the real constraint.
            # Let's run until we hit 90% of budget or time limit.
            
            while calls < self.budget * 0.9:
                if time.time() > deadline:
                    break
                
                step = sigma * rng.standard_normal(dim)
                y = current_x + step
                y = np.clip(y, -5, 5)
                
                fy = try_eval(y)
                
                if fy < current_f:
                    current_x = y
                    current_f = fy
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                    
                if sigma < 1e-8:
                    break

        return best_x, best_f
