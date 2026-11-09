import numpy as np
import time

class Optimizer:
    """
    Gradient-Descent Optimizer for BBOB f5 (Linear Slope).
    
    Strategy:
    1. Exploit the linearity of BBOB f5. The function is of the form f(x) = a^T x + b.
    2. The minimum over the box [-5, 5]^d is always at a vertex.
    3. We estimate the gradient vector 'a' using finite differences.
    4. We identify the optimal vertex by moving in the direction of -a (negative gradient)
       until we hit the boundary. For a linear function, the best point is the vertex
       that minimizes a^T x.
    5. This requires O(d) calls to estimate the gradient and O(1) to find the vertex,
       followed by a few refinement steps if the finite difference estimate is noisy
       due to the hidden rotation/shift (though rotation preserves linearity).
       
    Why this is different:
    - Current: Population-based local search (Nelder-Mead/ES) which wanders.
    - New: Deterministic gradient estimation + direct vertex projection.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls_made = 0
        self.best_f = float('inf')
        self.best_x = None
        self.start_time = None
        self.deadline = None

    def __call__(self, func):
        self.start_time = time.time()
        # 300s limit, use 280s to be safe against overhead
        self.deadline = self.start_time + 280.0
        self.calls_made = 0
        self.best_f = float('inf')
        self.best_x = None

        # 1. Wrapped function for strict tracking
        def wrapped_func(x):
            if self.calls_made >= self.budget:
                return float('inf')
            if time.time() > self.deadline:
                return float('inf')
            
            self.calls_made += 1
            try:
                x_arr = np.asarray(x, dtype=float)
                x_clipped = np.clip(x_arr, -5, 5)
                val = float(func(x_clipped))
                
                if val < self.best_f:
                    self.best_f = val
                    self.best_x = x_clipped.copy()
                    
                return val
            except Exception:
                return float('inf')

        # 2. Phase 1: Estimate Gradient
        # For a linear function f(x) = a^T x + b, the gradient is constant 'a'.
        # We can estimate 'a' by evaluating f at the origin and at points shifted
        # along each axis.
        
        # Start at the center (0, 0, ..., 0) which is safe within [-5, 5]
        x0 = np.zeros(self.dim)
        f0 = wrapped_func(x0)
        
        if self.calls_made >= self.budget or time.time() > self.deadline:
            return self.best_x, self.best_f

        # Estimate gradient components
        # Use a small step size for finite differences, but since it's linear,
        # the size doesn't matter for accuracy, only for numerical stability.
        # However, we want to be robust to any potential non-linearities in the 
        # "hidden" transformation if it's not perfectly linear (though BBOB f5 is).
        # Let's use a step of 1.0 to get a robust estimate of the slope per unit.
        step = 1.0
        gradient = np.zeros(self.dim)
        
        for i in range(self.dim):
            if self.calls_made >= self.budget or time.time() > self.deadline:
                break
            
            # Evaluate at x0 + step * e_i
            x_plus = x0.copy()
            x_plus[i] += step
            f_plus = wrapped_func(x_plus)
            
            # Evaluate at x0 - step * e_i
            # This allows for a central difference, which is more robust if there's
            # any slight curvature or noise, though for pure linear it's identical.
            if self.calls_made >= self.budget or time.time() > self.deadline:
                # If we run out of budget, use forward difference
                if f_plus < float('inf'):
                    gradient[i] = (f_plus - f0) / step
                continue

            x_minus = x0.copy()
            x_minus[i] -= step
            f_minus = wrapped_func(x_minus)
            
            # Central difference
            gradient[i] = (f_plus - f_minus) / (2 * step)

        if self.calls_made >= self.budget or time.time() > self.deadline:
            return self.best_x, self.best_f

        # 3. Phase 2: Find the Optimal Vertex
        # For f(x) = a^T x + b, the minimum over the box [-5, 5]^d is at the vertex
        # x* where x*_i = -5 if a_i > 0, and x*_i = 5 if a_i < 0.
        # If a_i is close to 0, the coordinate doesn't matter much, but we can pick
        # the one that minimizes the contribution.
        
        x_opt = np.zeros(self.dim)
        for i in range(self.dim):
            if gradient[i] > 0:
                x_opt[i] = -5.0
            elif gradient[i] < 0:
                x_opt[i] = 5.0
            else:
                # If gradient is exactly 0 (or very close), the function is flat
                # in this direction. We can stay at 0 or pick randomly.
                x_opt[i] = 0.0

        # Evaluate the candidate optimal vertex
        f_opt_candidate = wrapped_func(x_opt)
        
        if self.calls_made >= self.budget or time.time() > self.deadline:
            return self.best_x, self.best_f

        # 4. Phase 3: Refinement / Verification
        # The finite difference estimate might be slightly off if the function
        # has any hidden non-linearities or if the "linear slope" is actually
        # a linear slope in a rotated coordinate system (which is still linear).
        # However, if the rotation is orthogonal, the gradient in the original
        # space is still constant.
        
        # To be safe, we can perform a few steps of gradient descent starting
        # from x_opt to ensure we haven't missed a better vertex due to noise
        # in the gradient estimation. But for a true linear function, x_opt IS
        # the global minimum.
        
        # Let's do a quick check: evaluate the 8 nearest vertices to x_opt?
        # No, that's expensive. 
        # Instead, let's just run a few iterations of a simple (1+1)-ES or 
        # Nelder-Mead starting from x_opt to polish the result in case the
        # gradient estimate was slightly noisy and pointed to the wrong vertex
        # for a coordinate with a very small gradient magnitude.
        
        # Actually, if the gradient is small, the function is flat, so any point
        # in that dimension is equally good. The risk is if the gradient sign is
        # wrong due to noise. 
        # Let's evaluate the "opposite" vertex for any dimension where the gradient
        # magnitude was very small (close to 0) to be sure.
        
        # Define a threshold for "small" gradient
        # The scale of f is unknown, but we can look at relative magnitude.
        # Let's just trust the sign for now, but if we have budget, we can
        # refine.
        
        # If we have budget left, let's do a simple local search from x_opt
        # to ensure we are at the true minimum, just in case the finite
        # difference was misleading (e.g. due to clipping effects if x0 was
        # near boundary, but x0=0 is safe).
        
        if self.budget - self.calls_made > 10:
            # Simple (1+1)-ES refinement
            x_curr = x_opt.copy()
            f_curr = self.best_f
            sigma = 0.5
            rng = np.random.default_rng(42)
            
            for _ in range(20): # Max 20 refinement steps
                if self.calls_made >= self.budget or time.time() > self.deadline:
                    break
                
                y = np.clip(x_curr + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = wrapped_func(y)
                
                if fy < f_curr:
                    x_curr = y
                    f_curr = fy
                    sigma *= 1.1
                else:
                    sigma *= 0.8
                    
                if sigma < 1e-4:
                    break

        return self.best_x, self.best_f
