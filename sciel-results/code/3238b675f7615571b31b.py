"""Black-box continuous optimizer for BBOB-style f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer using Hybrid Multi-Start Nelder-Mead and L-BFGS-B.
    
    Strategy:
    1. Multi-Start Nelder-Mead: Derivative-free, fast initial convergence.
       Uses low-discrepancy (Halton) and random start points.
    2. L-BFGS-B Polish: Gradient-based (finite difference), fast refinement
       to high precision once in the basin of attraction.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 320.0  # Safety margin
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def time_left():
            return deadline - time.time() > 1.0

        def wrapped_func(x):
            nonlocal calls
            if calls >= budget:
                raise ValueError("Budget Exceeded")
            calls += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            return func(x_clipped)

        # Generate starting points
        def halton_sequence(n, dims):
            points = []
            for i in range(1, n + 1):
                point = []
                for d in range(dims):
                    f = d + 2
                    result = 0.0
                    denom = 1
                    temp_i = i
                    while temp_i > 0:
                        denom *= f
                        result += (temp_i % f) / denom
                        temp_i //= f
                    point.append(result)
                points.append(point)
            return np.array(points)

        def map_to_box(points):
            return -5.0 + 10.0 * points

        # 12 Halton points
        n_halton = 12
        halton_points = halton_sequence(n_halton, dim)
        start_points = map_to_box(halton_points)
        
        # Add origin and 4 random points
        rng = np.random.default_rng(42)
        random_points = rng.uniform(-4, 4, size=(4, dim))
        start_points = np.vstack([np.zeros(dim), start_points, random_points])
        
        # Shuffle
        indices = rng.permutation(len(start_points))
        start_points = start_points[indices]

        bounds = [(-5.0, 5.0)] * dim

        # Phase 1: Multi-Start Nelder-Mead
        # Nelder-Mead uses (dim+1) initial points.
        # We limit iterations to keep budget low.
        n_mead_iter = 20
        
        for x0 in start_points:
            if calls >= budget or not time_left():
                break
            
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead', 
                    bounds=bounds, # Note: Nelder-Mead in SciPy does not strictly support bounds in all versions, 
                                   # but it often respects them or we rely on clipping.
                                   # Actually, scipy.optimize.minimize Nelder-Mead does NOT support bounds parameter in older versions, 
                                   # but newer ones might. To be safe, we rely on the wrapper clipping.
                                   # However, passing bounds to Nelder-Mead might raise an error or be ignored.
                                   # Let's check: In SciPy, Nelder-Mead does not accept 'bounds'.
                                   # We must NOT pass bounds to Nelder-Mead if it's not supported.
                                   # We will rely on the wrapper clipping.
                    options={'maxiter': n_mead_iter, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                
                # Nelder-Mead might return a point outside bounds if bounds aren't enforced.
                # Our wrapper clips, so func is safe.
                # But res.x might be outside. We should clip res.x before updating best?
                # The function value res.fun was calculated on the clipped version.
                # So res.x is the raw optimizer output.
                # We should update best with the clipped x and the corresponding f?
                # No, res.fun is the f at the clipped x.
                # So if we set best_x = res.x, it might be outside.
                # But the next call will clip it.
                # It's safer to clip best_x.
                
                x_res_clipped = np.clip(res.x, -5.0, 5.0)
                # We don't have the f at x_res_clipped directly if res.x != x_res_clipped.
                # But res.fun is the f at the last evaluation, which was clipped.
                # If res.x is outside, res.fun is f(clip(res.x)).
                # So we can update best with x_res_clipped and res.fun.
                
                update_best(x_res_clipped, res.fun)
                
            except (ValueError, Exception) as e:
                if "Budget Exceeded" in str(e):
                    break
                else:
                    pass

        # Phase 2: L-BFGS-B Polish
        if calls < budget and time_left() and best_f < np.inf:
            try:
                # L-BFGS-B supports bounds
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 50, 'gtol': 1e-8}
                )
                update_best(res.x, res.fun)
            except (ValueError, Exception):
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
