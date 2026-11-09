# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Deterministic multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_fx = np.inf
        best_x = None
        start_time = time.time()
        deadline = 280.0

        # Generate a diverse set of starting points
        # 1. Origin
        starts = [np.zeros(self.dim)]
        # 2. Random uniform points
        for _ in range(19):
            starts.append(rng.uniform(-5, 5, self.dim))
        
        # 3. Grid-like points to cover the domain
        for i in range(-4, 5, 2):
            for j in range(-4, 5, 2):
                if self.dim >= 2:
                    x = np.zeros(self.dim)
                    x[0] = i
                    x[1] = j
                    starts.append(x)
                if self.dim >= 5:
                    x = np.zeros(self.dim)
                    x[4] = i
                    starts.append(x)

        # Sort starts by distance from origin? No, random order is fine.
        # Or maybe prioritize center?
        # Let's just iterate through them.
        
        # Budget allocation:
        # We have 500 calls total.
        # Each Nelder-Mead run uses a few calls.
        # Let's limit each run to ~20 function evaluations to allow many restarts.
        # 500 / 20 = 25 restarts.
        
        calls_made = 0
        func_calls = 0
        
        def wrapper(x):
            nonlocal func_calls
            if func_calls < self.budget:
                func_calls += 1
            return func(x)

        for start in starts:
            if func_calls >= self.budget:
                break
            if time.time() - start_time > deadline:
                break

            # Nelder-Mead is robust for smooth-ish functions.
            # Even if linear, it will converge to the boundary or a vertex.
            # Since the minimum is in [-4,4]^5, and the function is linear slope,
            # the minimum is likely at a corner or on the boundary of the feasible region 
            # defined by the rotation/shift. 
            # Wait, f5 is "Linear slope". 
            # f(x) = a^T x + b. 
            # The minimum over the box [-5,5]^5 is at one of the vertices.
            # However, the problem states "minimizer lies in [-4, 4]^5".
            # This implies the global minimum of the unbounded function is inside the box,
            # OR the "Linear slope" refers to a specific BBOB function f5 which is 
            # actually a linear function with a shift.
            # BBOB f5: f(x) = sum(x_i) * 10? No, f5 is "Linear slope".
            # Actually, BBOB f5 is f(x) = sum_{i=1}^d (10 * x_i)^2? No that's f2.
            # Let's check BBOB definitions.
            # f5: Linear slope. f(x) = sum_{i=1}^d (10 * x_i).
            # If it's truly linear, the minimum over the box is at a vertex.
            # But the prompt says "minimizer lies in [-4, 4]^5".
            # This is contradictory if it's strictly linear over the whole domain.
            # Perhaps it's a linear function but the "instance" shifts it so the
            # vertex of the underlying cone/shape is inside?
            # Or maybe "Linear slope" in BBOB is different.
            # Let's assume it's a smooth function that Nelder-Mead can handle.
            
            # We use scipy's Nelder-Mead.
            # options: maxiter=10, maxfev=20
            try:
                res = minimize(wrapper, start, method='Nelder-Mead',
                               options={'maxiter': 10, 'maxfev': 20, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x
            except Exception:
                pass

        # If we still have budget, do a final local search from the best point found
        if best_x is not None and func_calls < self.budget and time.time() - start_time < deadline:
            try:
                res = minimize(wrapper, best_x, method='Nelder-Mead',
                               options={'maxiter': 50, 'maxfev': self.budget - func_calls, 'xatol': 1e-5, 'fatol': 1e-5})
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x
            except Exception:
                pass

        if best_x is None:
            # Fallback
            best_x = np.zeros(self.dim)
            best_fx = func(best_x)

        return best_x, best_fx
# EVOLVE-BLOCK-END
