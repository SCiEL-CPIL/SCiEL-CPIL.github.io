"""Rewritten program: gradient-based optimizer for linear slope BBOB f5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Gradient-based optimizer exploiting linearity of f5."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget
        calls = 0

        def safe_call(x):
            nonlocal calls
            if calls >= budget:
                return None
            calls += 1
            return func(x)

        # Initial point: center of box
        x0 = np.zeros(d)
        f0 = safe_call(x0)
        if f0 is None:
            return x0, f0

        best_x = x0.copy()
        best_f = f0

        # Estimate gradient using forward differences from 0
        # For a linear function f(x) = g^T x + b, f(e_i) - f(0) = g_i
        # This requires d additional calls.
        grad = np.zeros(d)
        for i in range(d):
            if calls >= budget or time.time() > deadline:
                break
            ei = np.zeros(d)
            ei[i] = 1.0
            fei = safe_call(ei)
            if fei is None:
                break
            grad[i] = fei - f0

        grad_norm = np.linalg.norm(grad)
        
        # If gradient is zero, the function is constant (or we are in a flat region)
        if grad_norm < 1e-10:
            # Try a few random points to be sure
            for _ in range(min(10, budget - calls)):
                if time.time() > deadline:
                    break
                xr = rng.uniform(-5, 5, d)
                fxr = safe_call(xr)
                if fxr is None:
                    break
                if fxr < best_f:
                    best_x = xr.copy()
                    best_f = fxr
            return best_x, best_f

        # For a linear function, the minimum over [-5, 5]^d is at the vertex
        # x* where x*_i = -5 * sign(grad_i)
        x_vertex = -5.0 * np.sign(grad)
        # Handle zero components: if grad_i is 0, sign is 0, x*_i is 0.
        # This is fine as the function is flat in that direction.
        
        f_vertex = safe_call(x_vertex)
        if f_vertex is not None and f_vertex < best_f:
            best_f = f_vertex
            best_x = x_vertex.copy()

        # If we have budget left, we might want to verify if the minimum is indeed
        # at the vertex. If the problem statement "minimizer in [-4,4]" implies
        # the function is not purely linear or has a different structure, we might
        # need to search locally. However, for BBOB f5, vertex is the answer.
        # Let's do a quick check of neighbors of the vertex in case of numerical issues
        # or if the function is slightly non-linear.
        
        # Check a few points near the vertex
        if budget - calls > 5 and time.time() < deadline:
            # Try moving slightly inwards from the vertex
            # If the function is linear, this will increase f, so we won't update.
            # If it's convex with min inside, this might help.
            shrink_factors = [0.9, 0.8, 0.5]
            for sf in shrink_factors:
                if time.time() > deadline:
                    break
                x_try = x_vertex * sf
                fx_try = safe_call(x_try)
                if fx_try is None:
                    break
                if fx_try < best_f:
                    best_f = fx_try
                    best_x = x_try.copy()
            
            # If we found a better point inside, do a local Nelder-Mead
            if best_x is not x_vertex:
                # Simple coordinate descent polish
                for _ in range(min(10, budget - calls)):
                    if time.time() > deadline:
                        break
                    i = rng.integers(d)
                    for delta in [0.1, 0.5, 1.0]:
                        for sign in [1, -1]:
                            if calls >= budget:
                                break
                            x_try = best_x.copy()
                            x_try[i] += sign * delta
                            x_try = np.clip(x_try, -5, 5)
                            fx_try = safe_call(x_try)
                            if fx_try is None:
                                break
                            if fx_try < best_f:
                                best_f = fx_try
                                best_x = x_try.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
