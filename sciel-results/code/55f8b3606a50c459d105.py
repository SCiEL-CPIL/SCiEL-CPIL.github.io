"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits linear structure: estimate gradient, jump to optimal vertex, refine."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        def timed_out():
            return time.time() > deadline

        dim = self.dim
        budget = self.budget
        calls = 0

        def call_func(x):
            nonlocal calls
            calls += 1
            return func(x)

        # Start at origin
        x0 = np.zeros(dim)
        f0 = call_func(x0)

        # Estimate gradient via central differences with small step
        # Use a step size that's small enough for linearity but large enough for numerical precision
        h = 1e-4
        grad = np.zeros(dim)
        for i in range(dim):
            if timed_out():
                break
            xp = x0.copy()
            xp[i] += h
            xm = x0.copy()
            xm[i] -= h
            fxp = call_func(np.clip(xp, -5, 5))
            fxm = call_func(np.clip(xm, -5, 5))
            grad[i] = (fxp - fxm) / (2 * h)

        if timed_out():
            return x0, f0

        # For a linear function f(x) = g·x + b, the minimum on [-5,5]^dim
        # is at the vertex where x_i = -5*sign(g_i) (if g_i != 0)
        # But the function has a hidden rotation and shift, so the "effective"
        # gradient in the original coordinates is what we estimated.
        # The minimum should be at: x_i = -5 if grad[i] > 0, +5 if grad[i] < 0
        # (since we want to minimize g·x, we go opposite to g)
        
        x_vertex = np.where(grad > 0, -5.0, 5.0)
        # Handle near-zero gradient components - keep at 0 or try both
        for i in range(dim):
            if abs(grad[i]) < 1e-6:
                x_vertex[i] = 0.0

        f_vertex = call_func(np.clip(x_vertex, -5, 5))

        # Also try the opposite vertex and a few other candidates
        candidates = [
            x0,
            x_vertex,
            -x_vertex,
        ]
        
        # Try vertices with some components flipped
        # For robustness, try a few more
        if not timed_out():
            # Try flipping each component one at a time from the best vertex
            best_x = x_vertex if f_vertex < f0 else x0
            best_f = min(f_vertex, f0)
            
            # If the vertex is much better, we're likely close
            # Do a local refinement around the vertex
            if f_vertex < f0:
                # Local search around vertex with small steps
                x_cur = x_vertex.copy()
                f_cur = f_vertex
                sigma = 0.1
                for _ in range(min(20, budget - calls - 1)):
                    if timed_out():
                        break
                    y = np.clip(x_cur + sigma * np.random.randn(dim), -5, 5)
                    fy = call_func(y)
                    if fy < f_cur:
                        x_cur, f_cur = y, fy
                        sigma *= 1.3
                    else:
                        sigma *= 0.8
                if f_cur < best_f:
                    best_x, best_f = x_cur, f_cur
            else:
                # Gradient estimate might be off; do a broader search
                # Try all 2^5 = 32 vertices? That's too many.
                # Try a subset: the vertex, its negation, and a few random points
                for _ in range(min(10, budget - calls - 1)):
                    if timed_out():
                        break
                    x_rand = np.random.uniform(-5, 5, dim)
                    fr = call_func(x_rand)
                    if fr < best_f:
                        best_x, best_f = x_rand, fr

        return best_x, best_f
# EVOLVE-BLOCK-END
