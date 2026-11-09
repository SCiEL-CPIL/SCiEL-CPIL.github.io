"""
Optimizer for BBOB-style Discus (f11) function.
Strategy: Multi-start Powell's Method with adaptive restarts.
Rationale: Powell's method maintains a set of conjugate directions and dynamically
replaces the worst direction with a new one formed by the overall step. This is
particularly effective for anisotropic functions like the Discus, where the
curvature varies significantly across dimensions. Unlike Nelder-Mead, which
maintains a simplex, Powell's method works with a set of directions that can
adapt to the local geometry of the function, leading to faster convergence.
Multi-starting with diverse initial points helps escape the flat plateau.
"""

import numpy as np
import time


class Optimizer:
    """Multi-start Powell's Method optimizer for smooth anisotropic landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track best
        best_x = None
        best_f = np.inf
        calls = 0

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def eval_point(x):
            nonlocal calls, best_x, best_f
            x = clip(x)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Diverse exploration (~30 calls)
        # Include origin, some structured points, and random points
        candidates = []
        candidates.append(np.zeros(dim))
        # Add some structured points
        for scale in [1.0, 2.0, 3.0]:
            for i in range(dim):
                pt = np.zeros(dim)
                pt[i] = scale
                candidates.append(pt)
                pt[i] = -scale
                candidates.append(pt)
        # Random points
        for _ in range(10):
            candidates.append(rng.uniform(-5, 5, dim))

        # Evaluate all candidates
        for c in candidates:
            if calls >= budget or time.time() > deadline:
                break
            eval_point(c)

        # Phase 2: Powell's Method refinement from best point
        if calls < budget and time.time() < deadline:
            self._powell(func, best_x, best_f, budget, dim, clip, eval_point, deadline)

        # Phase 3: If budget remains, try a second Powell from a different region
        if calls < budget - 20 and time.time() < deadline:
            # Pick a random point far from best
            far = rng.uniform(-5, 5, dim)
            if np.linalg.norm(far - best_x) < 3.0:
                far = best_x + rng.normal(0, 3, dim)
                far = clip(far)
            f_far = eval_point(far)
            if f_far < best_f + 50:  # only if not terrible
                if calls < budget - 20 and time.time() < deadline:
                    self._powell(func, far, f_far, budget, dim, clip, eval_point, deadline)

        # Phase 4: Final fine-tuning with small steps around best
        if calls < budget and time.time() < deadline:
            self._fine_tune(func, best_x, best_f, budget, dim, clip, eval_point, deadline, rng)

        return best_x, best_f

    def _line_minimize(self, x0, direction, f0, clip, eval_point, budget, deadline):
        """
        Minimize along a line: x = x0 + t * direction.
        Uses ternary search to find the minimum.
        Returns (x_min, f_min, t_min, num_calls_used)
        """
        # First, find a bracket for the minimum
        t = 0.1
        x1 = clip(x0 + t * direction)
        f1 = eval_point(x1)
        
        # If moving in the direction increases the function, try the opposite
        if f1 > f0:
            direction = -direction
            x1 = clip(x0 + t * direction)
            f1 = eval_point(x1)
        
        # Expand to find a bracket
        while f1 < f0 and t < 10.0:
            t *= 2.0
            x1 = clip(x0 + t * direction)
            f1 = eval_point(x1)
        
        # If we couldn't find a bracket, return the best we found
        if f1 >= f0:
            return x0, f0, 0.0, 2
        
        # Ternary search within [0, t]
        a, b = 0.0, t
        fa, fb = f0, f1
        for _ in range(20):  # 20 iterations of ternary search
            if time.time() > deadline or calls >= budget:
                break
            m1 = a + (b - a) / 3.0
            m2 = b - (b - a) / 3.0
            x_m1 = clip(x0 + m1 * direction)
            x_m2 = clip(x0 + m2 * direction)
            fm1 = eval_point(x_m1)
            fm2 = eval_point(x_m2)
            
            if fm1 < fm2:
                b, fb = m2, fm2
            else:
                a, fa = m1, fm1
            
            if abs(b - a) < 1e-6:
                break
        
        # Return the point with the lower function value
        if fa < fb:
            return clip(x0 + a * direction), fa, a, 2 + 40
        else:
            return clip(x0 + b * direction), fb, b, 2 + 40

    def _powell(self, func, x0, f0, budget, dim, clip, eval_point, deadline):
        """
        Powell's Method: maintains a set of directions and minimizes along each.
        Dynamically replaces the worst direction with a new one formed by the overall step.
        """
        nonlocal calls
        
        # Initialize directions as the standard basis
        directions = [np.eye(dim)[i] for i in range(dim)]
        x = x0.copy()
        f = f0
        max_iter = 50
        
        for it in range(max_iter):
            if calls >= budget or time.time() > deadline:
                break
            
            x_old = x.copy()
            f_old = f
            f_min = f
            i_min = 0
            
            # Minimize along each direction
            for i in range(dim):
                if calls >= budget or time.time() > deadline:
                    break
                x, f, t, _ = self._line_minimize(x, directions[i], f, clip, eval_point, budget, deadline)
                if f < f_min:
                    f_min = f
                    i_min = i
            
            # Check convergence
            if abs(f_old - f) < 1e-8:
                break
            
            # Compute the new direction
            x_new = clip(x_old + (x - x_old))
            f_new = eval_point(x_new)
            
            # If the new point is better, replace the worst direction
            if f_new < f_old:
                directions[i_min] = x - x_old
            
            x = x_new
            f = f_new

    def _fine_tune(self, func, x0, f0, budget, dim, clip, eval_point, deadline, rng):
        """Small-step local search around best point."""
        sigma = 0.1
        for _ in range(100):
            if calls >= budget or time.time() > deadline:
                break
            y = clip(x0 + sigma * rng.standard_normal(dim))
            fy = eval_point(y)
            if fy < best_f:
                x0 = y
                sigma *= 1.2
            else:
                sigma *= 0.7
                if sigma < 1e-6:
                    break
