"""Black-box continuous optimizer on the BBOB-style function f5 (Linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for a linear (or near-linear) function on [-5, 5]^d.
    Strategy:
      1. Evaluate the center (origin) as the first, guaranteed best-so-far.
      2. Bidirectional probing to estimate gradient accurately.
      3. Compute the analytic minimum vertex of the linear model on the box.
      4. Evaluate that vertex (the true minimum for a linear function).
      5. Polish with (1+1)-ES to handle slight non-linearities / noise.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before the 330s kill
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        call_count = 0
        best_x = None
        best_f = float('inf')

        def _call(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            if time.time() > deadline:
                return None
            call_count += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Step 1: Center (origin) -- first guaranteed best-so-far.
        base = np.zeros(d)
        f0 = _call(base)
        if f0 is None:
            return best_x, best_f

        # Step 2: Bidirectional probing to estimate gradient
        # We probe both +step and -step for each dimension.
        # To maximize early AOCC, we can interleave: for each dim, 
        # probe the direction more likely to improve first.
        # But since we don't know the gradient yet, we probe negative first
        # (common in BBOB that optimal is shifted, but let's be symmetric).
        # Actually, let's probe all negatives first, then all positives,
        # so if the gradient is positive, we get early improvements.
        
        step = 5.0  # reach the boundary from center
        f_pos = np.zeros(d)
        f_neg = np.zeros(d)
        
        # Probe negative directions first for early improvements
        for i in range(d):
            if time.time() > deadline:
                break
            p_neg = base.copy()
            p_neg[i] -= step
            fn = _call(p_neg)
            if fn is None:
                f_neg[i] = f0  # fallback
            else:
                f_neg[i] = fn
        
        # Probe positive directions
        for i in range(d):
            if time.time() > deadline:
                break
            p_pos = base.copy()
            p_pos[i] += step
            fp = _call(p_pos)
            if fp is None:
                f_pos[i] = f0  # fallback
            else:
                f_pos[i] = fp
        
        # Step 3: Estimate gradient using central difference
        # grad[i] = (f_pos[i] - f_neg[i]) / (2 * step)
        gradients = np.zeros(d)
        for i in range(d):
            if f_pos[i] != f0 or f_neg[i] != f0:  # if we got valid probes
                gradients[i] = (f_pos[i] - f_neg[i]) / (2.0 * step)
            else:
                gradients[i] = 0.0

        # Step 4: Determine the analytic minimum vertex of the linear model.
        # f(x) = f0 + grad^T x. Minimize over [-5,5]^d:
        #   x_i = -5 if grad[i] > 0, x_i = +5 if grad[i] < 0, else 0.
        vertex = np.zeros(d)
        for i in range(d):
            if gradients[i] > 0:
                vertex[i] = -5.0
            elif gradients[i] < 0:
                vertex[i] = 5.0
            else:
                vertex[i] = 0.0

        # Step 5: Evaluate the vertex (true minimum for a linear function).
        f_vertex = _call(vertex)
        if f_vertex is not None:
            best_f = f_vertex
            best_x = vertex.copy()

        # Step 6: Polish with (1+1)-ES to handle non-linearities / noise.
        if best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            remaining = budget - call_count
            
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(d)
                y = np.clip(y, -5.0, 5.0)
                fy = _call(y)
                if fy is None:
                    break
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                if sigma < 1e-6:
                    # Converged; keep spending budget to lock in best-so-far
                    continue

        return best_x, best_f
# EVOLVE-BLOCK-END
