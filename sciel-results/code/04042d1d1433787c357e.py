"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit linearity: estimate gradient, jump to optimal corner, then refine."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        d = self.dim
        budget = self.budget

        calls = 0
        best_x = None
        best_f = float('inf')

        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return best_f
            x = np.asarray(x, dtype=float).reshape(d)
            x = np.clip(x, -5.0, 5.0)
            calls += 1
            try:
                f_val = float(func(x))
            except Exception:
                return best_f
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Step 1: Evaluate at origin
        x0 = np.zeros(d)
        f0 = safe_call(x0)

        if time.time() > deadline or calls >= budget:
            return best_x, best_f

        # Step 2: Estimate gradient using central differences with large step
        # For linear f(x) = g^T x + c:
        # f(h*e_i) - f(-h*e_i) = 2*h*g_i  =>  g_i = (f(h*e_i) - f(-h*e_i)) / (2h)
        # Use h=5 to maximize signal (full box range)
        h = 5.0
        grad = np.zeros(d)
        for i in range(d):
            if time.time() > deadline or calls >= budget:
                break
            xp = np.zeros(d)
            xm = np.zeros(d)
            xp[i] = h
            xm[i] = -h
            fp = safe_call(xp)
            fm = safe_call(xm)
            grad[i] = (fp - fm) / (2.0 * h)

        if time.time() > deadline or calls >= budget:
            return best_x, best_f

        # Step 3: Construct optimal corner
        # For linear f: min on [-5,5]^d is at x_i = -5 if g_i > 0, +5 if g_i < 0
        x_corner = np.where(grad > 0, -5.0, 5.0)
        # Handle near-zero gradient components: they don't matter much
        x_corner[np.abs(grad) < 1e-10] = 0.0
        f_corner = safe_call(x_corner)

        # Step 4: Also check the opposite corner as a sanity check
        x_opp = np.where(grad > 0, 5.0, -5.0)
        x_opp[np.abs(grad) < 1e-10] = 0.0
        if time.time() < deadline and calls < budget:
            f_opp = safe_call(x_opp)
            # best_f already tracks the minimum

        # Step 5: Use remaining budget for local refinement
        # The function is linear, so the corner IS the global minimum.
        # But due to the hidden rotation/shift, our gradient estimate might be
        # slightly off if there's any non-linearity or numerical issue.
        # Do a simple random search around the best point found.
        rng = np.random.default_rng(12345)
        sigma = 1.0
        remaining = budget - calls

        for _ in range(remaining):
            if time.time() > deadline or calls >= budget:
                break
            y = best_x + sigma * rng.standard_normal(d)
            fy = safe_call(y)
            if fy < best_f:
                sigma *= 1.3
            else:
                sigma *= 0.85
                sigma = max(sigma, 0.01)

        return best_x, best_f
# EVOLVE-BLOCK-END
