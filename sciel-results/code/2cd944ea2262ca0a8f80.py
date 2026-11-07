# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB f5 (Linear Slope).
    Key insight: f is exactly linear f(x) = c + g^T x.
    Strategy:
      1. Evaluate at origin and at +/- h * e_i for each coordinate (2*dim calls).
      2. Fit a full linear model (intercept + gradient) via least squares.
      3. The minimum of a linear function on a box is at a corner:
         x_i = -5 if g_i > 0, +5 if g_i < 0, 0 if g_i ≈ 0.
      4. Evaluate the predicted corner.
      5. If the prediction doesn't look perfect, do a few rounds of
         coordinate-wise refinement (1D search along each axis) to handle
         any residual non-linearity or numerical issues.
      6. Spend remaining budget on adaptive (1+1)-ES polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        dim = self.dim
        budget = self.budget
        calls = 0
        rng = np.random.default_rng(12345)

        def call(x):
            nonlocal calls
            if calls >= budget:
                return None
            val = func(np.asarray(x, dtype=float))
            calls += 1
            return val

        # --- Phase 1: Fit a full linear model ---
        # Evaluate at origin and at +/-h along each axis.
        # This gives 1 + 2*dim points. For dim=5, that's 11 calls.
        # The linear model is: f(x) = c + g^T x
        # Using centered differences for robustness.
        
        x0 = np.zeros(dim)
        f0 = call(x0)
        if f0 is None:
            return x0, float('inf')

        h = 1.0  # step size for gradient estimation
        points = [x0]
        values = [f0]

        for i in range(dim):
            xp = x0.copy()
            xp[i] += h
            xp = np.clip(xp, -5, 5)
            fp = call(xp)
            if fp is None:
                break
            points.append(xp)
            values.append(fp)

            xm = x0.copy()
            xm[i] -= h
            xm = np.clip(xm, -5, 5)
            fm = call(xm)
            if fm is None:
                break
            points.append(xm)
            values.append(fm)

        # Build design matrix for least squares: f = c + g^T x
        # A = [1, x^T], b = f values
        # Use numpy lstsq
        A = np.column_stack([np.ones(len(points)), np.array(points)])
        b = np.array(values, dtype=float)
        
        try:
            sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            c = sol[0]
            g = sol[1:]
        except:
            # Fallback: simple forward-difference gradient
            g = np.zeros(dim)
            for i in range(dim):
                if calls < budget:
                    xp = x0.copy()
                    xp[i] += h
                    xp = np.clip(xp, -5, 5)
                    fp = call(xp)
                    if fp is not None:
                        g[i] = (fp - f0) / h
                    else:
                        break

        # --- Phase 2: Predict minimum from linear model ---
        # For f(x) = c + g^T x on [-5,5]^d:
        #   x_i = -5 if g_i > 0, +5 if g_i < 0, 0 if |g_i| tiny
        eps_g = 1e-8
        x_pred = np.zeros(dim)
        for i in range(dim):
            if g[i] > eps_g:
                x_pred[i] = -5.0
            elif g[i] < -eps_g:
                x_pred[i] = 5.0
            else:
                x_pred[i] = 0.0

        f_pred = call(x_pred)
        if f_pred is None:
            f_pred = float('inf')

        # --- Phase 3: Coordinate-wise refinement ---
        # For each coordinate, do a small 1D search around the current best.
        # This handles any slight non-linearity or if the gradient estimate
        # was slightly off.
        x_cur = x_pred.copy()
        f_cur = f_pred

        for coord in range(dim):
            if time.time() > deadline or calls >= budget:
                break
            # Try moving in both directions along this coordinate
            for direction in [-1, 1]:
                for step in [1.0, 0.5, 0.1, 0.01]:
                    if calls >= budget or time.time() > deadline:
                        break
                    x_try = x_cur.copy()
                    x_try[coord] += direction * step
                    x_try = np.clip(x_try, -5, 5)
                    f_try = call(x_try)
                    if f_try is None:
                        break
                    if f_try < f_cur:
                        x_cur = x_try.copy()
                        f_cur = f_try
                    else:
                        # If moving in this direction didn't help, stop
                        break

        # --- Phase 4: Adaptive (1+1)-ES polish ---
        if time.time() < deadline and calls < budget:
            x = x_cur.copy()
            fx = f_cur
            sigma = 0.1
            no_improve = 0
            max_no_improve = 30

            while calls < budget:
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = call(y)
                if fy is None:
                    break
                if fy < fx:
                    x = y
                    fx = fy
                    no_improve = 0
                    sigma = min(sigma * 1.3, 3.0)
                else:
                    sigma = max(sigma * 0.85, 1e-6)
                    no_improve += 1
                    if no_improve > max_no_improve:
                        # Restart around best
                        x = x_cur.copy()
                        fx = f_cur
                        sigma = 0.5
                        no_improve = 0

        # Track best
        best_x = x_cur.copy()
        best_fx = f_cur
        if fx < best_fx:
            best_x = x.copy()
            best_fx = fx

        return best_x, best_fx
# EVOLVE-BLOCK-END
