"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid (1+1)-ES with quadratic surrogate refinement.
    - Starts at origin for stable early progress.
    - Uses (1+1)-ES with adaptive step size for robust exploration.
    - Periodically fits a local quadratic model to predict basin minimum.
    - If predicted minimum is close and yields improvement, adopt it.
    - Time-guarded to stay within the 300s limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget

        # Initialize at origin (center of box) for symmetric landscapes
        x = np.zeros(d)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        sigma = 2.5
        min_sigma = 1e-6
        max_sigma = 10.0
        growth = 1.15
        decay = 0.92
        no_improve = 0

        # History for surrogate model (sliding window)
        hist_X = []
        hist_f = []
        max_hist = 30
        min_hist_for_surrogate = 21  # (d+1)(d+2)/2 for d=5

        calls = 1

        for i in range(budget - 1):
            if time.time() > deadline:
                break

            # Add current point to history
            hist_X.append(x.copy())
            hist_f.append(fx)
            if len(hist_X) > max_hist:
                hist_X.pop(0)
                hist_f.pop(0)

            # Try surrogate model refinement every ~15 steps if we have enough data
            used_surrogate = False
            if len(hist_X) >= min_hist_for_surrogate and i > 20:
                # Try every 15 steps early, then every 30 steps
                freq = 15 if i < 100 else 30
                if i % freq == 0:
                    new_x, new_f = self._try_surrogate_refinement(
                        func, x, fx, hist_X, hist_f, d
                    )
                    if new_x is not None:
                        x, fx = new_x, new_f
                        used_surrogate = True

            if used_surrogate:
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                no_improve = 0
                calls += 1
                continue

            # Standard (1+1)-ES step
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * growth, max_sigma)
                no_improve = 0
            else:
                sigma = max(sigma * decay, min_sigma)
                no_improve += 1
                # Stagnation: occasionally try a larger jump
                if no_improve >= 25:
                    sigma = min(sigma * 2.0, max_sigma)
                    no_improve = 0

            # Update best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Final high-precision polish pass with geometrically decreasing sigma
        x = best_x.copy()
        fx = best_f
        remaining = budget - calls
        if remaining > 0:
            sigma = 0.05
            polish_budget = min(remaining, max(50, budget // 5))

            for i in range(polish_budget):
                if time.time() > deadline:
                    break
                # Gradually decrease sigma for high precision
                sigma *= 0.98
                if sigma < 1e-8:
                    sigma = 1e-8

                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.05, 0.5)
                else:
                    sigma = max(sigma * 0.85, 1e-10)

        return best_x, best_f

    def _try_surrogate_refinement(self, func, x, fx, hist_X, hist_f, d):
        """
        Fit a quadratic model to recent history and try to move to its minimum.
        Returns (new_x, new_f) if successful, else (None, None).
        """
        try:
            X = np.array(hist_X)
            f_vals = np.array(hist_f)
            n = len(X)

            if n < 2 * d + 1:
                return None, None

            # Build design matrix for quadratic: [1, x, x_i*x_j for i<=j]
            d_quad = d * (d + 1) // 2
            A = np.ones((n, 1 + d + d_quad))

            for k in range(n):
                A[k, 1:1 + d] = X[k]
                idx = 1 + d
                for ii in range(d):
                    for jj in range(ii, d):
                        A[k, idx] = X[k, ii] * X[k, jj]
                        idx += 1

            # Solve least squares
            coeffs, _, _, _ = np.linalg.lstsq(A, f_vals, rcond=None)

            # Extract linear and quadratic coefficients
            c1 = coeffs[1:1 + d]

            # Reconstruct symmetric matrix C from quadratic coefficients
            C = np.zeros((d, d))
            idx = 1 + d
            for ii in range(d):
                for jj in range(ii, d):
                    val = coeffs[idx]
                    C[ii, jj] = val
                    if ii != jj:
                        C[jj, ii] = val
                    idx += 1

            # Check if Hessian (2C) is positive definite -> local minimum
            eigvals = np.linalg.eigvalsh(C)
            if eigvals[0] > 1e-8:  # Positive definite
                # Solve C * delta = -c1/2 => x_new = -C^{-1} c1 / 2
                try:
                    x_pred = -np.linalg.solve(C, c1) / 2.0
                    x_pred = np.clip(x_pred, -5, 5)

                    # Only take the step if it's reasonable (not too far)
                    dist = np.linalg.norm(x_pred - x)
                    if 1e-8 < dist < 5.0:
                        fy = func(x_pred)
                        if fy < fx - 1e-10:
                            return x_pred, fy
                except np.linalg.LinAlgError:
                    pass
        except Exception:
            pass

        return None, None
# EVOLVE-BLOCK-END
