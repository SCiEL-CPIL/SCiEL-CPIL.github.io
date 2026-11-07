"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase (1+1)-ES with quadratic surrogate refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        d = self.dim
        x = np.zeros(d)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        # Store recent evaluations for surrogate model
        # We need at least d+1 points for a quadratic fit, but more is better
        # A full quadratic in d dims has 1 + d + d(d+1)/2 = (d+1)(d+2)/2 coefficients
        # For d=5, that's 21 coefficients. We need at least 21 points.
        # We'll use a sliding window of the last 30 evaluations.
        hist_X = []
        hist_f = []
        max_hist = 30

        # Phase 1: Aggressive exploration
        sigma = 2.0
        sigma_cap = 15.0
        growth = 1.25
        decay = 0.85
        phase1_end = 250  # switch to polish after ~250 calls

        calls = 1

        for i in range(self.budget - 1):
            if time.time() > deadline:
                break

            # Add current point to history
            hist_X.append(x.copy())
            hist_f.append(fx)
            if len(hist_X) > max_hist:
                hist_X.pop(0)
                hist_f.pop(0)

            # Phase transition: switch to polish mode
            if i >= phase1_end:
                sigma = 0.05
                growth = 1.1
                decay = 0.7
                sigma_cap = 1.0

            # Try to use surrogate model if we have enough data
            used_surrogate = False
            if len(hist_X) >= 21 and i > 20:
                # Fit a quadratic model: f(x) ~ c0 + c1^T x + x^T C x
                # We'll use a simple least squares approach
                # Design matrix: [1, x, x_i*x_j for i<=j]
                try:
                    X = np.array(hist_X)
                    f_vals = np.array(hist_f)
                    
                    # Build design matrix
                    n = len(X)
                    # Features: constant, linear (d), quadratic (d(d+1)/2)
                    d_quad = d * (d + 1) // 2
                    A = np.ones((n, 1 + d + d_quad))
                    
                    for k in range(n):
                        A[k, 1:1+d] = X[k]
                        idx = 1 + d
                        for ii in range(d):
                            for jj in range(ii, d):
                                A[k, idx] = X[k, ii] * X[k, jj]
                                idx += 1
                    
                    # Solve least squares
                    coeffs, _, _, _ = np.linalg.lstsq(A, f_vals, rcond=None)
                    
                    # Extract gradient and Hessian
                    # f = c0 + c1^T x + x^T C x
                    # grad f = c1 + 2 C x
                    # Hessian = 2 C
                    c1 = coeffs[1:1+d]
                    
                    # Reconstruct C from coeffs
                    C = np.zeros((d, d))
                    idx = 1 + d
                    for ii in range(d):
                        for jj in range(ii, d):
                            val = coeffs[idx]
                            C[ii, jj] = val
                            if ii != jj:
                                C[jj, ii] = val
                            idx += 1
                    
                    # Check if Hessian is positive definite (local minimum)
                    # If not, we might be at a saddle or maximum
                    eigvals = np.linalg.eigvalsh(C)
                    
                    if eigvals[0] > 1e-6:  # Positive definite -> local min
                        # Solve C * delta = -c1/2 for the step
                        # Actually, grad f = c1 + 2 C x = 0 => C x = -c1/2
                        # So x_new = -C^{-1} c1 / 2
                        try:
                            x_pred = -np.linalg.solve(C, c1) / 2.0
                            # Clip to box
                            x_pred = np.clip(x_pred, -5, 5)
                            
                            # Only take the step if it's not too far
                            dist = np.linalg.norm(x_pred - x)
                            if dist < sigma * 5:  # Reasonable step
                                fy = func(x_pred)
                                calls += 1
                                if fy < fx:
                                    x, fx = x_pred, fy
                                    if fx < best_f:
                                        best_f = fx
                                        best_x = x.copy()
                                    used_surrogate = True
                                else:
                                    # Surrogate was wrong, fall back to ES
                                    pass
                        except np.linalg.LinAlgError:
                            pass
                    elif eigvals[-1] < -1e-6:
                        # Local maximum, move away
                        pass
                except Exception:
                    pass

            if not used_surrogate:
                # Standard (1+1)-ES step
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= growth
                else:
                    sigma *= decay

            sigma = np.clip(sigma, 1e-6, sigma_cap)

        return best_x, best_f
# EVOLVE-BLOCK-END
