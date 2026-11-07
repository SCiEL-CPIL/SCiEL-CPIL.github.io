import numpy as np
import time

class Optimizer:
    """Quadratic surrogate model optimizer for BBOB f12 (Bent cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0

        rng = np.random.default_rng(42)

        # Initialize
        x = rng.uniform(-3, 3, d)
        fx = func(x)
        call_count = 1
        best_x = x.copy()
        best_f = fx

        # History for quadratic fit
        # Need at least d*(d+1)/2 + 1 = 16 points for d=5
        min_points = d * (d + 1) // 2 + 1
        max_points = 30
        hist_X = []
        hist_f = []

        # Step size for random perturbations when surrogate fails
        sigma = 1.5
        sigma_min = 0.01
        sigma_max = 4.0

        # Add initial point to history
        hist_X.append(x.copy())
        hist_f.append(fx)

        while call_count < budget:
            if time.time() > deadline:
                break

            # Try quadratic surrogate if we have enough points
            if len(hist_X) >= min_points:
                recent = hist_X[-max_points:]
                recent_f = hist_f[-max_points:]
                X = np.array(recent)
                F = np.array(recent_f)

                # Build design matrix for quadratic model:
                # f(x) ≈ c + b^T x + x^T A x
                # Features: [1, x_1, ..., x_d, x_1^2, x_1*x_2, ..., x_d^2]
                n_pts = len(X)
                m = d * (d + 1) // 2
                Q = np.zeros((n_pts, 1 + d + m))

                for i in range(n_pts):
                    xi = X[i]
                    Q[i, 0] = 1.0
                    Q[i, 1:1+d] = xi
                    idx = 1 + d
                    for a in range(d):
                        for b in range(a, d):
                            Q[i, idx] = xi[a] * xi[b]
                            idx += 1

                try:
                    coeffs, residuals, rank, sv = np.linalg.lstsq(Q, F, rcond=None)
                except np.linalg.LinAlgError:
                    coeffs = None

                if coeffs is not None:
                    c = coeffs[0]
                    b_vec = coeffs[1:1+d]
                    # Reconstruct A from quadratic coefficients
                    A = np.zeros((d, d))
                    idx = 1 + d
                    for a in range(d):
                        for b in range(a, d):
                            val = coeffs[idx]
                            if a == b:
                                A[a, a] = val
                            else:
                                A[a, b] = val / 2.0
                                A[b, a] = val / 2.0
                            idx += 1

                    # Check if A is positive definite enough for a minimum
                    try:
                        eigvals = np.linalg.eigvalsh(A)
                        if np.min(eigvals) > -1e-3:  # Allow slightly indefinite
                            # Find minimizer: grad = b_vec + 2*A*x = 0
                            # x* = -0.5 * A^{-1} * b_vec
                            A_reg = A + 1e-8 * np.eye(d)
                            x_star = -0.5 * np.linalg.solve(A_reg, b_vec)
                            x_star = np.clip(x_star, -5, 5)

                            # Damped step toward x_star
                            step = x_star - x
                            step_norm = np.linalg.norm(step)
                            if step_norm > 0:
                                step_size = min(step_norm, sigma)
                                x_new = np.clip(x + step / step_norm * step_size, -5, 5)
                            else:
                                x_new = x_star

                            fx_new = func(x_new)
                            call_count += 1

                            if fx_new < fx:
                                x, fx = x_new, fx_new
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                                sigma = min(sigma * 1.3, sigma_max)
                            else:
                                sigma = max(sigma * 0.7, sigma_min)

                            hist_X.append(x_new.copy())
                            hist_f.append(fx_new)
                            if len(hist_X) > max_points:
                                hist_X.pop(0)
                                hist_f.pop(0)
                            continue

                    except np.linalg.LinAlgError:
                        pass

            # Fallback: random perturbation
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            call_count += 1

            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma = min(sigma * 1.2, sigma_max)
            else:
                sigma = max(sigma * 0.85, sigma_min)

            hist_X.append(y.copy())
            hist_f.append(fy)
            if len(hist_X) > max_points:
                hist_X.pop(0)
                hist_f.pop(0)

        return best_x, best_f
