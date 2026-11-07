"""Rewritten optimizer using a local quadratic surrogate model."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Local quadratic surrogate model optimizer for BBOB f12 (Bent cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # Initialize with a few random points to build initial history
        rng = np.random.default_rng(42)

        # Start with a reasonable initial point
        x = rng.uniform(-3, 3, n)
        fx = func(x)

        # History: list of (x, f) pairs
        history = [(x.copy(), fx)]
        call_count = 1

        # Initial step size
        sigma = 1.0

        # Minimum number of points needed for quadratic fit
        min_points = n * (n + 1) // 2 + 1  # 16 for n=5
        # Use up to 25 points for the fit
        max_points = 25

        # Track best point found
        x_best = x.copy()
        fx_best = fx

        # Number of consecutive failures to adapt sigma
        fail_count = 0

        while call_count < budget:
            # If we have enough points, try to fit a quadratic
            if len(history) >= min_points:
                # Take the last max_points evaluations
                recent = history[-max_points:]
                X = np.array([pt for pt, _ in recent])
                F = np.array([f for _, f in recent])

                # Fit quadratic: f(x) ≈ c + x^T b + x^T A x
                # Design matrix for quadratic terms
                quad_terms = []
                for xi in X:
                    row = [1.0]
                    row.extend(xi)
                    # Squared and cross terms
                    for i in range(n):
                        for j in range(i, n):
                            row.append(xi[i] * xi[j])
                    quad_terms.append(row)
                Q = np.array(quad_terms)

                # Solve least squares
                try:
                    coeffs, residuals, rank, sv = np.linalg.lstsq(Q, F, rcond=None)
                except np.linalg.LinAlgError:
                    coeffs = None

                if coeffs is not None:
                    c = coeffs[0]
                    b = coeffs[1:1 + n]
                    # Reconstruct A from the quadratic coefficients
                    A = np.zeros((n, n))
                    idx = 1 + n  # start index for quadratic terms
                    for i in range(n):
                        for j in range(i, n):
                            val = coeffs[idx]
                            if i == j:
                                A[i, i] = val
                            else:
                                A[i, j] = val / 2.0
                                A[j, i] = val / 2.0
                            idx += 1

                    # Find minimizer of the quadratic model
                    # grad = b + 2*A*x = 0 => x* = -0.5 * A^{-1} * b
                    try:
                        # Add small regularization for stability
                        A_reg = A + 1e-6 * np.eye(n)
                        x_star = -0.5 * np.linalg.solve(A_reg, b)
                    except np.linalg.LinAlgError:
                        x_star = None

                    if x_star is not None:
                        # Clip to bounds
                        x_star = np.clip(x_star, -5, 5)

                        # Take a step toward x_star, but limit the step size
                        step = x_star - x
                        step_norm = np.linalg.norm(step)
                        if step_norm > sigma:
                            step = step / step_norm * sigma

                        x_new = np.clip(x + step, -5, 5)
                        fx_new = func(x_new)
                        call_count += 1

                        if fx_new < fx:
                            x, fx = x_new, fx_new
                            if fx_new < fx_best:
                                x_best, fx_best = x_new.copy(), fx_new
                            # Adapt sigma: increase if we made progress
                            sigma = min(sigma * 1.3, 4.0)
                            fail_count = 0
                        else:
                            sigma = max(sigma * 0.7, 0.05)
                            fail_count += 1

                        history.append((x_new.copy(), fx_new))
                        continue

            # If quadratic fit failed or we don't have enough points,
            # do a random perturbation
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            call_count += 1

            if fy < fx:
                x, fx = y, fy
                if fy < fx_best:
                    x_best, fx_best = y.copy(), fy
                sigma = min(sigma * 1.2, 4.0)
                fail_count = 0
            else:
                sigma = max(sigma * 0.85, 0.05)
                fail_count += 1

            history.append((y.copy(), fy))

        return x_best, fx_best
# EVOLVE-BLOCK-END
