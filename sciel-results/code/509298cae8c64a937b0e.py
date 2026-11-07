"""DFTR optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Derivative-Free Trust-Region method with quadratic surrogate."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial point
        x = np.zeros(n)
        fx = func(x)
        f_best = fx
        x_best = x.copy()

        # Trust region radius
        rho = 1.0
        rho_min = 1e-3
        rho_max = 10.0

        calls_used = 1

        # Precompute a set of directions for building the quadratic model.
        # We use 2n coordinate directions plus a few random directions.
        # Actually, for a quadratic model in n dimensions, we need at least n(n+1)/2 + 1 points
        # but we can use a simpler approach: maintain a local quadratic model using
        # directional derivatives estimated by finite differences.
        #
        # A simpler and more robust approach: use a local quadratic model built from
        # a set of sample points around the current center. We'll maintain a model
        # of the form: f(x) ≈ f0 + g^T (x - x0) + 0.5 (x - x0)^T H (x - x0)
        #
        # To estimate g and H, we need evaluations at x0 + d_i for various directions d_i.
        # We'll use 2n coordinate directions to estimate the gradient, and then use
        # additional points to estimate the Hessian.
        #
        # However, with a limited budget, we need to be efficient. Let's use a
        # coordinate-wise quadratic model or a simpler approach.
        #
        # Alternative: Use a pattern search or Nelder-Mead with adaptive step size,
        # but the directive asks for DFTR with quadratic surrogates.
        #
        # Let's implement a simplified DFTR:
        # 1. At each iteration, build a quadratic model from a set of points.
        # 2. Solve the trust-region subproblem.
        # 3. Accept/reject based on actual vs predicted improvement.

        # We'll maintain a set of points for the local model.
        # Start with the center and 2n points along coordinate axes.

        def build_model(x0, f0, points, fvals, rho):
            """
            Build a quadratic model: f(x) ≈ f0 + g^T (x - x0) + 0.5 (x - x0)^T H (x - x0)
            points: (m, n) array of points
            fvals: (m,) array of function values
            Returns g (n,) and H (n, n)
            """
            n = len(x0)
            # points should include x0 as the first point, with fvals[0] = f0
            # Let's assume points[0] = x0 and fvals[0] = f0
            m = len(points)
            if m < n + 1:
                # Not enough points, return zeros
                return np.zeros(n), np.zeros((n, n))

            # Build the design matrix for the quadratic model
            # For each point i, let d_i = points[i] - x0
            # We want to solve: fvals[i] - f0 = g^T d_i + 0.5 d_i^T H d_i
            # This is a linear system in the parameters [g; vec(H)]
            # But H is symmetric, so we have n + n(n+1)/2 - 1 parameters (since 0.5 factor)
            # Actually, let's use a simpler approach: estimate gradient by finite differences
            # and Hessian by second-order differences.

            # Estimate gradient using central differences if possible, else forward differences
            g = np.zeros(n)
            for j in range(n):
                # Find points along dimension j
                # Look for points that differ from x0 only in dimension j
                found_forward = False
                found_backward = False
                for i in range(1, m):
                    d = points[i] - x0
                    if np.allclose(d, 0) or not np.allclose(np.abs(d), np.abs(d[j]) * (np.eye(n)[j] | np.zeros((n, n))), atol=1e-10):
                        continue
                    # Check if d is along axis j
                    if np.allclose(d, d[j] * np.eye(n)[j], atol=1e-10):
                        if d[j] > 0:
                            found_forward = True
                            g[j] = (fvals[i] - f0) / d[j]
                        else:
                            found_backward = True
                            g[j] = (fvals[i] - f0) / d[j]
                        break
                if not found_forward and not found_backward:
                    # Use a small step
                    step = rho / 10
                    x_plus = x0.copy()
                    x_plus[j] += step
                    x_minus = x0.copy()
                    x_minus[j] -= step
                    # We don't have these values, so just use forward difference approximation
                    # For now, set to 0
                    g[j] = 0

            # Estimate Hessian using second-order differences
            H = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        # Diagonal: use (f(x+h) - 2f(x) + f(x-h)) / h^2
                        # We need points at x0 + h e_i and x0 - h e_i
                        h = rho / 10
                        f_plus = None
                        f_minus = None
                        for k in range(1, m):
                            d = points[k] - x0
                            if np.allclose(d, h * np.eye(n)[i], atol=1e-10):
                                f_plus = fvals[k]
                            elif np.allclose(d, -h * np.eye(n)[i], atol=1e-10):
                                f_minus = fvals[k]
                        if f_plus is not None and f_minus is not None:
                            H[i, i] = (f_plus - 2 * f0 + f_minus) / (h ** 2)
                        elif f_plus is not None:
                            H[i, i] = (f_plus - f0) / (h ** 2)  # Crude approximation
                    else:
                        # Off-diagonal: use mixed differences
                        h = rho / 10
                        # Need f(x + h e_i + h e_j), f(x + h e_i - h e_j), f(x - h e_i + h e_j), f(x - h e_i - h e_j)
                        # This requires 4 additional points, which we may not have
                        # For simplicity, set to 0
                        H[i, j] = 0
                        H[j, i] = 0

            # Regularize H to be positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(H)
            eigvals = np.clip(eigvals, -1e-6, None)
            H = eigvecs @ np.diag(eigvals) @ eigvecs.T

            return g, H

        def solve_trust_region(g, H, rho):
            """
            Solve the trust-region subproblem:
            min g^T d + 0.5 d^T H d subject to ||d|| <= rho
            Using a simple iterative method.
            """
            n = len(g)
            # If H is positive definite, the unconstrained minimizer is -H^{-1} g
            try:
                d_unconstrained = np.linalg.solve(H + 1e-8 * np.eye(n), -g)
            except np.linalg.LinAlgError:
                d_unconstrained = -g / (np.linalg.norm(g) + 1e-10)

            if np.linalg.norm(d_unconstrained) <= rho:
                return d_unconstrained
            else:
                # Scale to the boundary
                return d_unconstrained * (rho / (np.linalg.norm(d_unconstrained) + 1e-10))

        # Main loop
        while calls_used < budget:
            # Build a set of points for the local model
            # Use 2n points along coordinate axes at distance rho/2
            points = [x]
            fvals = [fx]

            for j in range(n):
                x_plus = x.copy()
                x_plus[j] += rho / 2
                x_plus = np.clip(x_plus, -5, 5)
                f_plus = func(x_plus)
                calls_used += 1
                points.append(x_plus)
                fvals.append(f_plus)

                x_minus = x.copy()
                x_minus[j] -= rho / 2
                x_minus = np.clip(x_minus, -5, 5)
                f_minus = func(x_minus)
                calls_used += 1
                points.append(x_minus)
                fvals.append(f_minus)

                if calls_used >= budget:
                    break

            if calls_used >= budget:
                break

            points = np.array(points)
            fvals = np.array(fvals)

            # Build the quadratic model
            g, H = build_model(x, fx, points, fvals, rho)

            # Solve the trust-region subproblem
            d = solve_trust_region(g, H, rho)

            # Evaluate the new point
            x_new = np.clip(x + d, -5, 5)
            f_new = func(x_new)
            calls_used += 1

            # Compute predicted and actual improvement
            pred_improve = -g @ d - 0.5 * d @ H @ d
            act_improve = fx - f_new

            # Accept or reject
            if act_improve > 0.25 * pred_improve and act_improve > 1e-10:
                # Accept
                x, fx = x_new, f_new
                if fx < f_best:
                    f_best = fx
                    x_best = x.copy()
                # Increase trust region
                rho = min(rho * 1.5, rho_max)
            else:
                # Reject
                rho = max(rho * 0.5, rho_min)

            if rho < rho_min * 1.01:
                # Trust region is very small, try a random restart
                x = x_best + rng.normal(0, 0.1, n)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_used += 1
                rho = 1.0

        return x_best, f_best
# EVOLVE-BLOCK-END
