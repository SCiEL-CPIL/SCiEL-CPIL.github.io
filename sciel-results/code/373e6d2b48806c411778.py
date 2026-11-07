"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-Free Trust-Region (DFTR) optimizer for BBOB f18."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill
        n = self.dim
        budget = self.budget

        # Initial point
        rng = np.random.default_rng(42)
        x = rng.uniform(-4, 4, n)
        fx = func(x)
        calls = 1
        best_x = x.copy()
        best_f = fx

        # Trust-region parameters
        delta = 2.0  # initial trust-region radius
        min_delta = 1e-8
        max_delta = 10.0

        # Build local quadratic model from directional derivatives
        # We use a set of directions to estimate gradient and Hessian
        # For n=5, we use 2*n+1 points (n+1 forward differences + n backward + center)
        # Actually, let's use a more efficient approach:
        # Use n forward differences to get gradient estimate, then use those to build a model

        # Strategy: At each trust-region center, estimate gradient using forward differences
        # Then take a Newton-like step using a diagonal Hessian approximation
        # This is simpler but effective for ill-conditioned problems

        # Better: Use a proper quadratic model from 2n+1 evaluations
        # But that's expensive. Let's use a gradient-based approach with adaptive step size.

        # Actually, let me implement a proper DFTR:
        # 1. At center x, evaluate f(x)
        # 2. Estimate gradient using n forward differences: g_i ≈ (f(x + h*e_i) - f(x)) / h
        # 3. Estimate diagonal Hessian: H_ii ≈ (f(x + h*e_i) - 2*f(x) + f(x - h*e_i)) / h^2
        # 4. Solve trust-region subproblem: min 0.5*x^T H x + g^T x s.t. ||x|| <= delta
        # 5. Take step, evaluate, update trust region

        # For efficiency, let's use a simpler but effective approach:
        # Gradient estimation + adaptive Newton step with diagonal Hessian

        h = 1e-4  # finite difference step
        sigma = 1.0  # step size multiplier

        for iteration in range(budget - 1):
            if time.time() > deadline - 5.0:
                break
            if budget - calls <= 0:
                break

            # Estimate gradient using forward differences
            grad = np.zeros(n)
            hessian_diag = np.zeros(n)

            for i in range(n):
                if time.time() > deadline - 5.0:
                    break
                if budget - calls <= 2:
                    break

                # Forward difference for gradient
                x_plus = x.copy()
                x_plus[i] += h
                x_plus = np.clip(x_plus, -5, 5)
                fp = func(x_plus)
                calls += 1

                # Check if we're at the boundary
                if x_plus[i] == x[i]:
                    # Use backward difference instead
                    x_minus = x.copy()
                    x_minus[i] -= h
                    x_minus = np.clip(x_minus, -5, 5)
                    fm = func(x_minus)
                    calls += 1
                    grad[i] = (fx - fm) / h
                    # For Hessian, use central difference if possible
                    if budget - calls > 0 and time.time() < deadline - 5.0:
                        x_plus2 = x.copy()
                        x_plus2[i] += 2 * h
                        x_plus2 = np.clip(x_plus2, -5, 5)
                        fp2 = func(x_plus2)
                        calls += 1
                        hessian_diag[i] = max((fp2 - 2 * fx + fm) / (4 * h * h), 1e-10)
                else:
                    grad[i] = (fp - fx) / h
                    # For Hessian, use central difference
                    if budget - calls > 0 and time.time() < deadline - 5.0:
                        x_minus = x.copy()
                        x_minus[i] -= h
                        x_minus = np.clip(x_minus, -5, 5)
                        fm = func(x_minus)
                        calls += 1
                        hessian_diag[i] = max((fp - 2 * fx + fm) / (h * h), 1e-10)
                    else:
                        hessian_diag[i] = max(abs(grad[i]) / h, 1e-10)

            if budget - calls <= 0:
                break

            # Ensure Hessian diagonal is positive
            hessian_diag = np.maximum(hessian_diag, 1e-10)

            # Newton step: x_new = x - H^{-1} g
            step = -grad / hessian_diag

            # Scale step to trust region
            step_norm = np.linalg.norm(step)
            if step_norm > delta:
                step = step * (delta / step_norm)
            elif step_norm < delta * 0.1:
                step = step * (delta * 0.1 / max(step_norm, 1e-15))

            # Take step
            x_new = np.clip(x + step, -5, 5)
            f_new = func(x_new)
            calls += 1

            # Update best
            if f_new < best_f:
                best_x = x_new.copy()
                best_f = f_new

            # Update trust region based on actual vs predicted reduction
            predicted_reduction = 0.5 * np.dot(step, hessian_diag * step) + np.dot(grad, step)
            actual_reduction = fx - f_new

            if predicted_reduction > 0:
                rho = actual_reduction / predicted_reduction
            else:
                rho = -1.0

            if rho > 0.75:
                delta = min(max_delta, delta * 2.0)
            elif rho > 0.25:
                delta = min(max_delta, delta * 1.5)
            elif rho > 0:
                delta = max(min_delta, delta * 0.5)
            else:
                delta = max(min_delta, delta * 0.25)

            # Accept the step if it improves
            if f_new < fx:
                x = x_new
                fx = f_new
            else:
                # Try a smaller step in the same direction
                if budget - calls > 0 and time.time() < deadline - 5.0:
                    x_half = np.clip(x + 0.5 * step, -5, 5)
                    f_half = func(x_half)
                    calls += 1
                    if f_half < fx:
                        x = x_half
                        fx = f_half
                        if f_half < best_f:
                            best_x = x_half.copy()
                            best_f = f_half

        return best_x, best_f
# EVOLVE-BLOCK-END
