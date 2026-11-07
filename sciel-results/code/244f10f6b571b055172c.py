"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-Free Trust-Region optimizer for BBOB f18.

    Uses diagonal Hessian estimation from sparse point evaluations to
    directly model local ill-conditioning, enabling larger effective steps
    along shallow eigen-directions compared to isotropic ES.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _estimate_gradient_hessian_diag(self, func, x, fx, h, calls_remaining, deadline):
        """Estimate gradient and diagonal Hessian using 2n+1 point evaluations.

        Uses central differences for gradient and forward differences for
        diagonal Hessian components. Returns (grad, hess_diag, calls_used).
        """
        dim = self.dim
        grad = np.zeros(dim)
        hess_diag = np.zeros(dim)
        used = 0

        # Estimate gradient with central differences
        for i in range(dim):
            if used >= calls_remaining or time.time() > deadline:
                return grad, hess_diag, used

            x_plus = x.copy()
            x_plus[i] = np.clip(x_plus[i] + h, -5.0, 5.0)
            x_minus = x.copy()
            x_minus[i] = np.clip(x_minus[i] - h, -5.0, 5.0)

            f_plus = func(x_plus)
            f_minus = func(x_minus)
            used += 2

            if used >= calls_remaining or time.time() > deadline:
                return grad, hess_diag, used

            grad[i] = (f_plus - f_minus) / (2.0 * h)

            # Estimate diagonal Hessian component
            hess_diag[i] = (f_plus - 2.0 * fx + f_minus) / (h * h)

        return grad, hess_diag, used

    def _dftr_step(self, func, x, fx, delta, h, rng, deadline, max_iter=20):
        """Perform DFTR local optimization with trust-region radius adaptation.

        Returns (x, fx, calls_used).
        """
        dim = self.dim
        used = 0
        best_x = x.copy()
        best_f = fx

        for _ in range(max_iter):
            if time.time() > deadline:
                break

            # Estimate gradient and Hessian diagonal
            grad, hess_diag, est_calls = self._estimate_gradient_hessian_diag(
                func, x, fx, h, budget - used, deadline
            )
            used += est_calls

            if est_calls == 0:
                break

            # Check if we're at a minimum (gradient nearly zero)
            if np.linalg.norm(grad) < 1e-10:
                break

            # Compute Newton step with diagonal Hessian
            # Avoid division by zero and clip to trust region
            hess_clipped = np.clip(hess_diag, 1e-6, None)
            step = -grad / hess_clipped

            # Scale step to trust region
            step_norm = np.linalg.norm(step)
            if step_norm > delta:
                step = step * (delta / step_norm)

            # Propose new point
            y = np.clip(x + step, -5.0, 5.0)
            fy = func(y)
            used += 1

            if fy < fx:
                # Accept step, adjust trust region
                rho = (fx - fy) / (np.dot(grad, step) + 0.5 * np.dot(step, hess_diag * step))
                x, fx = y, fy
                if fx < best_f:
                    best_f, best_x = fx, x.copy()

                if rho > 0.75:
                    delta = min(2.0 * delta, 4.0)  # increase trust region
                elif rho < 0.25:
                    delta *= 0.5  # decrease trust region
                else:
                    delta *= 0.8
            else:
                # Reject step, decrease trust region
                delta *= 0.5
                if delta < 1e-6:
                    break

            # Adapt finite difference step size
            if delta < 0.1:
                h = max(1e-4, delta * 0.1)
            elif delta > 1.0:
                h = min(0.5, delta * 0.1)

        return best_x, best_f, used

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(0)

        # ---- Phase 1: diverse multi-start exploration ----
        n_starts = 7
        phase1_calls = min(105, max(1, budget // 5))
        per_start = max(15, phase1_calls // n_starts)

        best_x = None
        best_f = np.inf
        calls = 0

        # Generate diverse starting points
        start_points = []
        for _ in range(n_starts):
            x = rng.uniform(-4.0, 4.0, dim)
            start_points.append(x)

        for x0 in start_points:
            if calls >= phase1_calls or time.time() > deadline:
                break

            x = x0.copy()
            fx = func(x)
            calls += 1

            if fx < best_f:
                best_f, best_x = fx, x.copy()

            # DFTR local optimization from this start
            remaining_for_this = min(per_start, budget - calls)
            if remaining_for_this > 0:
                x, fx, used = self._dftr_step(
                    func, x, fx, delta=2.0, h=0.1, rng=rng,
                    deadline=deadline, max_iter=10
                )
                calls += used
                if fx < best_f:
                    best_f, best_x = fx, x.copy()

        # ---- Phase 2: committed DFTR from best point ----
        remaining = max(0, budget - calls)
        if remaining > 0 and best_x is not None:
            x, fx, used = self._dftr_step(
                func, best_x, best_f, delta=1.5, h=0.05, rng=rng,
                deadline=deadline, max_iter=50
            )
            calls += used
            if fx < best_f:
                best_f, best_x = fx, x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
