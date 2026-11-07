"""Derivative-free trust-region BFGS optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Derivative-free Trust-Region BFGS (DF-TBFGS):
    - Uses finite-difference gradients to build a local quadratic model.
    - Maintains a BFGS Hessian approximation for efficient step computation.
    - Trust-region controls step size based on model accuracy.
    - Rapid local convergence leveraging f14 smoothness.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        d = self.dim
        max_calls = self.budget

        best_x = None
        best_f = float('inf')
        calls_used = 0

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= max_calls:
                return None
            if time.time() > deadline - 2:
                return None
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Initial sampling - focused on likely basin locations
        # f14 has different powers per dimension; minimizer is in [-4,4]^5
        initial_points = [np.zeros(d)]
        # Probe along axes with moderate offsets
        for i in range(d):
            for sign in [-1, 1]:
                p = np.zeros(d)
                p[i] = sign * 2.0
                initial_points.append(p)
        # A few random points for diversity
        for _ in range(3):
            initial_points.append(rng.uniform(-3, 3, d))

        for s in initial_points:
            fx = try_eval(s)
            if fx is None:
                break

        if calls_used >= max_calls or time.time() > deadline - 5:
            return best_x, best_f

        # Phase 2: DF-TBFGS trust-region optimization
        x0 = best_x.copy()
        fx0 = best_f
        trust_radius = 2.0
        H_bfgs = np.eye(d)  # BFGS Hessian approximation
        iter_count = 0

        # Adaptive finite-difference step
        def compute_gradient(x, fd_step=None):
            """Compute gradient via central differences with adaptive step."""
            nonlocal calls_used
            if fd_step is None:
                # Adaptive step based on trust radius
                fd_step = max(trust_radius * 0.1, 1e-5)
                fd_step = min(fd_step, 0.5)

            grad = np.zeros(d)
            for i in range(d):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += fd_step
                x_minus[i] -= fd_step

                fx_plus = try_eval(x_plus)
                if fx_plus is None:
                    return None
                fx_minus = try_eval(x_minus)
                if fx_minus is None:
                    return None

                grad[i] = (fx_plus - fx_minus) / (2 * fd_step)

            return grad

        # Main trust-region loop
        no_improve = 0
        max_no_improve = 15

        while calls_used < max_calls and time.time() < deadline - 10:
            # Compute gradient at current point
            grad = compute_gradient(x0)
            if grad is None:
                break

            grad_norm = np.linalg.norm(grad)

            # Convergence check
            if grad_norm < 1e-10:
                break

            # Compute BFGS step: s = -H_bfgs @ grad
            s = -H_bfgs @ grad
            s_norm = np.linalg.norm(s)

            # If BFGS step is too small or too large, fall back to steepest descent
            if s_norm < 1e-8 or s_norm > trust_radius * 10:
                s = -grad / grad_norm * trust_radius
                s_norm = trust_radius

            # Clip step to trust region
            if s_norm > trust_radius:
                s = s * (trust_radius / s_norm)

            x_new = x0 + s
            x_new = np.clip(x_new, -5, 5)

            fx_new = try_eval(x_new)
            if fx_new is None:
                break

            # Compute actual reduction
            actual_reduction = fx0 - fx_new

            # Compute predicted reduction using quadratic model
            # Use solve instead of inv for stability
            try:
                H_solve = H_bfgs + 1e-8 * np.eye(d)
                # Predicted reduction: -grad^T s - 0.5 s^T inv(H) s
                # We can compute s^T inv(H) s via solve
                temp = np.linalg.solve(H_solve, s)
                predicted_reduction = -(grad @ s) - 0.5 * (s @ temp)
            except np.linalg.LinAlgError:
                predicted_reduction = -(grad @ s)

            # Compute rho = actual / predicted
            if predicted_reduction > 1e-12:
                rho = actual_reduction / predicted_reduction
            else:
                rho = 1.0 if actual_reduction > 0 else -1.0

            # Update trust region based on rho
            if rho > 0.75:
                trust_radius = min(trust_radius * 1.5, 4.0)
            elif rho > 0.25:
                pass
            else:
                trust_radius = max(trust_radius * 0.5, 0.01)

            # Update BFGS Hessian approximation
            if actual_reduction > 1e-12:
                # Compute new gradient
                grad_new = compute_gradient(x_new)
                if grad_new is not None:
                    y = grad_new - grad
                    s_actual = x_new - x0
                    sy = s_actual @ y

                    if sy > 1e-10:
                        # BFGS update
                        Hy = H_bfgs @ y
                        Hy_norm = np.linalg.norm(Hy)
                        if Hy_norm > 1e-10:
                            H_bfgs = H_bfgs + (np.outer(s_actual, s_actual)) / sy - (np.outer(Hy, Hy)) / (y @ Hy)

                    # Accept the step
                    x0 = x_new
                    fx0 = fx_new
                    no_improve = 0
                    iter_count += 1

                    # Periodically regularize H_bfgs to prevent ill-conditioning
                    if iter_count % 8 == 0:
                        eigvals, eigvecs = np.linalg.eigh(H_bfgs)
                        eigvals = np.clip(eigvals, 1e-6, 1e6)
                        H_bfgs = eigvecs @ np.diag(eigvals) @ eigvecs.T

            else:
                # No improvement - shrink and try different direction
                no_improve += 1
                trust_radius = max(trust_radius * 0.7, 0.01)

                if no_improve >= max_no_improve:
                    # Restart with a random perturbation
                    jump = rng.standard_normal(d) * 0.5
                    x_try = np.clip(x0 + jump, -5, 5)
                    fx_try = try_eval(x_try)
                    if fx_try is not None and fx_try < fx0:
                        x0 = x_try
                        fx0 = fx_try
                        no_improve = 0
                        trust_radius = 1.0
                        H_bfgs = np.eye(d)
                    else:
                        # Try a larger jump
                        jump = rng.standard_normal(d) * 1.5
                        x_try = np.clip(x0 + jump, -5, 5)
                        fx_try = try_eval(x_try)
                        if fx_try is not None and fx_try < fx0:
                            x0 = x_try
                            fx0 = fx_try
                            no_improve = 0
                            trust_radius = 1.0
                            H_bfgs = np.eye(d)
                        else:
                            no_improve = 0
                            trust_radius = max(trust_radius * 0.5, 0.005)

        # Phase 3: Final high-precision polish with proper line search
        if calls_used < max_calls - 20 and time.time() < deadline - 5 and best_x is not None:
            x_polish = best_x.copy()
            f_polish = best_f
            H_polish = np.eye(d)
            fd_small = 1e-6
            polish_iter = 0

            for _ in range(30):
                if calls_used >= max_calls or time.time() > deadline - 3:
                    break

                # Compute gradient with small step
                grad_p = np.zeros(d)
                for i in range(d):
                    xp = x_polish.copy()
                    xm = x_polish.copy()
                    xp[i] += fd_small
                    xm[i] -= fd_small
                    fxp = try_eval(xp)
                    if fxp is None:
                        break
                    fxm = try_eval(xm)
                    if fxm is None:
                        break
                    grad_p[i] = (fxp - fxm) / (2 * fd_small)

                if calls_used >= max_calls or time.time() > deadline - 3:
                    break

                gn = np.linalg.norm(grad_p)
                if gn < 1e-12:
                    break

                # Newton-like step
                try:
                    step_p = -np.linalg.solve(H_polish + 1e-8 * np.eye(d), grad_p)
                except np.linalg.LinAlgError:
                    step_p = -grad_p / gn

                sp_norm = np.linalg.norm(step_p)
                if sp_norm > 0.1:
                    step_p = step_p * (0.1 / sp_norm)

                # Line search: try full step, then halve
                alpha = 1.0
                improved = False
                for _ in range(5):
                    x_trial = x_polish + alpha * step_p
                    x_trial = np.clip(x_trial, -5, 5)
                    fx_trial = try_eval(x_trial)
                    if fx_trial is None:
                        break
                    if fx_trial < f_polish:
                        # Update BFGS with line search result
                        grad_new_p = np.zeros(d)
                        for i in range(d):
                            xp = x_trial.copy()
                            xm = x_trial.copy()
                            xp[i] += fd_small
                            xm[i] -= fd_small
                            fxp = try_eval(xp)
                            if fxp is None:
                                break
                            fxm = try_eval(xm)
                            if fxm is None:
                                break
                            grad_new_p[i] = (fxp - fxm) / (2 * fd_small)

                        if calls_used >= max_calls or time.time() > deadline - 3:
                            break

                        y_p = grad_new_p - grad_p
                        s_p = x_trial - x_polish
                        sy_p = s_p @ y_p
                        if sy_p > 1e-10:
                            Hy_p = H_polish @ y_p
                            Hy_p_norm = np.linalg.norm(Hy_p)
                            if Hy_p_norm > 1e-10:
                                H_polish = H_polish + (np.outer(s_p, s_p)) / sy_p - (np.outer(Hy_p, Hy_p)) / (y_p @ Hy_p)

                        x_polish = x_trial
                        f_polish = fx_trial
                        improved = True
                        polish_iter += 1
                        break
                    else:
                        alpha *= 0.5

                if not improved:
                    break

            if f_polish < best_f:
                best_f = f_polish
                best_x = x_polish.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
