"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill


def _time_ok():
    return (time.time() - _start_time) < _DEADLINE


def _safe_F(F, x):
    """Call F and return (Fx, fx_norm) or (None, inf) on failure."""
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
            return None, np.inf
        return Fx, np.linalg.norm(Fx)
    except Exception:
        return None, np.inf


def _compute_jacobian_fd(F, x, Fx, h=1e-6):
    """Compute Jacobian using forward differences. Returns J or None on failure."""
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fp, _ = _safe_F(F, x + e)
        if Fp is None:
            return None
        J[:, j] = (Fp - Fx) / h
    if not np.all(np.isfinite(J)):
        return None
    return J


def _broyden_lm(F, x, Fx, fx_norm, J, max_iters=40):
    """
    Broyden's Good Method with adaptive Levenberg-Marquardt damping.
    Solves (J^T J + lambda*I) s = -J^T F for the step.
    lambda is adjusted based on whether the step reduces the residual.
    """
    x = x.copy()
    Fx = Fx.copy()
    fx_norm = fx_norm
    J = J.copy()

    best_x = x.copy()
    best_norm = fx_norm

    # Initial lambda based on diagonal of J^T J
    JtJ = J.T @ J
    diag = np.diag(JtJ)
    lambda_lm = 1e-3 * np.mean(diag) if np.all(diag > 0) else 1e-3

    for _ in range(max_iters):
        if not _time_ok():
            break

        if fx_norm <= 1e-10:
            return x, Fx, fx_norm, best_x, best_norm

        # Solve (J^T J + lambda*I) s = -J^T F
        A = JtJ + lambda_lm * np.eye(2)
        b = -J.T @ Fx

        try:
            s = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # If singular, increase lambda and retry
            lambda_lm *= 10
            A = JtJ + lambda_lm * np.eye(2)
            try:
                s = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                break

        if not np.all(np.isfinite(s)):
            break

        # Try the step
        x_new = x + s
        Fx_new, fx_new_norm = _safe_F(F, x_new)

        if Fx_new is None:
            lambda_lm *= 10
            continue

        # Check reduction
        if fx_new_norm < fx_norm:
            # Success: update Broyden, decrease lambda
            y = Fx_new - Fx
            s_actual = x_new - x
            denom = s_actual @ s_actual
            if denom > 1e-15:
                J = J + np.outer(y - J @ s_actual, s_actual) / denom
            JtJ = J.T @ J

            x = x_new
            Fx = Fx_new
            fx_norm = fx_new_norm

            if fx_norm < best_norm:
                best_norm = fx_norm
                best_x = x.copy()

            lambda_lm *= 0.5
            if lambda_lm < 1e-12:
                lambda_lm = 1e-12
        else:
            # Failure: increase lambda
            lambda_lm *= 10
            if lambda_lm > 1e12:
                break

    return x, Fx, fx_norm, best_x, best_norm


def _fallback_newton(F, x, Fx, best_x, best_norm):
    """Fallback to standard Newton with forward-difference Jacobian."""
    for _ in range(20):
        if not _time_ok():
            break

        J = _compute_jacobian_fd(F, x, Fx, h=1e-5)
        if J is None:
            return best_x

        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            return best_x

        if not np.all(np.isfinite(s)):
            return best_x

        alpha = 1.0
        converged = False
        for _ in range(20):
            x_new = x + alpha * s
            Fx_new, fx_new_norm = _safe_F(F, x_new)
            if Fx_new is None:
                alpha *= 0.5
                continue

            if fx_new_norm < np.linalg.norm(Fx):
                x = x_new
                Fx = Fx_new
                if fx_new_norm < best_norm:
                    best_norm = fx_new_norm
                    best_x = x.copy()
                if fx_new_norm <= 1e-10:
                    return x
                converged = True
                break
            alpha *= 0.5

        if not converged:
            break

    return best_x


def find_root_system(F, x0):
    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_norm = np.inf

    # Initial evaluation
    Fx, fx_norm = _safe_F(F, x)
    if Fx is None:
        return best_x

    if fx_norm < best_norm:
        best_norm = fx_norm
        best_x = x.copy()

    if fx_norm <= 1e-10:
        return x

    # Initialize Jacobian with forward differences
    J = _compute_jacobian_fd(F, x, Fx, h=1e-6)
    if J is None:
        return _fallback_newton(F, x, Fx, best_x, best_norm)

    # Primary: Broyden with LM damping
    x, Fx, fx_norm, best_x_lm, best_norm_lm = _broyden_lm(F, x, Fx, fx_norm, J, max_iters=40)

    # Track best
    if best_norm_lm < best_norm:
        best_norm = best_norm_lm
        best_x = best_x_lm

    if best_norm <= 1e-10:
        return best_x

    # If not converged, try a few Newton steps with fresh Jacobians (polish)
    # This handles cases where Broyden approximation drifted
    if _time_ok():
        x_curr = best_x.copy()
        Fx_curr, fx_curr_norm = _safe_F(F, x_curr)
        if Fx_curr is not None and fx_curr_norm < 1e-4:
            # Only polish if we are reasonably close
            for _ in range(5):
                if not _time_ok():
                    break
                J = _compute_jacobian_fd(F, x_curr, Fx_curr, h=1e-7)
                if J is None:
                    break
                try:
                    s = np.linalg.solve(J, -Fx_curr)
                except np.linalg.LinAlgError:
                    break
                if not np.all(np.isfinite(s)):
                    break

                x_new = x_curr + s
                Fx_new, fx_new_norm = _safe_F(F, x_new)
                if Fx_new is None:
                    break
                if fx_new_norm < fx_curr_norm:
                    x_curr = x_new
                    Fx_curr = Fx_new
                    fx_curr_norm = fx_new_norm
                    if fx_curr_norm < best_norm:
                        best_norm = fx_curr_norm
                        best_x = x_curr.copy()
                    if fx_curr_norm <= 1e-10:
                        return best_x
                else:
                    break

    return best_x

# EVOLVE-BLOCK-END
