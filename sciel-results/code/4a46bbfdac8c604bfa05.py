"""Root finder for 3 fixed 2D systems using Broyden's method with robust restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def time_left():
        return deadline - time.time()

    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_norm = np.inf

    # Initial function evaluation
    try:
        Fx = F(x)
    except Exception:
        return best_x

    if not np.all(np.isfinite(Fx)):
        return best_x

    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx < best_norm:
        best_norm = norm_Fx
        best_x = x.copy()

    if norm_Fx <= 1e-8:
        return best_x

    # Compute initial Jacobian using finite differences
    def compute_jacobian(x, Fx):
        J = np.zeros((2, 2))
        h = 1e-7
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            try:
                Fx_e = F(x + e)
            except Exception:
                # Fallback to central difference if forward fails
                e[j] = -h
                try:
                    Fx_m = F(x + e)
                    J[:, j] = (Fx - Fx_m) / h
                except Exception:
                    J[:, j] = 1.0
            else:
                J[:, j] = (Fx_e - Fx) / h
        return J

    J = compute_jacobian(x, Fx)

    # Broyden's method
    max_iter = 100
    for _ in range(max_iter):
        if time_left() < 5.0:
            break

        # Try to solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If J is singular, try least squares
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break

        # Check for NaN or Inf in step
        if not np.all(np.isfinite(dx)):
            break

        # Damping if step is too large
        step_norm = np.linalg.norm(dx)
        if step_norm > 1.0:
            dx = dx / step_norm * 1.0

        x_new = x + dx

        # Check if x_new is finite
        if not np.all(np.isfinite(x_new)):
            break

        try:
            Fx_new = F(x_new)
        except Exception:
            break

        if not np.all(np.isfinite(Fx_new)):
            break

        norm_Fx_new = np.linalg.norm(Fx_new)

        # Update best
        if norm_Fx_new < best_norm:
            best_norm = norm_Fx_new
            best_x = x_new.copy()

        if norm_Fx_new <= 1e-8:
            return best_x

        # Broyden update
        delta_x = x_new - x
        delta_F = Fx_new - Fx
        try:
            # Broyden rank-1 update: J_new = J + (delta_F - J * delta_x) * delta_x^T / (delta_x^T * delta_x)
            denom = np.dot(delta_x, delta_x)
            if denom > 1e-15:
                J = J + (delta_F - J @ delta_x) * np.outer(delta_x, delta_x) / denom
        except Exception:
            pass

        x = x_new
        Fx = Fx_new

    return best_x
# EVOLVE-BLOCK-END
