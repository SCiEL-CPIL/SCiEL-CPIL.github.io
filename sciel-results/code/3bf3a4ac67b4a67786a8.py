"""Root finder for 3 fixed 2D systems using Broyden's method with robust fallbacks."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def time_left():
        return deadline - time.time()

    x = np.array(x0, dtype=float, copy=True)
    best_x = x.copy()
    best_norm = np.inf

    def safe_call(F, x):
        try:
            result = F(x)
            result = np.asarray(result, dtype=float)
            if result.shape != (2,) or not np.all(np.isfinite(result)):
                return None
            return result
        except Exception:
            return None

    # Initial evaluation
    Fx = safe_call(F, x)
    if Fx is None:
        return x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    best_norm = norm_Fx
    best_x = x.copy()

    # Compute initial Jacobian using central differences
    def compute_jacobian(x, Fx):
        J = np.zeros((2, 2))
        h = 1e-6
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = safe_call(F, x + e)
            Fx_minus = safe_call(F, x - e)
            if Fx_plus is not None and Fx_minus is not None:
                J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
            else:
                # Fallback to forward difference
                Fx_plus = safe_call(F, x + e)
                if Fx_plus is not None:
                    J[:, j] = (Fx_plus - Fx) / h
                else:
                    J[:, j] = 1.0  # Default
        return J

    J = compute_jacobian(x, Fx)

    # Broyden's method
    for iteration in range(100):
        if time_left() < 5.0:
            break

        # Check convergence
        if norm_Fx <= 1e-8:
            return x

        # Compute step
        try:
            step = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse
            try:
                step = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break

        # Damping: ensure step is not too large
        step_norm = np.linalg.norm(step)
        if step_norm > 1.0:
            step = step / step_norm

        x_new = x + step

        # Evaluate at new point
        Fx_new = safe_call(F, x_new)
        if Fx_new is None:
            # Try smaller step
            x_new = x + 0.5 * step
            Fx_new = safe_call(F, x_new)
            if Fx_new is None:
                break

        norm_Fx_new = np.linalg.norm(Fx_new)

        # Update Broyden's Jacobian
        delta_x = x_new - x
        delta_F = Fx_new - Fx
        if np.linalg.norm(delta_x) > 1e-15:
            J = J + np.outer(delta_F - J @ delta_x, delta_x) / np.dot(delta_x, delta_x)

        x = x_new
        Fx = Fx_new
        norm_Fx = norm_Fx_new

        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x.copy()

    # If Broyden didn't converge, try a few Newton steps with fresh Jacobian
    if best_norm > 1e-8 and time_left() > 5.0:
        for _ in range(20):
            if time_left() < 5.0:
                break
            Fx = safe_call(F, best_x)
            if Fx is None:
                break
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx <= 1e-8:
                return best_x
            J = compute_jacobian(best_x, Fx)
            try:
                step = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
            step_norm = np.linalg.norm(step)
            if step_norm > 1.0:
                step = step / step_norm
            x_new = best_x + step
            Fx_new = safe_call(F, x_new)
            if Fx_new is None:
                break
            norm_Fx_new = np.linalg.norm(Fx_new)
            if norm_Fx_new < best_norm:
                best_norm = norm_Fx_new
                best_x = x_new.copy()

    return best_x
# EVOLVE-BLOCK-END
