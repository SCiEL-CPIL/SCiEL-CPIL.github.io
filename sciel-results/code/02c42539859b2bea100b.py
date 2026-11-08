"""Root finder for 3 fixed two-dimensional systems using Broyden's method."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: kill program at 150s wall-clock, use 145s as safety margin
_START_TIME = time.time()
_DEADLINE = 145.0


def _time_ok():
    return (time.time() - _START_TIME) < _DEADLINE


def _is_finite(arr):
    try:
        return arr.shape == (2,) and np.all(np.isfinite(arr))
    except Exception:
        return False


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8 using Broyden's method.
    Broyden's method uses only 1 function call per iteration after initial setup,
    compared to 3 calls per iteration for Newton with finite-difference Jacobian.
    """
    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_norm = np.inf

    # Initial function evaluation
    try:
        Fx = F(x)
    except Exception:
        return best_x

    Fx = np.asarray(Fx, dtype=float)
    if not _is_finite(Fx):
        return best_x

    fx_norm = np.linalg.norm(Fx)
    if fx_norm <= 1e-8:
        return x

    best_x = x.copy()
    best_norm = fx_norm

    # Initialize Jacobian with finite differences (3 calls: F(x), F(x+e0), F(x+e1))
    # Actually we already have F(x), so we need 2 more calls
    J = np.zeros((2, 2))
    h = 1e-7
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        try:
            Fx_e = F(x + e)
        except Exception:
            # If we can't compute the perturbation, fall back to identity
            J = np.eye(2)
            break
        Fx_e = np.asarray(Fx_e, dtype=float)
        if not _is_finite(Fx_e):
            J = np.eye(2)
            break
        J[:, j] = (Fx_e - Fx) / h
    else:
        # Successfully computed Jacobian
        pass

    # Broyden's method iterations
    # Each iteration: 1 function call
    max_iter = 80
    for it in range(max_iter):
        if not _time_ok():
            break

        # Check if we're already good enough
        if best_norm <= 1e-8:
            break

        # Compute Newton step using current Jacobian approximation
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If J is singular, try adding a small regularization
            try:
                dx = np.linalg.solve(J + 1e-10 * np.eye(2), -Fx)
            except np.linalg.LinAlgError:
                break

        # Check if step is reasonable
        if not np.all(np.isfinite(dx)):
            break

        # Line search: try full step first, then damped steps
        found_step = False
        for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625]:
            x_new = x + alpha * dx
            if not np.all(np.isfinite(x_new)):
                continue

            try:
                Fx_new = F(x_new)
            except Exception:
                continue

            Fx_new = np.asarray(Fx_new, dtype=float)
            if not _is_finite(Fx_new):
                continue

            fx_norm_new = np.linalg.norm(Fx_new)

            if fx_norm_new <= 1e-8:
                x = x_new
                best_x = x.copy()
                best_norm = fx_norm_new
                return x

            # Accept if we made progress (reduced norm)
            if fx_norm_new < best_norm * 0.9 or fx_norm_new < best_norm:
                # Update Jacobian using Broyden's rank-1 update
                # J_new = J + (Fx_new - Fx - J*dx) * dx^T / (dx^T * dx)
                if np.dot(dx, dx) > 1e-30:
                    delta_F = Fx_new - Fx
                    delta_J = (delta_F - J @ dx) * dx / np.dot(dx, dx)
                    J = J + delta_J

                x = x_new
                Fx = Fx_new
                best_x = x.copy()
                best_norm = fx_norm_new
                found_step = True
                break

        if not found_step:
            # No acceptable step found, try a smaller random perturbation
            # to escape potential stagnation
            perturbation = np.random.uniform(-0.1, 0.1, size=2)
            x_new = x + perturbation
            try:
                Fx_new = F(x_new)
            except Exception:
                break
            Fx_new = np.asarray(Fx_new, dtype=float)
            if not _is_finite(Fx_new):
                break
            fx_norm_new = np.linalg.norm(Fx_new)
            if fx_norm_new < best_norm:
                # Update Jacobian
                if np.dot(perturbation, perturbation) > 1e-30:
                    delta_F = Fx_new - Fx
                    delta_J = (delta_F - J @ perturbation) * perturbation / np.dot(perturbation, perturbation)
                    J = J + delta_J
                x = x_new
                Fx = Fx_new
                best_x = x.copy()
                best_norm = fx_norm_new
            else:
                # No progress, break
                break

        # Early termination if converged
        if best_norm <= 1e-8:
            break

    return best_x
# EVOLVE-BLOCK-END
