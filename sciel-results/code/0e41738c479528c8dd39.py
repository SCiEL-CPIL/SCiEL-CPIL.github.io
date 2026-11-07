# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8 using Broyden's quasi-Newton method
    with adaptive damping and a time guard.
    """
    start_time = time.time()
    TIME_LIMIT = 115.0  # Safety margin before 120s kill

    def time_ok():
        return (time.time() - start_time) < TIME_LIMIT

    x = np.array(x0, dtype=float).copy()
    
    # Check initial point
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,):
            Fx = Fx.reshape(2)
        if not np.all(np.isfinite(Fx)):
            # If initial point is bad, try to find a better one via local search
            x = _local_search(F, x, time_ok, start_time)
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,):
                Fx = Fx.reshape(2)
            if not np.all(np.isfinite(Fx)):
                return x
        if np.linalg.norm(Fx) <= 1e-8:
            return x
    except Exception:
        # If F fails at x0, try a slightly perturbed point
        x = x + np.array([1e-4, 1e-4])
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,):
                Fx = Fx.reshape(2)
            if not np.all(np.isfinite(Fx)):
                return x
            if np.linalg.norm(Fx) <= 1e-8:
                return x
        except Exception:
            return x

    # Initialize Jacobian estimate using finite differences (3 calls: x, x+e0, x+e1)
    # We already have F(x). Need F(x+e0) and F(x+e1).
    J = np.zeros((2, 2))
    h = 1e-6
    try:
        x0_pert = x.copy()
        x0_pert[0] += h
        Fx0 = F(x0_pert)
        Fx0 = np.asarray(Fx0, dtype=float)
        if Fx0.shape != (2,):
            Fx0 = Fx0.reshape(2)
        J[:, 0] = (Fx0 - Fx) / h

        x1_pert = x.copy()
        x1_pert[1] += h
        Fx1 = F(x1_pert)
        Fx1 = np.asarray(Fx1, dtype=float)
        if Fx1.shape != (2,):
            Fx1 = Fx1.reshape(2)
        J[:, 1] = (Fx1 - Fx) / h
    except Exception:
        # If finite diff fails, use identity
        J = np.eye(2)

    # Broyden's method (Good variant)
    for _ in range(450):  # Max iterations, but we have ~497 calls left
        if not time_ok():
            break

        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, use least squares
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break

        # Damping: find alpha in (0, 1] such that ||F(x + alpha*dx)|| < ||F(x)||
        # Use a simple backtracking line search
        alpha = 1.0
        Fx_norm = np.linalg.norm(Fx)
        x_new = x + alpha * dx
        try:
            Fx_new = F(x_new)
            Fx_new = np.asarray(Fx_new, dtype=float)
            if Fx_new.shape != (2,):
                Fx_new = Fx_new.reshape(2)
            Fx_new_norm = np.linalg.norm(Fx_new)
        except Exception:
            # If F fails, reduce alpha
            alpha *= 0.5
            x_new = x + alpha * dx
            try:
                Fx_new = F(x_new)
                Fx_new = np.asarray(Fx_new, dtype=float)
                if Fx_new.shape != (2,):
                    Fx_new = Fx_new.reshape(2)
                Fx_new_norm = np.linalg.norm(Fx_new)
            except Exception:
                break

        if not np.all(np.isfinite(Fx_new)):
            alpha *= 0.5
            x_new = x + alpha * dx
            try:
                Fx_new = F(x_new)
                Fx_new = np.asarray(Fx_new, dtype=float)
                if Fx_new.shape != (2,):
                    Fx_new = Fx_new.reshape(2)
                Fx_new_norm = np.linalg.norm(Fx_new)
            except Exception:
                break
            if not np.all(np.isfinite(Fx_new)):
                break

        # Check convergence
        if Fx_new_norm <= 1e-8:
            return x_new

        # Backtracking if needed
        max_backtracks = 10
        for _ in range(max_backtracks):
            if Fx_new_norm < Fx_norm:
                break
            alpha *= 0.5
            x_new = x + alpha * dx
            try:
                Fx_new = F(x_new)
                Fx_new = np.asarray(Fx_new, dtype=float)
                if Fx_new.shape != (2,):
                    Fx_new = Fx_new.reshape(2)
                Fx_new_norm = np.linalg.norm(Fx_new)
            except Exception:
                break
            if not np.all(np.isfinite(Fx_new)):
                break

        if not np.all(np.isfinite(Fx_new)):
            break

        # Update Broyden's Jacobian (Good variant)
        s = x_new - x
        y = Fx_new - Fx
        if np.linalg.norm(s) > 1e-15:
            J = J + (np.outer(y, s) - np.outer(J @ s, s)) / (s @ s)

        x = x_new
        Fx = Fx_new

    # If we didn't converge, return the best point we have
    return x


def _local_search(F, x, time_ok, start_time):
    """Simple local search to find a point where F is finite."""
    x = np.array(x, dtype=float)
    # Try a few nearby points
    for dx in [1e-2, 1e-1, 1.0, 10.0]:
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            if not time_ok():
                return x
            step = dx * np.array([np.cos(angle), np.sin(angle)])
            x_try = x + step
            try:
                Fx = F(x_try)
                Fx = np.asarray(Fx, dtype=float)
                if Fx.shape != (2,):
                    Fx = Fx.reshape(2)
                if np.all(np.isfinite(Fx)):
                    x = x_try
                    return x
            except Exception:
                continue
    return x
# EVOLVE-BLOCK-END
