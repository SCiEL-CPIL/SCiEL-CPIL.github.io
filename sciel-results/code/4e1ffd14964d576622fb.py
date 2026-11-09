"""Root finder for 3 fixed two-dimensional systems using Broyden's method with safeguards."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0


def _time_ok():
    return (time.time() - _START_TIME) < _DEADLINE


def _is_finite(arr):
    try:
        arr = np.asarray(arr, dtype=float)
        return arr.shape == (2,) and np.all(np.isfinite(arr))
    except Exception:
        return False


def _fd_jacobian_forward(F, x, h=1e-6):
    """Compute forward-difference Jacobian (3 function calls total: Fx + 2 perturbations)."""
    Fx = F(x)
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        J[:, j] = (F(x + e) - Fx) / h
    return J, Fx


def _fd_jacobian_central(F, x, h=1e-6):
    """Compute central-difference Jacobian (5 function calls total: Fx + 4 perturbations)."""
    Fx = F(x)
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        J[:, j] = (F(x + e) - F(x - e)) / (2 * h)
    return J, Fx


def _broyden(F, x0, max_iter=100):
    """
    Broyden's good method for 2D systems.
    Returns (x, n_calls, success).
    """
    x = np.array(x0, dtype=float)
    n_calls = 0

    # Initial Jacobian estimate via forward differences
    J, Fx = _fd_jacobian_forward(F, x)
    n_calls += 3  # F(x), F(x+e0), F(x+e1)

    if not _is_finite(Fx):
        return x, n_calls, False

    best_x = x.copy()
    best_norm = np.linalg.norm(Fx)

    for i in range(max_iter):
        if not _time_ok():
            break

        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x.copy()

        if norm_Fx <= 1e-12:
            return x, n_calls, True

        # Try to solve J dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse or gradient step
            try:
                dx = -np.linalg.pinv(J) @ Fx
            except Exception:
                break

        # Damping: ensure step doesn't go too far
        step_norm = np.linalg.norm(dx)
        if step_norm == 0 or not np.isfinite(step_norm):
            break

        # Limit step size to avoid divergence
        max_step = 10.0
        if step_norm > max_step:
            dx = dx * (max_step / step_norm)

        x_new = x + dx

        # Check if new point is finite
        if not _is_finite(x_new):
            break

        Fx_new = F(x_new)
        n_calls += 1

        if not _is_finite(Fx_new):
            break

        norm_Fx_new = np.linalg.norm(Fx_new)

        if norm_Fx_new < best_norm:
            best_norm = norm_Fx_new
            best_x = x_new.copy()

        # Broyden update: J_new = J + (s - J*dx) * y^T / (s^T)
        # where s = x_new - x, y = F(x_new) - F(x)
        s = x_new - x
        y = Fx_new - Fx
        s_norm_sq = np.dot(s, s)

        if s_norm_sq > 1e-20:
            J = J + np.outer(s - J @ s, y) / s_norm_sq

        x = x_new
        Fx = Fx_new

    return best_x, n_calls, best_norm <= 1e-8


def _newton_fd(F, x0, max_iter=40):
    """
    Newton's method with central-difference Jacobian.
    Returns (x, n_calls, success).
    """
    x = np.array(x0, dtype=float)
    n_calls = 0

    best_x = x.copy()
    best_norm = np.inf

    for i in range(max_iter):
        if not _time_ok():
            break

        # Central difference Jacobian: 5 calls (Fx + 4 perturbations)
        J, Fx = _fd_jacobian_central(F, x)
        n_calls += 5

        if not _is_finite(Fx):
            break

        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x.copy()

        if norm_Fx <= 1e-12:
            return x, n_calls, True

        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break

        step_norm = np.linalg.norm(dx)
        if step_norm == 0 or not np.isfinite(step_norm):
            break

        max_step = 10.0
        if step_norm > max_step:
            dx = dx * (max_step / step_norm)

        x_new = x + dx

        if not _is_finite(x_new):
            break

        Fx_new = F(x_new)
        n_calls += 1

        if not _is_finite(Fx_new):
            break

        norm_Fx_new = np.linalg.norm(Fx_new)
        if norm_Fx_new < best_norm:
            best_norm = norm_Fx_new
            best_x = x_new.copy()

        x = x_new
        Fx = Fx_new

    return best_x, n_calls, best_norm <= 1e-8


def _powell(F, x0, max_iter=200):
    """
    Powell's method (derivative-free) for 2D.
    Returns (x, n_calls, success).
    """
    x = np.array(x0, dtype=float)
    n_calls = 0

    def objective(x):
        nonlocal n_calls
        fx = F(x)
        n_calls += 1
        if not _is_finite(fx):
            return 1e10
        return np.dot(fx, fx)

    best_x = x.copy()
    best_val = objective(x)

    # Initial directions
    dirs = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]

    for i in range(max_iter):
        if not _time_ok():
            break

        x_old = x.copy()
        x_start = x.copy()

        for d in dirs:
            # Line search along direction d
            best_t = 0.0
            best_x_line = x.copy()
            best_val_line = objective(x)

            # Try a few steps
            for t in [0.1, 0.5, 1.0, -0.1, -0.5, -1.0, 0.01, 0.2, 2.0, -0.01, -0.2, -2.0]:
                x_try = x + t * d
                if not _is_finite(x_try):
                    continue
                val_try = objective(x_try)
                if val_try < best_val_line:
                    best_val_line = val_try
                    best_x_line = x_try
                    best_t = t

            x = best_x_line

        # Check convergence
        if np.linalg.norm(x - x_old) < 1e-12:
            break

        # Update directions
        new_dir = x - x_start
        new_dir_norm = np.linalg.norm(new_dir)
        if new_dir_norm < 1e-12:
            break

        new_dir = new_dir / new_dir_norm

        # Find the direction with the largest improvement
        vals = [objective(x_start + d * np.linalg.norm(new_dir)) for d in dirs]
        # Replace the worst direction
        worst_idx = np.argmax(vals)
        dirs[worst_idx] = new_dir

        if objective(x) < best_val:
            best_val = objective(x)
            best_x = x.copy()

    fx = F(best_x)
    n_calls += 1
    return best_x, n_calls, np.linalg.norm(fx) <= 1e-8


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method primarily, with Newton and Powell as fallbacks.
    """
    x0 = np.array(x0, dtype=float)

    # Track best result across all methods
    best_x = x0.copy()
    best_norm = np.inf

    # Method 1: Broyden's method (most efficient in calls)
    if _time_ok():
        x_b, calls_b, success_b = _broyden(F, x0, max_iter=80)
        if success_b and _is_finite(x_b):
            fx = F(x_b)
            n_calls_eval = 1  # This call is counted
            norm_b = np.linalg.norm(fx) if _is_finite(fx) else np.inf
            if norm_b < best_norm:
                best_norm = norm_b
                best_x = x_b.copy()
            if norm_b <= 1e-8:
                return x_b

    # Method 2: Newton with central differences
    if _time_ok():
        x_n, calls_n, success_n = _newton_fd(F, x0, max_iter=30)
        if _is_finite(x_n):
            fx = F(x_n)
            if _is_finite(fx):
                norm_n = np.linalg.norm(fx)
                if norm_n < best_norm:
                    best_norm = norm_n
                    best_x = x_n.copy()
                if norm_n <= 1e-8:
                    return x_n

    # Method 3: Powell's method (derivative-free)
    if _time_ok():
        x_p, calls_p, success_p = _powell(F, x0, max_iter=150)
        if _is_finite(x_p):
            fx = F(x_p)
            if _is_finite(fx):
                norm_p = np.linalg.norm(fx)
                if norm_p < best_norm:
                    best_norm = norm_p
                    best_x = x_p.copy()
                if norm_p <= 1e-8:
                    return x_p

    # Try Broyden from a slightly perturbed start if we haven't succeeded
    if _time_ok() and best_norm > 1e-8:
        for pert in [1e-3, 1e-2, 0.1]:
            x0_pert = x0 + pert * np.array([1.0, 1.0])
            x_b, calls_b, success_b = _broyden(F, x0_pert, max_iter=50)
            if _is_finite(x_b):
                fx = F(x_b)
                if _is_finite(fx):
                    norm_b = np.linalg.norm(fx)
                    if norm_b < best_norm:
                        best_norm = norm_b
                        best_x = x_b.copy()
                    if norm_b <= 1e-8:
                        return x_b

    return best_x
# EVOLVE-BLOCK-END
