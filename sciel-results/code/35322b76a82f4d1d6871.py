"""Root finder for smooth polynomial equations with call-budget awareness."""
import time
import math

_DEADLINE_START = time.time()
_WALL_LIMIT = 115.0  # Safety margin before 120s kill
_SAFETY_MARGIN = 5.0


def _time_left():
    return _WALL_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using the Secant method.
    The Secant method uses 1 function call per iteration (after initial setup),
    which is more efficient than Newton-FD (2 calls/iteration).
    """
    tol = 1e-12  # Tighter tolerance to ensure |f(x)| <= 1e-8
    max_iter = 100
    
    x_prev = float(x0)
    try:
        f_prev = f(x_prev)
    except Exception:
        return x_prev

    if not _is_finite(f_prev):
        return x_prev

    best_x = x_prev
    best_f = abs(f_prev)

    if best_f <= tol:
        return x_prev

    # Initial step for Secant method
    h = 1e-4 * max(1.0, abs(x_prev))
    x_curr = x_prev + h
    try:
        f_curr = f(x_curr)
    except Exception:
        return best_x

    if not _is_finite(f_curr):
        return best_x

    if abs(f_curr) < best_f:
        best_x = x_curr
        best_f = abs(f_curr)

    if best_f <= tol:
        return x_curr

    for _ in range(max_iter):
        if _time_left() <= 0:
            break

        dx = x_curr - x_prev
        if dx == 0:
            break
        
        m = (f_curr - f_prev) / dx
        if m == 0 or not _is_finite(m):
            break

        x_next = x_curr - f_curr / m
        if not _is_finite(x_next):
            break

        try:
            f_next = f(x_next)
        except Exception:
            break

        if not _is_finite(f_next):
            break

        abs_f_next = abs(f_next)
        if abs_f_next < best_f:
            best_x = x_next
            best_f = abs_f_next

        if abs_f_next <= tol:
            return x_next

        # Shift window
        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_next, f_next

    return best_x
