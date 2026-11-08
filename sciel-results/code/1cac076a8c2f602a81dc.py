"""Root finder for 3 fixed 2D systems; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Explicit deadline guard: kill margin well under the 150s wall-clock limit.
_DEADLINE = 140.0  # seconds of wall-clock budget we allow ourselves
_t0 = None


def _elapsed():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return time.time() - _t0


def _time_ok():
    return _elapsed() < _DEADLINE


def _norm(x):
    return float(np.linalg.norm(x))


def _safe_F(F, x):
    """Call F safely; return None if it fails or returns a non-finite (2,) array."""
    try:
        y = F(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=float)
        if y.shape != (2,):
            return None
        if not np.all(np.isfinite(y)):
            return None
        return y
    except Exception:
        return None


def _fd_jacobian(F, x, Fx, h):
    """Forward-difference Jacobian using the already-computed Fx."""
    J = np.empty((2, 2), dtype=float)
    for j in range(2):
        e = np.zeros(2, dtype=float)
        e[j] = h
        y = _safe_F(F, x + e)
        if y is None:
            return None
        J[:, j] = (y - Fx) / h
    return J


def _newton_attempt(F, x0, max_iter=40):
    """Damped Newton with adaptive FD Jacobian and backtracking line search.
    Returns (best_x, best_norm, n_calls) for this attempt."""
    x = np.array(x0, dtype=float)
    n_calls = 0
    best_x = x.copy()
    best_norm = np.inf

    # Initial function evaluation
    Fx = _safe_F(F, x)
    if Fx is None:
        return x, np.inf, 1
    n_calls += 1
    nrm = _norm(Fx)
    if nrm < best_norm:
        best_norm, best_x = nrm, x.copy()
    if nrm <= 1e-12:
        return best_x, best_norm, n_calls

    # Adaptive step size for finite differences
    h = 1e-6
    scale = max(1.0, _norm(x))

    for _ in range(max_iter):
        if not _time_ok():
            break

        # Recompute Jacobian with current h
        J = _fd_jacobian(F, x, Fx, h)
        if J is None:
            # Try a smaller step once, then give up on this attempt
            h *= 0.1
            J = _fd_jacobian(F, x, Fx, h)
            if J is None:
                break
        n_calls += 2  # two extra F calls for the Jacobian

        # Check conditioning; if badly ill-conditioned, shrink h and retry
        try:
            cond = np.linalg.cond(J)
        except np.linalg.LinAlgError:
            break
        if cond > 1e12:
            h *= 0.1
            continue

        # Solve J dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Try least-squares fallback
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break

        # If the step is huge relative to scale, damp it
        step_norm = _norm(dx)
        if step_norm > 10 * scale:
            dx = dx * (10 * scale / step_norm)
            step_norm = _norm(dx)

        # Backtracking line search
        accepted = False
        for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]:
            x_new = x + alpha * dx
            Fx_new = _safe_F(F, x_new)
            n_calls += 1
            if Fx_new is None:
                continue
            nrm_new = _norm(Fx_new)
            if nrm_new < best_norm:
                best_norm, best_x = nrm_new, x_new.copy()
            if nrm_new <= 1e-12:
                return best_x, best_norm, n_calls
            # Armijo-like: accept if it reduces the residual
            if nrm_new < nrm * 1.05:  # allow slight increase to avoid stalling
                x, Fx, nrm = x_new, Fx_new, nrm_new
                accepted = True
                break
        if not accepted:
            # No progress; try shrinking h and continue
            h *= 0.1
            if h < 1e-14:
                break

        if nrm <= 1e-12:
            return best_x, best_norm, n_calls

    return best_x, best_norm, n_calls


def find_root_system(F, x0):
    global _t0
    if _t0 is None:
        _t0 = time.time()

    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (2,):
        x0 = np.array([float(x0[0]), float(x0[1])]) if len(x0) >= 2 else np.array([0.0, 0.0])

    best_x = x0.copy()
    best_norm = np.inf
    total_calls = 0

    # Strategy 1: Newton from x0
    x1, n1, c1 = _newton_attempt(F, x0)
    total_calls += c1
    if n1 < best_norm:
        best_norm, best_x = n1, x1
    if best_norm <= 1e-12 or not _time_ok():
        return best_x

    # Strategy 2: a few cheap perturbations if the first attempt stalled
    # Keep perturbations small to stay near the basin of attraction.
    for scale_p in [1e-3, 1e-2, 1e-1]:
        if not _time_ok():
            break
        for dx in [np.array([scale_p, 0.0]), np.array([-scale_p, 0.0]),
                   np.array([0.0, scale_p]), np.array([0.0, -scale_p])]:
            if not _time_ok():
                break
            x_try = x0 + dx
            x2, n2, c2 = _newton_attempt(F, x_try, max_iter=20)
            total_calls += c2
            if n2 < best_norm:
                best_norm, best_x = n2, x2
            if best_norm <= 1e-12:
                return best_x

    return best_x
# EVOLVE-BLOCK-END
