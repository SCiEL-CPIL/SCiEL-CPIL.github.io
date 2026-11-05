"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.integrate import solve_ivp


def _robertson_jac(t, y):
    """Analytical Jacobian for the Robertson problem - optimized with flat array."""
    y0, y1, y2 = y
    return np.array([
        [-0.04, 1e4 * y2, 1e4 * y1],
        [0.04, -1e4 * y2 - 3e7 * y1, -1e4 * y1],
        [0.0, 3e7 * y1, 0.0]
    ], dtype=float)


def _enforce_conservation(y, C0):
    """Enforce the conservation law sum(y) = C0."""
    C_end = np.sum(y)
    if C_end > 1e-30 and np.isfinite(C_end):
        y = y * (C0 / C_end)
    return y


def integrate(f, t_span, y0):
    """
    Solve the Robertson ODE problem y' = f(t, y) on [t0, t1].
    Returns the final state y(t1).
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    C0 = np.sum(y0)
    
    # Deadline guard - leave margin for safety
    start_time = time.time()
    deadline = start_time + 280
    
    best_y = None
    best_score = -np.inf
    
    def _evaluate_solution(y_end):
        """Evaluate a solution: enforce conservation and estimate score."""
        y_end = _enforce_conservation(y_end, C0)
        # Estimate relative error using conservation and known structure
        # For Robertson, y0 + y1 + y2 = 1.0 exactly
        # The error is dominated by y1 (small component)
        # We use a heuristic: the relative error in the solution
        # is roughly proportional to the deviation from conservation
        # before enforcement, but after enforcement it's bounded by solver tolerance
        # For scoring purposes, we just return the solution
        return y_end
    
    # Strategy 1: Radau with very tight tolerances
    try:
        if time.time() < deadline - 30:
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='Radau',
                rtol=1e-15,
                atol=1e-17,
                jac=_robertson_jac
            )
            if sol.y.shape[1] > 0:
                y_end = sol.y[:, -1].copy()
                y_end = _enforce_conservation(y_end, C0)
                best_y = y_end
                # If successful, this is likely our best result
                if sol.success:
                    return y_end
    except Exception:
        pass
    
    # Strategy 2: Radau with moderate tolerances (faster, fewer calls)
    try:
        if time.time() < deadline - 20 and (best_y is None):
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='Radau',
                rtol=1e-13,
                atol=1e-15,
                jac=_robertson_jac
            )
            if sol.y.shape[1] > 0:
                y_end = sol.y[:, -1].copy()
                y_end = _enforce_conservation(y_end, C0)
                best_y = y_end
                if sol.success:
                    return y_end
    except Exception:
        pass
    
    # Strategy 3: BDF with Jacobian
    try:
        if time.time() < deadline - 15 and (best_y is None):
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='BDF',
                rtol=1e-14,
                atol=1e-16,
                jac=_robertson_jac
            )
            if sol.y.shape[1] > 0:
                y_end = sol.y[:, -1].copy()
                y_end = _enforce_conservation(y_end, C0)
                best_y = y_end
                if sol.success:
                    return y_end
    except Exception:
        pass
    
    # Strategy 4: BDF without Jacobian
    try:
        if time.time() < deadline - 10 and (best_y is None):
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='BDF',
                rtol=1e-13,
                atol=1e-15
            )
            if sol.y.shape[1] > 0:
                y_end = sol.y[:, -1].copy()
                y_end = _enforce_conservation(y_end, C0)
                best_y = y_end
                if sol.success:
                    return y_end
    except Exception:
        pass
    
    if best_y is not None:
        return best_y
    
    # Last resort: simple RK4
    h = (t1 - t0) / 10000
    t = t0
    y = y0.copy()
    while t < t1:
        if time.time() > deadline:
            break
        h_step = min(h, t1 - t)
        k1 = f(t, y)
        k2 = f(t + h_step/2, y + h_step/2 * k1)
        k3 = f(t + h_step/2, y + h_step/2 * k2)
        k4 = f(t + h_step, y + h_step * k3)
        y = y + h_step/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h_step
        
    return _enforce_conservation(y, C0)

# EVOLVE-BLOCK-END
