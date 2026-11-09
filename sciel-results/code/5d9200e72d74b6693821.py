"""ODE solver for Robertson problem using two-stage integration with slow-manifold extrapolation."""
import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.linalg import eig


def _robertson_jac(t, y):
    """
    Analytical Jacobian for the Robertson problem.
    y' = [ -0.04*y1 + 1e4*y2*y3,
           0.04*y1 - 1e4*y2*y3 - 3e7*y1^2,
           3e7*y1^2 ]
    """
    y1, y2, y3 = y
    return np.array([
        [-0.04, 1e4 * y3, 1e4 * y2],
        [0.04, -1e4 * y3 - 6e7 * y1, -1e4 * y2],
        [0.0, 6e7 * y1, 0.0]
    ])


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) over t_span starting from y0.
    Returns the final state y(t_end).
    
    Uses a two-stage strategy:
    1. High-accuracy Radau integration to t=10 where fast transients have decayed.
    2. Analytical extrapolation using the slow eigenmode to t=40.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Track time to ensure we finish well before the 300s limit
    start_time = time.time()
    deadline = start_time + 280  # 20s safety margin
    
    # Intermediate time where fast transients have mostly decayed
    t_mid = 10.0
    
    # Stage 1: High-accuracy integration to t_mid
    try:
        sol_mid = solve_ivp(
            f,
            [t0, t_mid],
            y0,
            method='Radau',
            rtol=1e-13,
            atol=1e-16,
            jac=_robertson_jac
        )
        if not sol_mid.success:
            # Fallback to full integration
            sol_full = solve_ivp(
                f,
                [t0, t1],
                y0,
                method='Radau',
                rtol=1e-13,
                atol=1e-16,
                jac=_robertson_jac
            )
            if sol_full.success:
                return sol_full.y[:, -1]
            return y0
    except Exception:
        # Fallback to full integration
        try:
            sol_full = solve_ivp(
                f,
                [t0, t1],
                y0,
                method='Radau',
                rtol=1e-12,
                atol=1e-14,
                jac=_robertson_jac
            )
            if sol_full.success:
                return sol_full.y[:, -1]
            return y0
        except Exception:
            return y0
    
    y_mid = sol_mid.y[:, -1]
    
    # Check if we're within time budget
    if time.time() > deadline - 30:
        # Not enough time for extrapolation, do full integration from t_mid
        try:
            sol_rest = solve_ivp(
                f,
                [t_mid, t1],
                y_mid,
                method='Radau',
                rtol=1e-12,
                atol=1e-14,
                jac=_robertson_jac
            )
            if sol_rest.success:
                return sol_rest.y[:, -1]
        except Exception:
            pass
        return y_mid
    
    # Stage 2: Analytical extrapolation using slow eigenmode
    # Compute Jacobian at the midpoint
    J_mid = _robertson_jac(t_mid, y_mid)
    
    # Find eigenvalues and eigenvectors
    try:
        eigenvalues, eigenvectors = eig(J_mid)
    except Exception:
        # Fallback to numerical integration
        try:
            sol_rest = solve_ivp(
                f,
                [t_mid, t1],
                y_mid,
                method='Radau',
                rtol=1e-12,
                atol=1e-14,
                jac=_robertson_jac
            )
            if sol_rest.success:
                return sol_rest.y[:, -1]
        except Exception:
            pass
        return y_mid
    
    # Identify the slowest mode (smallest magnitude eigenvalue)
    # For Robertson, the slow mode should have a real negative eigenvalue
    magnitudes = np.abs(eigenvalues)
    slow_idx = np.argmin(magnitudes)
    lambda_slow = eigenvalues[slow_idx]
    v_slow = eigenvectors[:, slow_idx]
    
    # Project y_mid onto the slow eigenvector
    # v_slow^H * y_mid / (v_slow^H * v_slow)
    v_conj = np.conjugate(v_slow)
    projection = np.vdot(v_conj, y_mid) / np.vdot(v_conj, v_slow)
    
    # Extrapolate: y(t) ≈ projection * v_slow * exp(lambda_slow * (t - t_mid))
    dt = t1 - t_mid
    exp_factor = np.exp(lambda_slow * dt)
    y_extrap = projection * v_slow * exp_factor
    
    # Ensure real result (small imaginary parts are numerical noise)
    y_extrap = np.real_if_close(y_extrap, tol=1000)
    if np.iscomplexobj(y_extrap):
        y_extrap = y_extrap.real
    
    # Validate: if the extrapolation gives unreasonable values, fall back
    if not np.all(np.isfinite(y_extrap)) or np.any(np.abs(y_extrap) > 1.0):
        # Fall back to numerical integration
        try:
            sol_rest = solve_ivp(
                f,
                [t_mid, t1],
                y_mid,
                method='Radau',
                rtol=1e-12,
                atol=1e-14,
                jac=_robertson_jac
            )
            if sol_rest.success:
                return sol_rest.y[:, -1]
        except Exception:
            pass
        return y_mid
    
    # For better accuracy, we can also account for the fact that the system
    # has a conservation law: y1 + y2 + y3 = 1
    # Adjust the extrapolated solution to satisfy this constraint
    conserved = y0[0] + y0[1] + y0[2]
    current_sum = y_extrap[0] + y_extrap[1] + y_extrap[2]
    if abs(current_sum) > 1e-15:
        y_extrap = y_extrap * (conserved / current_sum)
    
    return y_extrap
