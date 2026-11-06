"""
Strategy: Identify the linear operator A from f, then use matrix exponential for exact solution.
For the 2D heat equation on a 10x10 grid, f(t, y) = A @ y (autonomous linear system).
We can recover A by probing f with basis vectors, then compute y(t_end) = expm(A * dt) @ y0.

This requires ~n calls to f (where n=100) to build A, plus potentially a few more for verification.
The matrix exponential gives near-machine-precision accuracy.

Score = -log10(relative_error) - 0.5 * log10(num_calls)
With ~100-200 calls and ~1e-14 error: score ≈ 14 - 0.5*log10(200) ≈ 14 - 1.15 ≈ 12.85
"""

import numpy as np
import time
from scipy.linalg import expm


def _build_linear_operator(f, t, y0, n):
    """
    Recover the matrix A such that f(t, y) = A @ y for a linear autonomous ODE.
    Uses y0 as the base point and probes with perturbations.
    
    For a truly linear system f(t,y) = A@y, we can get columns of A by:
      f(t, e_j * scale) / scale ≈ A[:, j]
    
    But if there's any nonlinearity or offset, we use:
      A[:, j] = (f(t, y0 + h*e_j) - f(t, y0)) / h
    
    We need at least n+1 evaluations (1 for f(y0), n for the columns).
    """
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # First, get f at y0 (may be zero for homogeneous linear system)
    f_y0 = f(t, y0)
    
    # Determine perturbation size
    # Use a scale based on the magnitude of y0
    y0_scale = np.max(np.abs(y0)) if np.max(np.abs(y0)) > 0 else 1.0
    
    A = np.zeros((n, n))
    h = 1e-8 * y0_scale
    
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        f_perturbed = f(t, y0 + e)
        A[:, j] = (f_perturbed - f_y0) / h
    
    return A


def _build_linear_operator_v2(f, t, y0, n):
    """
    Alternative: if the system is homogeneous linear (f(0) = 0), we can use:
      A[:, j] = f(t, h * e_j) / h
    This saves one evaluation.
    """
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Check if f is approximately linear and homogeneous
    # Test with a small vector
    test_vec = np.ones(n) * 1e-6
    f_test = f(t, test_vec)
    f_zero = f(t, np.zeros(n))
    
    # If f(0) is close to zero and f is approximately linear, use homogeneous approach
    if np.linalg.norm(f_zero) < 1e-10 * (1 + np.max(np.abs(y0))):
        # Homogeneous: A[:, j] = f(h * e_j) / h
        h = 1e-8
        A = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            f_ej = f(t, e)
            A[:, j] = f_ej / h
        return A
    else:
        # Non-homogeneous or non-linear: use central difference around y0
        return _build_linear_operator(f, t, y0, n)


def _rk4_step(f, t, y, h):
    """Single RK4 step."""
    k1 = f(t, y)
    k2 = f(t + h/2, y + h/2 * k1)
    k3 = f(t + h/2, y + h/2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + h/6 * (k1 + 2*k2 + 2*k3 + k4)


def _rk4_solve(f, t0, t1, y0, steps):
    """Solve using RK4 with given number of steps."""
    h = (t1 - t0) / steps
    t = t0
    y = np.array(y0, dtype=float)
    for _ in range(steps):
        y = _rk4_step(f, t, y, h)
        t += h
        if not np.all(np.isfinite(y)):
            return None
    return y


def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard: aim to finish well before 330s
    start_time = time.time()
    deadline = start_time + 300  # 300s safety margin
    
    # Strategy 1: Try to identify as linear system and use matrix exponential
    # This is the most accurate approach for the heat equation
    
    try:
        # Build the linear operator
        # For a 10x10 grid, n=100, so we need ~100-101 calls
        A = _build_linear_operator_v2(f, t0, y0, n)
        
        # Check time
        if time.time() > deadline:
            # Fallback to RK4
            y = _rk4_solve(f, t0, t1, y0, 2000)
            return y if y is not None else y0
        
        # Compute matrix exponential: y(t1) = expm(A * dt) @ y0
        # For the heat equation, A is negative definite, so expm(A*dt) is well-conditioned
        expA_dt = expm(A * dt)
        y_result = expA_dt @ y0
        
        # Verify: check if the result is finite and reasonable
        if np.all(np.isfinite(y_result)):
            # Quick sanity check: compare with a few RK4 steps
            # Use a small number of RK4 steps to verify
            y_rk4_check = _rk4_solve(f, t0, t1, y0, 100)
            if y_rk4_check is not None and np.all(np.isfinite(y_rk4_check)):
                # If they're close, we're confident
                rel_diff = np.linalg.norm(y_result - y_rk4_check) / (1 + np.linalg.norm(y_rk4_check))
                if rel_diff < 1e-3:
                    return y_result
                # If not close, something went wrong with A construction
                # Fall through to RK4
            else:
                return y_result
        
        # If verification failed, fall through
    except Exception:
        pass
    
    # Strategy 2: Fallback to RK4 with adaptive-like approach
    # Use a reasonable number of steps
    y = _rk4_solve(f, t0, t1, y0, 2000)
    if y is not None:
        return y
    
    # Last resort: return initial condition
    return y0
