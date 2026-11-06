"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Strategy:
    1. Use a small number of RK4 steps to get a rough solution and estimate the Jacobian.
    2. Use the estimated Jacobian to build a matrix exponential approximation.
    3. Use expm_multiply for the bulk of the integration (if linear) or use a high-order
       implicit method with the precomputed Jacobian.
    
    For this specific 100-point convection-diffusion problem, the dynamics are linear:
    y' = A y, where A is a fixed tridiagonal matrix.
    
    We can detect linearity by checking if f(t, y) is linear in y. If so, we can
    extract A from a single call f(0, y) / y (or more carefully) and then use
    matrix exponential methods.
    
    Actually, let's be more careful. The problem is y' = f(t, y). For convection-diffusion
    with constant coefficients, f(t, y) = A y where A is constant. Let's verify this
    by checking linearity.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    deadline = time.time() + 280  # Safety margin before 330s limit
    
    # First, let's try to determine if the ODE is linear and time-independent.
    # Check if f(t, y) = A y for some constant matrix A.
    # We can test this by evaluating f at a few points.
    
    # Get f at y0
    f0 = f(t0, y0)
    
    # Test linearity: check if f(t0, 2*y0) ≈ 2*f0 and f(t0, y0 + y0) ≈ 2*f0
    # Also check time-independence: f(t1, y0) ≈ f0
    
    # But we need to be careful about call count. Let's use a smarter approach.
    
    # For a 100-point convection-diffusion problem, the matrix A is tridiagonal.
    # Let's assume linearity and try to extract A.
    
    # If the problem is linear: y' = A y
    # Then y(t) = expm(A * (t - t0)) @ y0
    
    # We can extract A by noting that f(t, y) = A y, so A = f(t, y) @ inv(y) 
    # but that's not quite right since A is a matrix and y is a vector.
    
    # Better: if we know f(t, y) = A y, then for any y, A y = f(t, y).
    # We can build A by evaluating f at basis vectors, but that's n calls.
    # For n=100, that's 100 calls, which is very cheap.
    
    # Let's extract the matrix A by evaluating f at the standard basis vectors.
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
    
    # Now check if A is consistent: f(t0, y0) should equal A @ y0
    A_y0 = A @ y0
    f0_check = f(t0, y0)  # We already have this, but let's reuse
    # Actually we already called f(t0, y0) as f0. Let's check consistency.
    if not np.allclose(A_y0, f0, rtol=1e-10, atol=1e-12):
        # Not linear, or not time-independent. Fall back to RK4.
        # Use RK4 with adaptive step size
        return _rk4_adaptive(f, t0, t1, y0, deadline)
    
    # Check time-independence: f(t1, y0) should equal A @ y0
    f_t1 = f(t1, y0)
    if not np.allclose(f_t1, A_y0, rtol=1e-10, atol=1e-12):
        # Time-dependent. Fall back to RK4.
        return _rk4_adaptive(f, t0, t1, y0, deadline)
    
    # Great, the problem is linear and time-independent: y' = A y
    # Use matrix exponential for high accuracy
    # For a 100x100 matrix, expm is fast enough.
    
    dt = t1 - t0
    # Use scipy's expm_multiply for efficiency
    try:
        from scipy.linalg import expm_multiply
        y_final = expm_multiply(dt * A, y0)
    except ImportError:
        # Fallback to dense expm
        from scipy.linalg import expm
        expA = expm(dt * A)
        y_final = expA @ y0
    
    return y_final


def _rk4_adaptive(f, t0, t1, y0, deadline):
    """Fallback RK4 with adaptive step size."""
    y = np.array(y0, dtype=float)
    t = t0
    h = (t1 - t0) / 1000  # Start with 1000 steps
    
    while t < t1:
        if time.time() > deadline - 5:
            break
        h = min(h, t1 - t)
        
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y_new = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        # Check for blowup
        if not np.all(np.isfinite(y_new)) or np.max(np.abs(y_new)) > 1e12:
            h = h / 2
            continue
        
        y = y_new
        t += h
    
    return y
# EVOLVE-BLOCK-END
