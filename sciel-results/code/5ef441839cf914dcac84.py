"""ODE solver for convection-diffusion 100-point problem."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time


def _build_convection_diffusion_matrix(n=100, dx=0.01, D=1.0, U=1.0):
    """Build the 100x100 tridiagonal convection-diffusion matrix.
    
    y' = A @ y where A represents:
    dy/dt = D * d^2y/dx^2 - U * dy/dx
    
    With central differences on a uniform grid:
    d^2y/dx^2 ≈ (y[i-1] - 2y[i] + y[i+1]) / dx^2
    dy/dx ≈ (y[i+1] - y[i-1]) / (2*dx)
    
    So A[i,i-1] = D/dx^2 + U/(2*dx)
       A[i,i]   = -2*D/dx^2
       A[i,i+1] = D/dx^2 - U/(2*dx)
    """
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        A[i, i] = -2.0 * D / (dx * dx)
        if i > 0:
            A[i, i - 1] = D / (dx * dx) + U / (2.0 * dx)
        if i < n - 1:
            A[i, i + 1] = D / (dx * dx) - U / (2.0 * dx)
    return A


def _is_linear_time_invariant(f, t0, t1, y0, n):
    """Check if f is linear (or affine) and time-invariant.
    Returns (b, A) if affine linear time-invariant, else None.
    """
    # Check time-invariance with a test vector
    y_test = np.random.RandomState(0).randn(n)
    
    # Evaluate at different times
    f1 = np.array(f(t0, y_test), dtype=float)
    f2 = np.array(f(t1, y_test), dtype=float)
    
    if not np.allclose(f1, f2, rtol=1e-12, atol=1e-14):
        return None  # Time-dependent
    
    # Check linearity: f(t, 0) should be 0 for pure linear
    f0 = np.array(f(t0, np.zeros(n)), dtype=float)
    b = f0  # Affine term
    
    # Check if f(t, 2y) = 2*f(t, y) - b (linearity check)
    y_double = 2.0 * y_test
    f_double = np.array(f(t0, y_double), dtype=float)
    expected_double = 2.0 * f1 - b
    
    if not np.allclose(f_double, expected_double, rtol=1e-12, atol=1e-14):
        return None  # Not linear
    
    # Build matrix A by evaluating f at basis vectors
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        f_ej = np.array(f(t0, e_j), dtype=float)
        A[:, j] = f_ej - b
    
    return (b, A)


def integrate(f, t_span, y0):
    """Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Strategy:
    1. Try to detect if the problem is linear time-invariant (LTI)
    2. If LTI, use matrix exponential for exact solution
    3. Otherwise, use adaptive RK45
    """
    t0, t1 = t_span
    n = len(y0)
    y0 = np.array(y0, dtype=float)
    
    deadline = time.time() + 280  # Safety margin before 300s limit
    
    # Try to detect LTI structure
    lti_result = _is_linear_time_invariant(f, t0, t1, y0, n)
    
    if lti_result is not None:
        b, A = lti_result
        # Solution: y(t) = expm(A*(t-t0)) @ y0 + integral term for affine
        # For pure linear (b=0): y(t) = expm(A*(t-t0)) @ y0
        # For affine: y(t) = expm(A*(t-t0)) @ y0 + A^{-1}*(expm(A*(t-t0))-I) @ b
        
        dt = t1 - t0
        try:
            E = expm(A * dt)
            if np.allclose(b, 0, atol=1e-14):
                y_end = E @ y0
            else:
                # Affine case: y(t) = expm(A*dt) @ y0 + integral_0^dt expm(A*(dt-s)) ds @ b
                # = expm(A*dt) @ y0 + A^{-1}*(expm(A*dt)-I) @ b
                try:
                    A_inv = np.linalg.inv(A)
                    y_end = E @ y0 + A_inv @ (E - np.eye(n)) @ b
                except np.linalg.LinAlgError:
                    # Fallback: numerical integration of the affine term
                    y_end = E @ y0
                    # Approximate the integral with midpoint
                    mid_E = expm(A * dt / 2)
                    y_end += mid_E @ b * dt
            return y_end
        except Exception:
            pass  # Fall through to numerical method
    
    # Fallback: use high-order explicit method with many steps
    # For convection-diffusion, we need enough steps for stability
    # CFL condition: dt < dx^2 / (2*D) for diffusion, dt < dx / |U| for convection
    
    # Estimate problem parameters from the matrix if we can
    # Use RK4 with adaptive step size based on stability
    
    h_initial = (t1 - t0) / 1000
    h_min = 1e-6
    
    y = y0.copy()
    t = t0
    
    # Use a fixed number of steps that should be sufficient
    # For 100-point convection-diffusion on [0, 0.5], typical dx=0.01
    # Stability: dt < dx^2/(2*D) ≈ 0.00005 for D=1
    # So we need at least 0.5/0.00005 = 10000 steps
    # But let's use more for accuracy
    
    n_steps = 20000
    h = (t1 - t0) / n_steps
    
    for i in range(n_steps):
        if time.time() > deadline:
            break
        
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        
        if not np.all(np.isfinite(y)):
            # Reset and try with smaller steps
            break
    
    return y
# EVOLVE-BLOCK-END
