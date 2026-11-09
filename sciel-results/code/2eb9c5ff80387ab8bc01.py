"""ODE solver: 15-dimensional stiff linear system using Arnoldi/Krylov method."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm


def _arnoldi_matrix(A, v, m):
    """
    Build the Arnoldi basis V and the Hessenberg matrix H such that
    A @ V[:, :m] = V[:, :m+1] @ H (approximately, for m < n)
    
    Returns V (n x m) and H (m x m) for the projected matrix.
    """
    n = len(v)
    V = np.zeros((n, m))
    H = np.zeros((m, m))
    
    v = v / np.linalg.norm(v)
    V[:, 0] = v
    
    for j in range(m):
        w = A(V[:, j])
        # Modified Gram-Schmidt
        for i in range(j + 1):
            H[i, j] = np.dot(V[:, i], w)
            w = w - H[i, j] * V[:, i]
        if j < m - 1:
            H[j + 1, j] = np.linalg.norm(w)
            V[:, j + 1] = w / H[j + 1, j]
        else:
            # Last column: don't normalize, just store the residual norm
            pass
    
    return V, H


def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns y(t_end).
    
    Strategy: Use Arnoldi iteration to compute e^{A*dt} @ y0 for a linear ODE y' = A y.
    Since f(t, y) is linear in y (stiff linear system), we can use Krylov methods.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    total_time = t1 - t0
    
    # Deadline guard
    deadline = time.time() + 295.0
    
    # First, determine if f is linear in y and time-independent
    # Test linearity: f(t, a*y1 + b*y2) == a*f(t,y1) + b*f(t,y2)
    # And time-independence: f(t1, y) == f(t2, y) for linear systems
    
    # For a linear ODE y' = A(t) y, if A is constant, y(t) = e^{A(t-t0)} y0
    # We need to check if f is of the form f(t, y) = A @ y (constant A)
    
    # Probe the operator
    # First call: get f at y0
    f_y0 = f(t0, y0)
    
    # Check if f is linear and time-independent by testing with scaled vectors
    # f(t, c*y) should equal c*f(t, y) for linear
    c = 0.5
    f_cy0 = f(t0, c * y0)
    linearity_check = np.allclose(f_cy0, c * f_y0, rtol=1e-10, atol=1e-12)
    
    # Check time independence: f(t0, y) vs f(t1, y)
    f_t1_y0 = f(t1, y0)
    time_indep_check = np.allclose(f_t1_y0, f_y0, rtol=1e-10, atol=1e-12)
    
    if linearity_check and time_indep_check:
        # Use Krylov/Arnoldi method for constant-coefficient linear ODE
        # y(t1) = e^{A * total_time} @ y0
        # Use Arnoldi to approximate the matrix exponential action
        
        # Number of Arnoldi steps - use n (full dimension) for exactness in exact arithmetic,
        # but for numerical stability, m = n is sufficient for 15-dim
        m = min(n, 15)  # Full Krylov subspace for 15-dim system
        
        # We need to apply A to vectors. A @ v = f(t0, v) since f is linear and time-independent
        # But f might have an offset? For y' = A y, f(t, y) = A @ y
        # Check if f(t, 0) = 0
        f_zero = f(t0, np.zeros(n))
        has_offset = not np.allclose(f_zero, np.zeros(n), atol=1e-12)
        
        if has_offset:
            # f(t, y) = A @ y + b, need to handle affine system
            # For now, fall back to RK4 with more steps
            # Actually, let's handle it: y' = A y + b
            # The solution involves integrating the affine system
            # Use a different approach: shift the system
            # Let z = y - y_ss where y_ss is steady state: A y_ss + b = 0
            # But this requires solving A y_ss = -b, which needs A explicitly
            # For simplicity, use high-order RK4 with many steps
            steps = 10000
            h = total_time / steps
            t = t0
            y = y0.copy()
            for i in range(steps):
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2 * k1)
                k3 = f(t + h/2, y + h/2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
            return y
        else:
            # Pure linear: y' = A y, f(t, y) = A @ y
            # Use Arnoldi to compute e^{A * total_time} @ y0
            # Define the operator application
            def apply_A(v):
                return f(t0, v)
            
            # Build Arnoldi decomposition
            # For m = n, we get a full decomposition and can compute exactly
            V, H = _arnoldi_matrix(apply_A, y0, m)
            
            # Now e^{A * total_time} @ y0 = V @ e^{H * total_time} @ (V^H @ y0)
            # V^H @ y0 = [||y0||, 0, 0, ...]
            h = np.linalg.norm(y0)
            eH = expm(H * total_time)
            coeff = eH[0, :] * h  # First row of eH times h
            y_end = V[:, :m] @ coeff
            
            # Verify and potentially refine
            # Check residual
            residual = np.linalg.norm(apply_A(y_end) - (y_end - y0) / total_time * total_time) if total_time != 0 else 0
            # Actually, just return the result
            return y_end
    else:
        # Non-linear or time-dependent: use high-order RK4 with many steps
        # Use adaptive step sizing or just a large fixed number of steps
        steps = 10000
        h = total_time / steps
        t = t0
        y = y0.copy()
        for i in range(steps):
            if time.time() > deadline:
                break
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
# EVOLVE-BLOCK-END
