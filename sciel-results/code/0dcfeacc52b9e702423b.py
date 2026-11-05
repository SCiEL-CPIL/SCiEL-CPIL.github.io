"""ODE solver using matrix-exponential for linear systems, with fallback."""
import numpy as np
from scipy.linalg import expm, eig, solve


def _is_linear_time_invariant(f, t0, t1, y0, n):
    """Check if f is linear (or affine) and time-invariant.
    Returns (b, A) if affine linear time-invariant, else None.
    """
    # Check time-invariance with a test vector
    y_test = np.random.RandomState(0).randn(n)
    y_test = y_test / np.linalg.norm(y_test)
    ft0 = f(t0, y_test)
    ft1 = f(t1, y_test)
    if not np.allclose(ft0, ft1, rtol=1e-10, atol=1e-12):
        return None  # time-dependent
    
    # Check affine: f(t, 0) should be constant (the affine part b)
    b = f(t0, np.zeros(n))
    
    # Check linearity: f(t, y1 + y2) == f(t,y1) + f(t,y2) - b
    # Use a couple of test vectors
    y1 = np.random.RandomState(1).randn(n)
    y2 = np.random.RandomState(2).randn(n)
    f_y1 = f(t0, y1)
    f_y2 = f(t0, y2)
    f_y12 = f(t0, y1 + y2)
    if not np.allclose(f_y12, f_y1 + f_y2 - b, rtol=1e-8, atol=1e-10):
        return None  # not affine linear
    
    return b


def _build_matrix_A(f, t0, n, b):
    """Build the matrix A such that f(t, y) = A y + b for all y.
    Uses n evaluations: f(t, e_j) = A[:,j] + b, so A[:,j] = f(t,e_j) - b.
    """
    A = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        A[:, j] = f(t0, e) - b
    return A


def _verify_A(f, t0, A, b, n):
    """Verify that f(t, y) ≈ A y + b for a few random vectors."""
    for seed in [3, 4, 5]:
        y = np.random.RandomState(seed).randn(n)
        y = y / np.linalg.norm(y)
        lhs = f(t0, y)
        rhs = A @ y + b
        if not np.allclose(lhs, rhs, rtol=1e-8, atol=1e-10):
            return False
    return True


def _integrate_linear_eig(f, t_span, y0, A, b):
    """Integrate y' = A y + b using eigenvalue decomposition (Spectral method)."""
    t0, t1 = t_span
    n = len(y0)
    y = np.array(y0, dtype=float)
    dt = t1 - t0
    
    # Calculate steady state y_ss = -A^{-1} b if b is not zero
    if np.allclose(b, 0.0, atol=1e-14):
        y_ss = np.zeros(n)
    else:
        try:
            y_ss = -solve(A, b)
        except Exception:
            return None
            
    # Shift system to homogeneous: z = y - y_ss => z' = A z
    z0 = y - y_ss
    
    # Eigen decomposition
    try:
        w, V = eig(A)
    except Exception:
        return None

    # Check condition number of V to ensure stability
    cond_V = np.linalg.cond(V)
    if cond_V > 1e12:
        return None

    # Solve V * c = z0  => c = V^{-1} z0
    try:
        c = solve(V, z0)
    except Exception:
        return None

    # Compute e^(w * dt) for each eigenvalue
    exp_w = np.exp(w * dt)
    
    # z_end = V * (exp_w * c)
    z_end = V @ (exp_w * c)
    
    # y_end = y_ss + z_end
    y_end = y_ss + z_end.real
    
    if not np.all(np.isfinite(y_end)):
        return None
        
    return y_end


def _rk45_adaptive(f, t_span, y0, rtol=1e-12, atol=1e-14, max_steps=2000000):
    """Fallback: adaptive RK45 (Dormand-Prince) for nonlinear/time-dependent systems."""
    t0, t1 = t_span
    n = len(y0)
    y = np.array(y0, dtype=float)
    t = t0
    
    # Dormand-Prince coefficients
    c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    A = [
        [],
        [1/5],
        [3/40, 9/40],
        [44/45, -56/15, 32/9],
        [19372/6561, -25360/2187, 64448/6561, -212/729],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84],
    ]
    b = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    b_star = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
    
    h = (t1 - t0) / 1000
    h = max(h, 1e-12)
    
    for step in range(max_steps):
        if t >= t1:
            break
        if t + h > t1:
            h = t1 - t
        
        # Compute stages
        k = [None] * 7
        k[0] = f(t, y)
        for i in range(1, 7):
            yi = y.copy()
            ti = t
            for j in range(i):
                yi = yi + h * A[i][j] * k[j]
                ti = ti + h * c[j]
            k[i] = f(ti, yi)
        
        # 5th order solution
        y_new = y.copy()
        for i in range(6):
            y_new = y_new + h * b[i] * k[i]
        
        # 4th order estimate for error
        y_est = y.copy()
        for i in range(6):
            y_est = y_est + h * b_star[i] * k[i]
        
        err = np.max(np.abs(y_new - y_est) / (atol + rtol * np.maximum(np.abs(y), np.abs(y_new))))
        
        if err <= 1.0:
            y = y_new
            t = t + h
            # Increase step
            if err > 0:
                h = h * min(5.0, max(0.2, 0.9 * err ** (-0.2)))
            else:
                h = h * 5.0
        else:
            # Reject step, decrease
            h = h * max(0.1, 0.9 * err ** (-0.25))
        
        if not np.all(np.isfinite(y)):
            return None
    
    return y


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Try Spectral (Eigen) approach for linear time-invariant systems
    try:
        check = _is_linear_time_invariant(f, t0, t1, y0, n)
        if check is not None:
            b = check
            A = _build_matrix_A(f, t0, n, b)
            if _verify_A(f, t0, A, b, n):
                y = _integrate_linear_eig(f, t_span, y0, A, b)
                if y is not None:
                    return y
    except Exception:
        pass
    
    # Fallback: adaptive RK45
    y = _rk45_adaptive(f, t_span, y0)
    if y is not None:
        return y
    
    # Final fallback: fixed-step RK4
    h = (t1 - t0) / 10000
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(10000):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
