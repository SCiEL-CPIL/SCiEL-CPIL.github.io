"""Rewritten program using Exact Matrix Exponential for Linear Systems."""
import numpy as np
import scipy.linalg as la

def _rk4(f, t0, t1, y0, steps):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
    return y

def _is_linear(f, t0, y0, A, g, tol=1e-9):
    """Check if f(t, y) is approximately A @ y + g."""
    y_actual = f(t0, y0)
    y_model = A @ y0 + g
    norm_actual = np.linalg.norm(y_actual)
    if norm_actual < 1e-12:
        return np.linalg.norm(y_model) < 1e-9
    rel_err = np.linalg.norm(y_actual - y_model) / norm_actual
    return rel_err < tol

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    N = len(y0_arr)
    
    # 1. Probe for Linear Operator A and Constant g
    A = np.zeros((N, N))
    e = np.zeros(N)
    
    for j in range(N):
        e[j] = 1.0
        v = f(t0, e)
        A[:, j] = v
        e[j] = 0.0
        
    g = f(t0, np.zeros(N))
    
    # 2. Verify Linearity at t0 and mid-point
    if not _is_linear(f, t0, y0_arr, A, g):
        return _rk4(f, t0, t1, y0_arr, 5000)
        
    t_mid = (t0 + t1) / 2
    # Check at a different point to ensure A is constant (autonomous)
    # Use y0 scaled to avoid trivial check
    y_check = y0_arr * 0.5 + np.ones(N) * 0.1
    if not _is_linear(f, t_mid, y_check, A, g):
        return _rk4(f, t0, t1, y0_arr, 5000)

    # 3. Exact Matrix Exponential Integration
    # y(t) = exp(A(t-t0)) y0 + integral exp(A(t-tau)) g dtau
    # We step by h.
    # y_{k+1} = exp(Ah) y_k + C
    # where C = integral_0^h exp(A sigma) g dsigma
    
    steps = 20  # Small number of steps is sufficient since it's exact
    h = (t1 - t0) / steps
    
    try:
        E = la.expm(A * h)
        
        # Compute C = integral_0^h exp(A sigma) g dsigma
        # Use Gauss-Legendre quadrature with 16 points
        # Points in [0, 1] mapped to [0, h]
        x, w = np.polynomial.legendre.leggauss(16)
        # Map to [0, h]
        sigma = 0.5 * h * (x + 1)
        weights = 0.5 * h * w
        
        C = np.zeros(N)
        for i in range(16):
            # exp(A * sigma[i]) @ g
            E_sigma = la.expm(A * sigma[i])
            C += weights[i] * (E_sigma @ g)
            
    except Exception:
        # Fallback if expm fails
        return _rk4(f, t0, t1, y0_arr, 5000)

    y_curr = y0_arr.copy()
    for _ in range(steps):
        y_curr = E @ y_curr + C
        
    return y_curr
