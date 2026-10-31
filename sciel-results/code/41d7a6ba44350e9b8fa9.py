"""
Optimized uncertainty bound computation for Hermite-Gaussian test function.
Uses precomputed matrix products for fast objective evaluation with analytic gradient.
"""
import time
import numpy as np
from scipy.optimize import minimize

T_START = time.time()
DEADLINE = 205.0  # safety margin from 210s hard limit

def _time_left():
    return DEADLINE - (time.time() - T_START)

def _precompute():
    """Precompute all matrix products needed for the objective."""
    # Grid
    x = np.linspace(-15, 15, 20000)
    dx = x[1] - x[0]
    phi = np.exp(-0.5 * x * x)
    
    # Compute Hermite polynomials H_0, H_4, H_8, H_12
    H = [None] * 13
    H[0] = np.ones_like(x)
    H[1] = 2.0 * x
    for n in range(2, 13):
        H[n] = 2.0 * x * H[n-1] - 2.0 * (n-1) * H[n-2]
    
    h0 = H[0]
    h4 = H[4]
    h8 = H[8]
    h12 = H[12]
    
    # Values at 0 for constraint
    h0_0 = H[0][0]  # This is at x=-15, not x=0. Let's compute properly.
    # Find index closest to 0
    idx0 = np.argmin(np.abs(x))
    h0_0 = h0[idx0]
    h4_0 = h4[idx0]
    h8_0 = h8[idx0]
    h12_0 = h12[idx0]
    
    # Precompute basis functions: b_i = H_i * phi
    b0 = h0 * phi
    b4 = h4 * phi
    b8 = h8 * phi
    b12 = h12 * phi
    
    # Precompute x^2 * b_i
    x2 = x**2
    xb0 = x2 * b0
    xb4 = x2 * b4
    xb8 = x2 * b8
    xb12 = x2 * b12
    
    # Precompute second derivatives numerically
    def _second_derivative(f):
        """Compute second derivative using central differences."""
        d2 = np.zeros_like(f)
        d2[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / dx**2
        d2[0] = (f[1] - 2*f[0] + f[2]) / dx**2  # forward at boundary
        d2[-1] = (f[-2] - 2*f[-1] + f[-3]) / dx**2  # backward at boundary
        return d2
    
    db0 = _second_derivative(b0)
    db4 = _second_derivative(b4)
    db8 = _second_derivative(b8)
    db12 = _second_derivative(b12)
    
    # Precompute all dot products (integrals)
    # ||b_i * b_j|| = sum(b_i * b_j) * dx
    def _dot(a, b):
        return np.sum(a * b) * dx
    
    # Gram matrix for normalization: M[i,j] = <b_i, b_j> for i,j in {0,4,8,12}
    M00 = _dot(b0, b0)
    M04 = _dot(b0, b4)
    M08 = _dot(b0, b8)
    M012 = _dot(b0, b12)
    M44 = _dot(b4, b4)
    M48 = _dot(b4, b8)
    M412 = _dot(b4, b12)
    M88 = _dot(b8, b8)
    M812 = _dot(b8, b12)
    M1212 = _dot(b12, b12)
    
    # Position 4th moment cross terms: P[i,j] = <x^2*b_i, x^2*b_j>
    P00 = _dot(xb0, xb0)
    P04 = _dot(xb0, xb4)
    P08 = _dot(xb0, xb8)
    P012 = _dot(xb0, xb12)
    P44 = _dot(xb4, xb4)
    P48 = _dot(xb4, xb8)
    P412 = _dot(xb4, xb12)
    P88 = _dot(xb8, xb8)
    P812 = _dot(xb8, xb12)
    P1212 = _dot(xb12, xb12)
    
    # Frequency 4th moment cross terms: F[i,j] = <db_i, db_j>
    F00 = _dot(db0, db0)
    F04 = _dot(db0, db4)
    F08 = _dot(db0, db8)
    F012 = _dot(db0, db12)
    F44 = _dot(db4, db4)
    F48 = _dot(db4, db8)
    F412 = _dot(db4, db12)
    F88 = _dot(db8, db8)
    F812 = _dot(db8, db12)
    F1212 = _dot(db12, db12)
    
    return {
        'x': x, 'dx': dx, 'idx0': idx0,
        'h0_0': h0_0, 'h4_0': h4_0, 'h8_0': h8_0, 'h12_0': h12_0,
        'M': [[M00, M04, M08, M012],
              [M04, M44, M48, M412],
              [M08, M48, M88, M812],
              [M012, M412, M812, M1212]],
        'P': [[P00, P04, P08, P012],
              [P04, P44, P48, P412],
              [P08, P48, P88, P812],
              [P012, P412, P812, P1212]],
        'F': [[F00, F04, F08, F012],
              [F04, F44, F48, F412],
              [F08, F48, F88, F812],
              [F012, F412, F812, F1212]]
    }

def _objective_and_grad(params, pre):
    """Compute objective value and gradient for given (c0, c4, c8)."""
    if _time_left() < 3.0:
        return 1e10, np.zeros(3)
    
    c0, c4, c8 = params
    h0_0 = pre['h0_0']
    h4_0 = pre['h4_0']
    h8_0 = pre['h8_0']
    h12_0 = pre['h12_0']
    
    # c12 from constraint: c0*h0_0 + c4*h4_0 + c8*h8_0 + c12*h12_0 = 0
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    
    coeffs = np.array([c0, c4, c8, c12])
    M = np.array(pre['M'])
    P = np.array(pre['P'])
    F = np.array(pre['F'])
    
    # Normalization: ||g||^2 = coeffs^T M coeffs
    norm_sq = coeffs @ M @ coeffs
    if norm_sq < 1e-15:
        return 1e10, np.zeros(3)
    
    norm = np.sqrt(norm_sq)
    g_norm = coeffs / norm
    
    # Position 4th moment squared: (g_norm^T P g_norm)
    pos_sq = g_norm @ P @ g_norm
    pos = np.sqrt(max(pos_sq, 0))
    
    # Frequency 4th moment squared: (g_norm^T F g_norm)
    freq_sq = g_norm @ F @ g_norm
    freq = np.sqrt(max(freq_sq, 0))
    
    obj = pos * freq
    
    # Gradient computation
    # obj = pos * freq
    # d(obj)/d(coeffs) = freq * d(pos)/d(coeffs) + pos * d(freq)/d(coeffs)
    # d(pos)/d(coeffs) = (1/(2*pos)) * d(pos_sq)/d(coeffs)
    # d(pos_sq)/d(coeffs) = 2 * P @ g_norm (but g_norm depends on coeffs through normalization)
    
    # Let u = coeffs, n = ||u||_M = sqrt(u^T M u)
    # g_norm = u / n
    # d(g_norm)/du = (1/n) I - (u/n^3) u^T M ... this gets complicated.
    
    # Alternative: use automatic-like differentiation manually
    # pos_sq = (u^T P u) / (u^T M u)
    # d(pos_sq)/du = 2(P u)/(u^T M u) - 2(u^T P u)(M u)/(u^T M u)^2
    # = 2/(u^T M u) * (P u - pos_sq * M u)
    
    u = coeffs
    n_sq = u @ M @ u
    Pu = P @ u
    Mu = M @ u
    Fu = F @ u
    
    # d(pos_sq)/du
    dpos_sq = (2.0 / n_sq) * (Pu - (u @ Pu / n_sq) * Mu)
    # d(freq_sq)/du
    dfreq_sq = (2.0 / n_sq) * (Fu - (u @ Fu / n_sq) * Mu)
    
    # d(pos)/du = dpos_sq / (2 * pos)
    # d(freq)/du = dfreq_sq / (2 * freq)
    
    if pos > 1e-15 and freq > 1e-15:
        dpos = dpos_sq / (2.0 * pos)
        dfreq = dfreq_sq / (2.0 * freq)
        grad_full = freq * dpos + pos * dfreq
    else:
        grad_full = np.zeros(4)
    
    # Now project gradient back to (c0, c4, c8) space
    # c12 = -(c0*h0_0 + c4*h4_0 + c8*h8_0) / h12_0
    # dc12/dc0 = -h0_0/h12_0
    # dc12/dc4 = -h4_0/h12_0
    # dc12/dc8 = -h8_0/h12_0
    
    dc12_dc0 = -h0_0 / h12_0
    dc12_dc4 = -h4_0 / h12_0
    dc12_dc8 = -h8_0 / h12_0
    
    grad0 = grad_full[0] + grad_full[3] * dc12_dc0
    grad4 = grad_full[1] + grad_full[3] * dc12_dc4
    grad8 = grad_full[2] + grad_full[3] * dc12_dc8
    
    grad = np.array([grad0, grad4, grad8])
    
    return obj, grad

def _objective(params, pre):
    val, _ = _objective_and_grad(params, pre)
    return val

def _objective_grad(params, pre):
    val, grad = _objective_and_grad(params, pre)
    return val, grad

def construct():
    """Find optimal coefficients (c0, c4, c8) minimizing the uncertainty bound."""
    if _time_left() < 15.0:
        # Return seed values if not enough time
        return np.array([-1.13, 0.04, 1.0/3240.0])
    
    # Precompute everything
    pre = _precompute()
    
    # Multiple starting points
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    starts = [
        x0,
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.5, 0.03, 0.0005]),
        np.array([-0.8, 0.06, 0.0002]),
        np.array([-1.2, 0.035, 0.00035]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([-2.0, 0.1, 0.001]),
        np.array([-0.5, 0.1, 0.001]),
        np.array([-1.1, 0.045, 0.00031]),
        np.array([-1.15, 0.038, 0.00029]),
    ]
    
    best_val = np.inf
    best_x = x0.copy()
    
    for start in starts:
        if _time_left() < 10.0:
            break
        try:
            # First try with analytic gradient
            res = minimize(
                lambda p: _objective_grad(p, pre),
                start,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            try:
                # Fallback to numerical gradient
                res = minimize(
                    lambda p: _objective(p, pre),
                    start,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
    
    # Polish with tighter tolerance if time allows
    if _time_left() > 15.0 and np.isfinite(best_val):
        try:
            res = minimize(
                lambda p: _objective_grad(p, pre),
                best_x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 5000}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x

if __name__ == "__main__":
    result = construct()
    print("Coefficients (c0, c4, c8):", result)
