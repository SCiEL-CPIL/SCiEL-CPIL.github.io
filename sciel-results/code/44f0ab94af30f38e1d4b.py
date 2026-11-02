"""Rewritten: correct uncertainty-product ratio objective + direct SVD solve + polish."""
import numpy as np
import time
import math
from scipy.optimize import minimize

# Hermite norms (physicists'): ||H_n||^2 = 2^n n! sqrt(pi); sqrt(pi) cancels in ratios.
_DEG = np.array([0, 4, 8, 12], dtype=float)
_N = np.array([float((2.0 ** d) * math.factorial(int(d))) for d in _DEG])
_K = _DEG  # degree values
# C(x^2 f, f) diagonal: k * N_k
_D1 = _K * _N
# C(x^2 f, x^2 f) diagonal: (k^2 + 2k) * N_k
_D2 = (_K ** 2 + 2.0 * _K) * _N

# P(0) = 0 constraint: a0*1 + a4*12 + a8*1680 + a12*2032128 = 0
_H0 = np.array([1.0, 12.0, 1680.0, 2032128.0])


def _bound_from_coeffs(c):
    """Compute the squared uncertainty product bound for coefficient vector c."""
    c2 = c * c
    Nf = float(np.dot(c2, _N))
    if Nf < 1e-30:
        return 1e10
    N1 = float(np.dot(c2, _D1))
    N2 = float(np.dot(c2, _D2))
    if N2 < 1e-30:
        return 1e10
    return (N1 * N1) / (Nf * N2)


def _bound_gauged(u):
    """Gauged objective: a0 = 1, a4 = u[0], a8 = u[1], a12 from P(0)=0."""
    a0 = 1.0
    a4 = u[0]
    a8 = u[1]
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 2032128.0
    return _bound_from_coeffs(np.array([a0, a4, a8, a12]))


def _gauged_from_coeffs(c):
    """Extract gauged parameters (a4, a8) from full coefficient vector."""
    a0 = c[0]
    if abs(a0) < 1e-15:
        return np.array([0.0, 0.0])
    return np.array([c[1] / a0, c[2] / a0])


def _svd_solve():
    """
    Solve the eigenvalue problem for the optimal ratio.
    The bound R = (c^T D1 c)^2 / ((c^T D0 c)(c^T D2 c)) under constraint H0^T c = 0.
    By symmetry, the optimal c satisfies a generalized eigenvalue problem.
    We use a direct approach: minimize R over the constraint subspace.
    """
    # Constraint subspace: H0^T c = 0. Parameterize c = M @ u where M spans null(H0).
    # H0 = [1, 12, 1680, 2032128]. Null space basis (3x4).
    H0 = np.array([[1.0, 12.0, 1680.0, 2032128.0]])
    # Compute null space via SVD
    U, s, Vt = np.linalg.svd(H0)
    M = Vt[1:].T  # 4x3 matrix, columns span null space
    
    # In this basis: c = M @ u
    # D0' = M^T D0 M, D1' = M^T D1 M, D2' = M^T D2 M
    D0p = M.T @ np.diag(_N) @ M
    D1p = M.T @ np.diag(_D1) @ M
    D2p = M.T @ np.diag(_D2) @ M
    
    # Objective: R(u) = (u^T D1' u)^2 / ((u^T D0' u)(u^T D2' u))
    # This is a ratio of quadratic forms. The minimum over u (up to scale) can be found
    # by noting that at the optimum, the gradient vanishes.
    # Alternative: parametrize u = [1, p, q] (fix gauge in null space) and do 2D optimization.
    # But let's try a direct eigenvalue approach.
    
    # The ratio R = (u^T B u)^2 / ((u^T A u)(u^T C u)) where A=D0', B=D1', C=D2'
    # At the minimum, by Lagrange multipliers, we get:
    # 2 B u (u^T B u) / ((u^T A u)(u^T C u)) = lambda_A A u + lambda_C C u
    # This is complex. Let's just do a well-initialized 2D/3D optimization.
    
    # Gauge in null space: set u[0] = 1, optimize u[1], u[2].
    def _obj_gauged_null(v):
        u = np.array([1.0, v[0], v[1]])
        uT_D1u = float(u @ D1p @ u)
        uT_D0u = float(u @ D0p @ u)
        uT_D2u = float(u @ D2p @ u)
        if abs(uT_D0u) < 1e-30 or abs(uT_D2u) < 1e-30:
            return 1e10
        return (uT_D1u * uT_D1u) / (uT_D0u * uT_D2u)
    
    # Multiple starting points
    best_val = 1e10
    best_u = None
    starts = [
        np.array([0.0, 0.0]),
        np.array([0.01, 0.0]),
        np.array([0.0, 0.01]),
        np.array([-0.01, 0.0]),
        np.array([0.001, 0.001]),
        np.array([0.1, 0.0]),
        np.array([0.0, 0.1]),
        np.array([-0.1, -0.1]),
        np.array([0.5, 0.5]),
        np.array([-0.5, 0.5]),
        np.array([0.01, -0.01]),
        np.array([-0.01, 0.01]),
    ]
    
    for x0 in starts:
        try:
            res = minimize(_obj_gauged_null, x0, method='L-BFGS-B',
                          options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_u = res.x
        except Exception:
            pass
    
    if best_u is None:
        return None
    
    # Reconstruct full coefficient vector
    u_full = np.array([1.0, best_u[0], best_u[1]])
    c = M @ u_full
    return c, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 160.0  # Safety margin (210s kill, leave 50s buffer)
    
    # Phase 1: Direct SVD-based solve with multiple starts
    best_coeffs = None
    best_val = 1e10
    
    svd_result = _svd_solve()
    if svd_result is not None:
        best_coeffs, best_val = svd_result
    
    # Phase 2: Also try the original gauged approach (a0=1) for cross-check
    if time.time() < deadline - 30:
        try:
            starts_g = [
                np.array([0.001, 0.001]),
                np.array([0.01, 0.0]),
                np.array([0.0, 0.01]),
                np.array([0.1, 0.0]),
                np.array([0.0, 0.1]),
                np.array([-0.1, 0.1]),
                np.array([0.5, -0.5]),
            ]
            for x0 in starts_g:
                if time.time() > deadline - 30:
                    break
                try:
                    res = minimize(_bound_gauged, x0, method='L-BFGS-B',
                                  options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-14})
                    if res.fun < best_val:
                        best_val = res.fun
                        a0 = 1.0
                        a4 = res.x[0]
                        a8 = res.x[1]
                        a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 2032128.0
                        best_coeffs = np.array([a0, a4, a8, a12])
                except Exception:
                    pass
        except Exception:
            pass
    
    # Phase 3: High-precision polish
    if best_coeffs is not None and time.time() < deadline - 10:
        try:
            u_gauged = _gauged_from_coeffs(best_coeffs)
            res_polish = minimize(_bound_gauged, u_gauged, method='Nelder-Mead',
                                  options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-16,
                                           'adaptive': True})
            if res_polish.fun < best_val:
                best_val = res_polish.fun
                a0 = 1.0
                a4 = res_polish.x[0]
                a8 = res_polish.x[1]
                a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 2032128.0
                best_coeffs = np.array([a0, a4, a8, a12])
        except Exception:
            pass
    
    # Fallback: if everything failed, return a reasonable default
    if best_coeffs is None:
        # Use the seed values as fallback
        best_coeffs = np.array([-1.13, 0.04, 1.0 / 3240.0])
        a12 = -(best_coeffs[0] + 12.0 * best_coeffs[1] + 1680.0 * best_coeffs[2]) / 2032128.0
        best_coeffs = np.array([best_coeffs[0], best_coeffs[1], best_coeffs[2], a12])
    
    # Return the three free coefficients (a0, a4, a8)
    return best_coeffs[:3]
