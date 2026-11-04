"""Refined Cohn-Goncalves Laguerre construction: generalized Laguerre alpha + analytic-gradient L-BFGS."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_genlaguerre, roots_laguerre
from numpy.polynomial.polynomial import polyfromroots, polymul

_START = time.time()
_DEADLINE = 600.0
_MARGIN = 45.0


def _time_left():
    return _DEADLINE - (time.time() - _START) - _MARGIN


# Precompute factorials up to 80 (degree of P^2 is 48, moments up to that)
_FACTS = np.array([math.factorial(k) for k in range(80)], dtype=np.float64)


def _poly_moment(coeffs):
    """Compute integral of poly (ascending coeff) * e^{-x} dx = sum_k coeffs[k]*k!."""
    n = min(len(coeffs), len(_FACTS))
    if n == 0:
        return 0.0
    return float(np.dot(coeffs[:n], _FACTS[:n]))


def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2 / I1^2 where I1 = int P e^{-x} dx, I2 = int P^2 e^{-x} dx.
    P = prod (x-x_i)^2. Returns (c4, grad) or just c4."""
    roots = np.asarray(roots, dtype=np.float64)
    n = len(roots)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    # Build B(x) = prod (x - x_i), then P = B^2, P2 = P^2 = B^4
    B = polyfromroots(roots)  # ascending order, degree 12
    P = polymul(B, B)         # degree 24
    P2 = polymul(P, P)        # degree 48

    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf
    c4 = I2 / (I1 * I1)
    if not need_grad:
        return float(c4)

    # Analytic gradient
    # P = B^2, dP/dx_i = 2*B * dB/dx_i where dB/dx_i = -R_i, R_i = prod_{j!=i}(x-x_j)
    # dI1/dx_i = -2 * int B*R_i * e^{-x} dx
    # P2 = B^4, dP2/dx_i = 4*B^3 * dB/dx_i = -4*B^3*R_i
    # dI2/dx_i = -4 * int B^3*R_i * e^{-x} dx
    # dC4/dx_i = dI2/(I1^2) - 2*I2*dI1/(I1^3)
    
    # Precompute B^3 once (degree 36)
    B3 = polymul(polymul(B, B), B)
    
    grad = np.zeros_like(roots)
    for i in range(n):
        # R_i = prod_{j!=i}(x-x_j)
        R_i = np.array([1.0])
        for j in range(n):
            if j == i:
                continue
            R_i = polymul(R_i, np.array([-roots[j], 1.0]))
        
        # dI1/dx_i = -2 * int B*R_i * e^{-x} dx
        BR_i = polymul(B, R_i)  # degree 12 + 11 = 23
        dI1_i = -2.0 * _poly_moment(BR_i)
        
        # dI2/dx_i = -4 * int B^3 * R_i * e^{-x} dx
        B3R_i = polymul(B3, R_i)  # degree 47
        dI2_i = -4.0 * _poly_moment(B3R_i)
        
        grad[i] = dI2_i / (I1 * I1) - 2.0 * I2 * dI1_i / (I1 ** 3)
    
    return float(c4), grad


def _objective(roots):
    return _compute_c4(roots)


def _objective_grad(roots):
    c4, g = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return np.inf, np.zeros_like(roots)
    return c4, g


def _c4_from_alpha(alpha):
    """Compute C4 for roots of generalized Laguerre polynomial L_12^{(alpha)}."""
    try:
        if alpha < -1:
            return np.inf
        xg, _ = roots_genlaguerre(12, alpha)
        xg = np.sort(xg)
        if not np.all(xg > 0) or not np.isfinite(xg).all():
            return np.inf
        return _compute_c4(xg)
    except Exception:
        return np.inf


def _jacobi_init(beta_scale=1.0, gamma_scale=1.0):
    """Construct 12x12 Jacobi matrix from modified Laguerre recurrence.
    Returns eigenvalues (positive, ordered) as initial root locations."""
    try:
        N = 12
        # Standard Laguerre recurrence: x p_k = (2k+1+alpha) p_k - k(k+alpha) p_{k-1}
        # For alpha=0: a_k = 2k+1, b_k = k^2
        # Modify with tunable scales to explore different root density profiles
        diag = np.array([2.0 * k + 1.0 for k in range(N)]) * beta_scale
        offdiag = np.array([k * k for k in range(1, N)]) * gamma_scale
        # Build symmetric tridiagonal Jacobi matrix
        J = np.zeros((N, N))
        for k in range(N):
            J[k, k] = diag[k]
        for k in range(N - 1):
            J[k, k + 1] = offdiag[k]
            J[k + 1, k] = offdiag[k]
        eigvals = np.linalg.eigvalsh(J)
        eigvals = np.sort(eigvals)
        if np.all(eigvals > 0) and np.isfinite(eigvals).all():
            return eigvals
    except Exception:
        pass
    return None


def construct():
    m = 12
    n_roots = 12

    # Phase 1: 1D search over alpha to find optimal root density profile
    best_alpha = 0.0
    best_c4_alpha = np.inf

    # Grid search over log-space alpha
    if _time_left() > 200:
        log_alphas = np.linspace(math.log(1e-3), math.log(1e4), 300)
        for la in log_alphas:
            if _time_left() < 20:
                break
            c4_val = _c4_from_alpha(np.exp(la))
            if c4_val < best_c4_alpha:
                best_c4_alpha = c4_val
                best_alpha = np.exp(la)

    # Refine alpha with local search
    if _time_left() > 150 and best_alpha > 0:
        try:
            from scipy.optimize import minimize_scalar
            res_alpha = minimize_scalar(
                lambda la: _c4_from_alpha(np.exp(la)),
                bounds=(math.log(max(best_alpha * 0.5, 1e-4)), math.log(min(best_alpha * 2.0, 1e5))),
                method='bounded',
                options={'xatol': 1e-10, 'maxiter': 500}
            )
            if res_alpha.success and np.isfinite(res_alpha.fun) and res_alpha.fun < best_c4_alpha:
                best_alpha = np.exp(res_alpha.x)
                best_c4_alpha = res_alpha.fun
        except Exception:
            pass

    # Get roots from best alpha
    best_roots = None
    best_c4 = np.inf
    try:
        xg, _ = roots_genlaguerre(12, best_alpha)
        best_roots = np.sort(xg)
        best_c4 = _compute_c4(best_roots)
    except Exception:
        pass

    # Also try standard Laguerre (alpha=0)
    try:
        x0, _ = roots_laguerre(12)
        x0 = np.sort(x0)
        c4_0 = _compute_c4(x0)
        if c4_0 < best_c4:
            best_c4 = c4_0
            best_roots = x0.copy()
    except Exception:
        pass

    # Phase 1.5: Jacobi-matrix-based initialization (modified Laguerre recurrence)
    # This maps to a continuous manifold of orthogonal polynomial parameters
    # that inherently enforce positive, ordered roots while allowing flexible density
    if _time_left() > 80:
        for bs in [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
            for gs in [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
                if _time_left() < 15:
                    break
                jroots = _jacobi_init(bs, gs)
                if jroots is None:
                    continue
                try:
                    res_j = minimize(
                        _objective_grad,
                        jroots,
                        jac=True,
                        method='L-BFGS-B',
                        bounds=[(1e-8, None)] * 12,
                        options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 40}
                    )
                    if np.all(res_j.x > 0) and np.isfinite(res_j.fun):
                        candidate = np.sort(res_j.x)
                        c4_j = _compute_c4(candidate)
                        if c4_j < best_c4:
                            best_c4 = c4_j
                            best_roots = candidate.copy()
                except Exception:
                    pass

    # Fallback if no valid roots found
    if best_roots is None:
        best_roots = np.array([4.0 * (i + 1) for i in range(n_roots)], dtype=float)
        best_c4 = _compute_c4(best_roots)

    # Phase 2: Full 12-DOF L-BFGS-B refinement with analytic gradient
    if _time_left() > 100:
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-8, None)] * 12,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 3: Multi-start with perturbed initializations
    if _time_left() > 60:
        rng = np.random.default_rng(42)
        for trial in range(15):
            if _time_left() < 12:
                break
            # Perturb the best roots
            scale = 0.08 + 0.08 * trial
            pert = best_roots * (1.0 + scale * rng.standard_normal(12))
            pert = np.maximum(pert, 1e-3)
            pert = np.sort(pert)
            try:
                res = minimize(
                    _objective_grad,
                    pert,
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(1e-8, None)] * 12,
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40}
                )
                if np.all(res.x > 0) and np.isfinite(res.fun):
                    candidate = np.sort(res.x)
                    c4 = _compute_c4(candidate)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
            except Exception:
                pass

    # Phase 4: Try neighboring alpha values and polish
    if _time_left() > 40:
        for alpha_offset in [0.5, 2.0, 0.1, 5.0, 0.2, 0.8, 10.0, 0.05]:
            if _time_left() < 10:
                break
            for sign in [-1, 1]:
                a_try = best_alpha * alpha_offset if sign > 0 else best_alpha / alpha_offset
                if a_try < -0.99 or a_try < 1e-6 or a_try > 1e5:
                    continue
                try:
                    xg2, _ = roots_genlaguerre(12, a_try)
                    xg2 = np.sort(xg2)
                    if not np.all(xg2 > 0):
                        continue
                    res2 = minimize(
                        _objective_grad,
                        xg2,
                        jac=True,
                        method='L-BFGS-B',
                        bounds=[(1e-8, None)] * 12,
                        options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 30}
                    )
                    if np.all(res2.x > 0) and np.isfinite(res2.fun):
                        candidate = np.sort(res2.x)
                        c4 = _compute_c4(candidate)
                        if c4 < best_c4:
                            best_c4 = c4
                            best_roots = candidate.copy()
                except Exception:
                    pass

    # Phase 5: Final high-precision polish (two passes for tighter convergence)
    if _time_left() > 30:
        for polish_pass in range(2):
            if _time_left() < 15:
                break
            try:
                res = minimize(
                    _objective_grad,
                    best_roots,
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(1e-10, None)] * 12,
                    options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100}
                )
                if np.all(res.x > 0) and np.isfinite(res.fun):
                    candidate = np.sort(res.x)
                    c4 = _compute_c4(candidate)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
            except Exception:
                pass

    return [float(r) for r in best_roots]
