# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, gammaln
from scipy.optimize import minimize

_DEADLINE = time.time() + 610  # safety margin below 630s hard kill

def _build_quadrature():
    """70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    cos_theta = x_gl
    phi = np.linspace(0, 2 * np.pi, 130, endpoint=False)
    dphi = 2.0 * np.pi / 130
    weights_1d = w_gl * dphi
    return cos_theta, phi, weights_1d

def _compute_harmonics_matrix(cos_theta, phi):
    """
    Compute X where X[i, j] = Y_j(cos_theta_i, phi_j).
    j in 0..20 corresponds to m = -10..10.
    """
    N_ct = len(cos_theta)
    N_phi = len(phi)
    N_pts = N_ct * N_phi
    X = np.zeros((N_pts, 21), dtype=np.float64)

    ct, ph = np.meshgrid(cos_theta, phi, indexing='ij')
    ct_flat = ct.ravel()
    ph_flat = ph.ravel()

    for k, m in enumerate(range(-10, 11)):
        abs_m = abs(m)
        log_N = 0.5 * (np.log(21.0 / (4.0 * np.pi)) +
                       gammaln(10 - abs_m + 1) - gammaln(10 + abs_m + 1))
        N_m = np.exp(log_N)

        P = lpmv(abs_m, 10, ct_flat)

        if m == 0:
            Y = N_m * P
        elif m > 0:
            Y = np.sqrt(2.0) * N_m * P * np.cos(m * ph_flat)
        else:
            Y = np.sqrt(2.0) * N_m * P * np.sin(abs_m * ph_flat)

        X[:, k] = Y

    return X

def _moment4(c, X, w_flat):
    """Return S4 = sum(w * f^4) for f = X @ c, assuming ||c||=1 (R = S4^(1/4))."""
    f = X @ c
    f2 = f * f
    f4 = f2 * f2
    return np.sum(w_flat * f4)

def _moment4_grad(c, X, w_flat):
    """Gradient of S4 w.r.t. c: 4 * X.T @ (w * f^3)."""
    f = X @ c
    f3 = f * f * f
    return 4.0 * (X.T @ (w_flat * f3))

def _ratio(c, X, w_flat):
    """R = ||f||_4 / ||f||_2 (computed directly for reporting/validation)."""
    f = X @ c
    f2 = f * f
    f4 = f2 * f2
    S2 = np.sum(w_flat * f2)
    S4 = np.sum(w_flat * f4)
    if S2 < 1e-30 or S4 < 1e-30:
        return np.inf
    return (S4 ** 0.25) / np.sqrt(S2)

def _ratio_and_grad(c, X, w_flat):
    """R and its gradient (for SLSQP on the constrained sphere)."""
    f = X @ c
    f2 = f * f
    f3 = f2 * f
    f4 = f2 * f2

    S2 = np.sum(w_flat * f2)
    S4 = np.sum(w_flat * f4)

    if S2 < 1e-10 or S4 < 1e-10:
        return 1e10, np.zeros(21)

    N2 = np.sqrt(S2)
    N4 = S4 ** 0.25
    R = N4 / N2

    term1 = X.T @ (w_flat * f3)
    term2 = X.T @ (w_flat * f)

    grad_log_R = term1 / S4 - term2 / S2
    grad_R = R * grad_log_R

    return R, grad_R

def _polish_constraint_free(c0, X, w_flat, maxiter, ftol):
    """
    Constraint-free polish: minimize S4(u/||u||) over u in R^21.
    This removes the ||c||=1 equality constraint via reparameterization,
    often reaching a tighter local minimum.
    """
    def obj(u):
        n = np.linalg.norm(u)
        if n < 1e-12:
            return 1e10
        c = u / n
        return _moment4(c, X, w_flat)

    def grad(u):
        n = np.linalg.norm(u)
        if n < 1e-12:
            return np.zeros(21)
        c = u / n
        g_c = _moment4_grad(c, X, w_flat)
        # d(c)/d(u) = (I - c c^T)/n  =>  grad_u = (g_c - (g_c.c) c)/n
        gc = np.dot(g_c, c)
        return (g_c - gc * c) / n

    try:
        res = minimize(obj, c0, jac=grad, method='BFGS',
                       options={'maxiter': maxiter, 'gtol': ftol})
        if res.success or np.isfinite(res.fun):
            c = res.x / np.linalg.norm(res.x)
            return c
    except Exception:
        pass
    return c0

def rudin_problem10() -> np.ndarray:
    """
    Minimize ||f||_4 / ||f||_2 for degree-10 real spherical harmonics.
    Returns 21 coefficients c_m for m = -10..10.
    """
    global _DEADLINE

    cos_theta, phi, w_1d = _build_quadrature()
    X = _compute_harmonics_matrix(cos_theta, phi)
    w_flat = np.tile(w_1d, 130)

    best_coeffs = None
    best_val = np.inf

    rng = np.random.default_rng(42)

    starts = []
    # Single-m starts for every m
    for m in range(-10, 11):
        c = np.zeros(21)
        c[10 + m] = 1.0
        starts.append(c)

    # Structured mixed starts: pairs of m values (opposite and same sign)
    for m1 in range(-10, 11, 2):
        for m2 in range(m1 + 2, 11, 2):
            if m1 == 0 or m2 == 0:
                continue
            c_mix = np.zeros(21)
            c_mix[10 + m1] = 1.0 / np.sqrt(2)
            c_mix[10 + m2] = 1.0 / np.sqrt(2)
            starts.append(c_mix)
            c_mix2 = np.zeros(21)
            c_mix2[10 + m1] = 1.0 / np.sqrt(2)
            c_mix2[10 + m2] = -1.0 / np.sqrt(2)
            starts.append(c_mix2)

    # Random starts
    for _ in range(80):
        c = rng.standard_normal(21)
        c = c / np.linalg.norm(c)
        starts.append(c)

    constraint = {'type': 'eq', 'fun': lambda c: np.dot(c, c) - 1.0,
                  'jac': lambda c: 2.0 * c}

    # Phase 1: SLSQP from each start (constrained on the sphere)
    for start in starts:
        if time.time() > _DEADLINE - 45:
            break
        try:
            res = minimize(
                lambda c: _ratio_and_grad(c, X, w_flat)[0],
                start,
                jac=lambda c: _ratio_and_grad(c, X, w_flat)[1],
                method='SLSQP',
                constraints=[constraint],
                options={'maxiter': 300, 'ftol': 1e-10}
            )
            c = res.x
            val = _ratio(c, X, w_flat)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_coeffs = c.copy()
        except Exception:
            pass

    # Phase 2: constraint-free polish (reparameterization) on the best
    if best_coeffs is not None and time.time() < _DEADLINE - 40:
        c_pol = _polish_constraint_free(best_coeffs, X, w_flat, maxiter=2000, ftol=1e-12)
        val = _ratio(c_pol, X, w_flat)
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_coeffs = c_pol.copy()

    # Phase 3: SLSQP refine (constrained) with tighter tolerance
    if best_coeffs is not None and time.time() < _DEADLINE - 30:
        try:
            res = minimize(
                lambda c: _ratio_and_grad(c, X, w_flat)[0],
                best_coeffs,
                jac=lambda c: _ratio_and_grad(c, X, w_flat)[1],
                method='SLSQP',
                constraints=[constraint],
                options={'maxiter': 2000, 'ftol': 1e-13}
            )
            c = res.x
            val = _ratio(c, X, w_flat)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_coeffs = c.copy()
        except Exception:
            pass

    # Phase 4: another constraint-free polish
    if best_coeffs is not None and time.time() < _DEADLINE - 20:
        c_pol = _polish_constraint_free(best_coeffs, X, w_flat, maxiter=3000, ftol=1e-14)
        val = _ratio(c_pol, X, w_flat)
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_coeffs = c_pol.copy()

    # Phase 5: final ultra-tight SLSQP polish
    if best_coeffs is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(
                lambda c: _ratio_and_grad(c, X, w_flat)[0],
                best_coeffs,
                jac=lambda c: _ratio_and_grad(c, X, w_flat)[1],
                method='SLSQP',
                constraints=[constraint],
                options={'maxiter': 5000, 'ftol': 1e-15}
            )
            c = res.x
            val = _ratio(c, X, w_flat)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_coeffs = c.copy()
        except Exception:
            pass

    if best_coeffs is None:
        best_coeffs = np.zeros(21)
        best_coeffs[10] = 1.0

    # Enforce validity bounds
    result = best_coeffs.astype(np.float64)
    if not np.all(np.isfinite(result)):
        result = np.zeros(21)
        result[10] = 1.0

    max_abs = np.max(np.abs(result))
    if max_abs < 1e-3:
        result = result * (0.01 / max_abs)
    elif max_abs > 1e5:
        result = result * (1e4 / max_abs)

    return result

# EVOLVE-BLOCK-END
