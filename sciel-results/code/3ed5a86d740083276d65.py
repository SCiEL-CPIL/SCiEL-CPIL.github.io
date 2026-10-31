"""Analytic uncertainty bound with spherical + log-magnitude reparameterization + L-BFGS-B + Trust-Constr polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195
_START_TIME = time.time()

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
def _precompute_I():
    """Precompute I_n for n=0..13"""
    I = np.zeros(14)
    sp = np.sqrt(np.pi)
    I[0] = sp
    for n in range(1, 14):
        I[n] = (2*n - 1) * I[n-1] / (4.0 * n)
    return I

_I = _precompute_I()

def _spherical_to_coefficients(theta, phi):
    """Convert spherical coordinates to (a0, a4, a8) on unit sphere."""
    a0 = np.cos(theta)
    a4 = np.sin(theta) * np.cos(phi)
    a8 = np.sin(theta) * np.sin(phi)
    return np.array([a0, a4, a8])

def _coefficients_to_spherical(a0, a4, a8):
    """Convert (a0, a4, a8) to (theta, phi) on unit sphere."""
    r = np.sqrt(a0**2 + a4**2 + a8**2)
    if r < 1e-15:
        return (np.pi/2, 0.0)
    a0n, a4n, a8n = a0/r, a4/r, a8/r
    theta = np.arccos(np.clip(a0n, -1.0, 1.0))
    phi = np.arctan2(a8n, a4n)
    return (theta, phi)

def _logsign_to_coefficients(params):
    """
    Convert log-magnitude + sign parameters to coefficients.
    params = [log|a0|, sign_a0, log|a4|, sign_a4, log|a8|, sign_a8]
    where sign is in (-1, 1) via tanh, log|a| is unbounded.
    Returns (a0, a4, a8).
    """
    log_mags = params[0::2]  # indices 0, 2, 4
    signs = params[1::2]     # indices 1, 3, 5
    # Clamp log magnitudes to avoid overflow
    log_mags_clamped = np.clip(log_mags, -30, 30)
    mags = np.exp(log_mags_clamped)
    # Map sign parameter through tanh to get (-1, 1), then scale
    sign_vals = np.tanh(signs)
    coeffs = mags * sign_vals
    return coeffs

def _coefficients_to_logsign(a0, a4, a8):
    """Convert coefficients to log-magnitude + sign parameters."""
    coeffs = np.array([a0, a4, a8])
    mags = np.abs(coeffs)
    signs = np.sign(coeffs)
    # Handle zero coefficients
    log_mags = np.zeros(3)
    for i in range(3):
        if mags[i] > 1e-30:
            log_mags[i] = np.log(mags[i])
        else:
            log_mags[i] = -30.0  # effectively zero
    # Invert tanh: sign_param = atanh(sign_val)
    sign_params = np.zeros(3)
    for i in range(3):
        s = np.clip(signs[i], -0.999999, 0.999999)
        sign_params[i] = np.arctanh(s)
    # Interleave: [log|a0|, sign_a0, log|a4|, sign_a4, log|a8|, sign_a8]
    params = np.zeros(6)
    params[0::2] = log_mags
    params[1::2] = sign_params
    return params

def _logsign_jacobian(params):
    """
    Jacobian of coefficients w.r.t. logsign parameters.
    params = [log|a0|, s0, log|a4|, s4, log|a8|, s8]
    Returns 3x6 matrix where row i is d(coeff_i)/d(params).
    """
    log_mags = np.clip(params[0::2], -30, 30)
    signs_raw = params[1::2]
    mags = np.exp(log_mags)
    sign_vals = np.tanh(signs_raw)
    
    jac = np.zeros((3, 6))
    for i in range(3):
        # d(coeff_i)/d(log|a_i|) = sign_val_i * |a_i|
        jac[i, 2*i] = sign_vals[i] * mags[i]
        # d(coeff_i)/d(s_i) = |a_i| * sech^2(s_i)
        sech2 = 1.0 - sign_vals[i]**2
        jac[i, 2*i+1] = mags[i] * sech2
    return jac


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    p = np.empty(7)
    p[0] = 0.0
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    
    I = _I
    pp = np.outer(p, p)
    m = np.zeros(3)
    for k in range(3):
        idx_mat = k + np.add.outer(np.arange(7), np.arange(7))
        m[k] = np.sum(pp * I[idx_mat])
    
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    q = p * signs
    qq = np.outer(q, q)
    n = np.zeros(3)
    for k in range(3):
        n[k] = 2.0 * np.pi * np.sum(qq * I[k + np.add.outer(np.arange(7), np.arange(7))])
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    S_p = np.zeros((3, 7))
    S_q = np.zeros((3, 7))
    for k in range(3):
        for i in range(7):
            s_p = 0.0
            s_q = 0.0
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    s_p += p[j] * I[idx]
                    s_q += q[j] * I[idx]
            S_p[k, i] = s_p
            S_q[k, i] = s_q
    
    dpda = np.zeros((3, 7))
    dpda[0, 1] = 8.0
    dpda[1, 1] = -30.0
    dpda[2, 1] = -1260.0
    dpda[0, 2] = -13860.0/945.0
    dpda[1, 2] = 45.0
    dpda[2, 2] = 1750.0
    dpda[0, 3] = 6930.0/945.0
    dpda[1, 3] = -22.0
    dpda[2, 3] = -798.0
    dpda[0, 4] = -1155.0/945.0
    dpda[1, 4] = 3465.0/945.0
    dpda[2, 4] = 1 + 1155.0*105.0/945.0
    dpda[0, 5] = 66.0/945.0
    dpda[1, 5] = -198.0/945.0
    dpda[2, 5] = -6930.0/945.0
    dpda[0, 6] = -1.0/945.0
    dpda[1, 6] = -3.0/945.0
    dpda[2, 6] = -105.0/945.0
    
    grad = np.zeros(3)
    for c in range(3):
        dp = dpda[c]
        dm = 2.0 * (dp @ S_p)
        dq = dp * signs
        dn = 4.0 * np.pi * (dq @ S_q)
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad

def _objective(params):
    val, _ = _bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad

def _objective_hess(params):
    """Finite difference Hessian for trust-constr."""
    h = 1e-6
    _, g = _bound_and_grad(params)
    H = np.zeros((3, 3))
    for i in range(3):
        ei = np.zeros(3)
        ei[i] = h
        _, gp = _bound_and_grad(params + ei)
        _, gm = _bound_and_grad(params - ei)
        H[:, i] = (gp - gm) / (2*h)
    return H

def _objective_logsign(params):
    """Objective in log-magnitude/sign parameterization."""
    coeffs = _logsign_to_coefficients(params)
    val, _ = _bound_and_grad(coeffs)
    return val

def _objective_grad_logsign(params):
    """Objective and gradient in log-magnitude/sign parameterization."""
    coeffs = _logsign_to_coefficients(params)
    bound, grad_coeff = _bound_and_grad(coeffs)
    jac = _logsign_jacobian(params)
    grad_logsign = jac.T @ grad_coeff
    return bound, grad_logsign

def _bound_from_spherical(theta, phi):
    """Compute bound from spherical coordinates (theta, phi) on unit sphere."""
    coeffs = _spherical_to_coefficients(theta, phi)
    val, _ = _bound_and_grad(coeffs)
    return val


def _bound_and_grad_spherical(params):
    """
    Compute bound and gradient in spherical coordinates (theta, phi).
    Uses chain rule: d/d(theta,phi) = (d/d a0, d/d a4, d/d a8) @ (d(a0,a4,a8)/d(theta,phi))
    """
    theta, phi = params[0], params[1]
    coeffs = _spherical_to_coefficients(theta, phi)
    bound, grad_coeff = _bound_and_grad(coeffs)
    
    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
    
    jac = np.array([
        [-st, 0.0],
        [ct*cp, -st*sp],
        [ct*sp, st*cp]
    ])
    
    grad_spherical = jac.T @ grad_coeff
    return bound, grad_spherical


def _objective_spherical(params):
    val, _ = _bound_and_grad_spherical(params)
    return val


def _objective_grad_spherical(params):
    val, grad = _bound_and_grad_spherical(params)
    return val, grad


def construct():
    # Convert seed to spherical coordinates
    seed_coeffs = np.array([-1.13, 0.04, 0.000308642])
    seed_theta, seed_phi = _coefficients_to_spherical(*seed_coeffs)
    seed_logsign = _coefficients_to_logsign(*seed_coeffs)
    
    # Generate diverse starting points in spherical coordinates
    rng = np.random.default_rng(42)
    starts_sph = []
    starts_logsign = []
    
    # Seed points
    starts_sph.append(np.array([seed_theta, seed_phi]))
    starts_logsign.append(seed_logsign.copy())
    
    # Moderate grid of starting points on the sphere for global coverage
    # Increased resolution slightly
    for theta in np.linspace(0.1, np.pi - 0.1, 25):
        for phi in np.linspace(0.1, 2*np.pi - 0.1, 25):
            starts_sph.append(np.array([theta, phi]))
    
    # Random spherical points
    for _ in range(80):
        theta = rng.uniform(0.05, np.pi - 0.05)
        phi = rng.uniform(0.05, 2*np.pi - 0.05)
        starts_sph.append(np.array([theta, phi]))
    
    # Log-sign starting points: seed + perturbations
    for _ in range(60):
        perturbed = seed_logsign + rng.normal(0, 1.5, size=6)
        # Clamp log magnitudes to reasonable range
        perturbed[0::2] = np.clip(perturbed[0::2], -5, 5)
        starts_logsign.append(perturbed)
    
    # Also add some structured log-sign starts
    for log_mag in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        for sign_val in [-0.5, 0.0, 0.5]:
            for sign_val2 in [-0.5, 0.0, 0.5]:
                params = np.array([log_mag, np.arctanh(np.clip(sign_val, -0.99, 0.99)),
                                   log_mag * 0.5, np.arctanh(np.clip(sign_val2, -0.99, 0.99)),
                                   log_mag * 0.1, np.arctanh(np.clip(sign_val * 0.5, -0.99, 0.99))])
                starts_logsign.append(params)
    
    best_val = np.inf
    best_params_sph = None
    best_params_linear = None
    best_params_logsign = None
    
    # Phase 1: Multi-start L-BFGS-B in spherical coordinates
    for start in starts_sph:
        if time.time() > _DEADLINE - 120:
            break
        try:
            res = minimize(_objective_grad_spherical, start, method='L-BFGS-B',
                          jac=True, bounds=[(0.001, np.pi - 0.001), (0.001, 2*np.pi - 0.001)],
                          options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-13, 'maxls': 30})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params_sph = res.x.copy()
                best_params_linear = _spherical_to_coefficients(res.x[0], res.x[1])
        except Exception:
            pass
    
    # Phase 2: Multi-start L-BFGS-B in log-sign coordinates
    for start in starts_logsign:
        if time.time() > _DEADLINE - 60:
            break
        try:
            res = minimize(_objective_grad_logsign, start, method='L-BFGS-B',
                          jac=True, bounds=[(-30, 30)] * 3 + [(-10, 10)] * 3,
                          options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params_logsign = res.x.copy()
                best_params_linear = _logsign_to_coefficients(res.x)
        except Exception:
            pass
    
    # Phase 3: Refine best in linear coordinates
    if best_params_linear is not None and time.time() < _DEADLINE - 40:
        try:
            res = minimize(_objective_grad, best_params_linear, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params_linear = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Trust-Constr polish in linear coordinates
    # This method is often more accurate near the optimum than L-BFGS-B
    if best_params_linear is not None and time.time() < _DEADLINE - 20:
        try:
            res = minimize(_objective_grad, best_params_linear, method='trust-constr',
                          jac=True, hess=_objective_hess,
                          options={'maxiter': 2000, 'gtol': 1e-12, 'xtol': 1e-12})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params_linear = res.x.copy()
        except Exception:
            pass
            
    # Phase 5: Nelder-Mead polish in linear coordinates (robust to flat regions)
    if best_params_linear is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective, best_params_linear, method='Nelder-Mead',
                          options={'maxiter': 30000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params_linear = res.x.copy()
        except Exception:
            pass

    # Phase 6: Final ultra-precise L-BFGS-B in linear coords
    if best_params_linear is not None and time.time() < _DEADLINE - 2:
        try:
            res = minimize(_objective_grad, best_params_linear, method='L-BFGS-B',
                          jac=True, options={'maxiter': 30000, 'ftol': 1e-19, 'gtol': 1e-17, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params_linear = res.x.copy()
        except Exception:
            pass
    
    if best_params_linear is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params_linear)
# EVOLVE-BLOCK-END
