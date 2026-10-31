"""Optimized program: exploit homogeneity, grid+Brent on sphere, L-BFGS polish."""
import numpy as np
from scipy.optimize import minimize, brent
import time
from math import factorial, sqrt, pi

_START = time.time()
_DEADLINE = 200.0  # safety margin before 210s kill

# Precompute constants
_DEGREES = np.array([0, 4, 8, 12])
_N = 4

# Precompute 2^k * k! * sqrt(pi) for each degree
_NORM_FACTORS = np.array([float(2**k * factorial(k)) for k in _DEGREES]) * sqrt(pi)

# Precompute the M_x matrix: M_x[i,j] = ∫ H_i(x) x^2 H_j(x) exp(-x^2) dx
# Using: x^2 H_j = (1/2) H_{j+2} + (2j+1) H_j + (j(j-1)/2) H_{j-2}
_MX = np.zeros((_N, _N))
for ii in range(_N):
    for jj in range(_N):
        i_deg = _DEGREES[ii]
        j_deg = _DEGREES[jj]
        nf = float(2**j_deg * factorial(j_deg)) * sqrt(pi)
        term = 0.0
        if i_deg == j_deg + 2:
            term += 0.5 * nf
        if i_deg == j_deg:
            term += (2 * j_deg + 1) * nf
        if i_deg == j_deg - 2 and j_deg >= 2:
            term += (j_deg * (j_deg - 1) / 2.0) * nf
        _MX[ii, jj] = term

# For degrees {0,4,8,12}: i^k = 1 for all k (since i^4=1, i^8=1, i^12=1)
# So M_p = M_x
_MP = _MX.copy()

# Constraint: g(0) = 0 => c0 + 12*c4 + 1680*c8 + 544320*c12 = 0
_C12_DENOM = 544320.0

def _get_full_coeffs(params):
    """Get all 4 coefficients from the 3 free parameters."""
    c0, c4, c8 = params
    c12 = -(c0 + 12.0 * c4 + 1680.0 * c8) / _C12_DENOM
    return np.array([c0, c4, c8, c12])

def _compute_bound(params):
    """Compute the upper bound A(f)*A(F(f)) for given 3 free coefficients."""
    coeffs = _get_full_coeffs(params)
    
    # ||g||^2 = sum c_k^2 * norm_factor_k
    norm_sq = np.sum(coeffs**2 * _NORM_FACTORS)
    if norm_sq <= 1e-30:
        return np.inf
    
    # <x^2> = coeffs^T @ M_x @ coeffs / norm_sq
    x2_moment = coeffs @ _MX @ coeffs
    x2_var = x2_moment / norm_sq
    
    # <p^2> = coeffs^T @ M_p @ coeffs / norm_sq
    p2_moment = coeffs @ _MP @ coeffs
    p2_var = p2_moment / norm_sq
    
    return x2_var * p2_var

def _compute_bound_and_grad(params):
    """Compute bound and gradient analytically."""
    coeffs = _get_full_coeffs(params)
    
    norm_sq = np.sum(coeffs**2 * _NORM_FACTORS)
    if norm_sq <= 1e-30:
        return np.inf, np.zeros(3)
    
    x2_moment = coeffs @ _MX @ coeffs
    p2_moment = coeffs @ _MP @ coeffs
    x2_var = x2_moment / norm_sq
    p2_var = p2_moment / norm_sq
    
    bound = x2_var * p2_var
    
    # Jacobian of coeffs w.r.t. params
    J = np.zeros((_N, 3))
    J[0, 0] = 1.0
    J[1, 1] = 1.0
    J[2, 2] = 1.0
    J[3, 0] = -1.0 / _C12_DENOM
    J[3, 1] = -12.0 / _C12_DENOM
    J[3, 2] = -1680.0 / _C12_DENOM
    
    f = x2_moment
    g = p2_moment
    N = norm_sq
    
    df_dc = 2.0 * _MX @ coeffs
    dg_dc = 2.0 * _MP @ coeffs
    dN_dc = 2.0 * coeffs * _NORM_FACTORS
    
    d_bound_dc = (g * df_dc - f * dg_dc) / (N**2) - 2.0 * f * g * dN_dc / (N**3)
    
    grad = J.T @ d_bound_dc
    
    return bound, grad

def _spherical_to_params(theta, phi):
    """Convert spherical coords to 3D params (unit sphere)."""
    st = np.sin(theta)
    return np.array([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])

def construct():
    start_time = time.time()
    
    # Seed value
    seed_params = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    seed_score = _compute_bound(seed_params)
    best_score = seed_score
    best_coeffs = seed_params.copy()
    
    def _time_ok():
        return time.time() < _DEADLINE - 5.0
    
    # Phase 1: Grid search on the sphere (theta, phi)
    # Since objective is homogeneous degree 0, we can fix norm=1
    # theta in [0, pi], phi in [0, 2*pi)
    # Use moderate grid first
    n_theta = 40
    n_phi = 80
    thetas = np.linspace(0.01, np.pi - 0.01, n_theta)
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    
    grid_best = np.inf
    grid_best_tp = (0, 0)
    
    for i, th in enumerate(thetas):
        if not _time_ok():
            break
        for j, ph in enumerate(phis):
            params = _spherical_to_params(th, ph)
            val = _compute_bound(params)
            if val < grid_best:
                grid_best = val
                grid_best_tp = (th, ph)
    
    if grid_best < best_score:
        best_score = grid_best
        best_coeffs = _spherical_to_params(*grid_best_tp)
    
    # Phase 2: Brent refinement around grid minimum
    if _time_ok():
        th0, ph0 = grid_best_tp
        # Refine theta for fixed phi
        try:
            res_th = brent(lambda th: _compute_bound(_spherical_to_params(th, ph0)),
                          bracket=(max(0.01, th0-0.3), min(np.pi-0.01, th0+0.3)),
                          xtol=1e-12, maxiter=100)
            th_opt = res_th[0]
            # Refine phi for fixed theta
            res_ph = brent(lambda ph: _compute_bound(_spherical_to_params(th_opt, ph)),
                          bracket=((ph0-0.3+2*np.pi)%(2*np.pi), (ph0+0.3)%(2*np.pi)) if False else (max(0, ph0-0.3), min(2*np.pi, ph0+0.3)),
                          xtol=1e-12, maxiter=100)
            ph_opt = res_ph[0]
            params_brent = _spherical_to_params(th_opt, ph_opt)
            val_brent = _compute_bound(params_brent)
            if val_brent < best_score:
                best_score = val_brent
                best_coeffs = params_brent
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B polish from multiple good starting points
    starts = [best_coeffs.copy()]
    # Add a few diverse starts
    for th, ph in [(np.pi/3, np.pi/4), (2*np.pi/3, 3*np.pi/4), (np.pi/2, 0), (np.pi/4, np.pi/2)]:
        starts.append(_spherical_to_params(th, ph))
    # Original seed
    starts.append(seed_params.copy())
    
    for x0 in starts:
        if not _time_ok():
            break
        try:
            result = minimize(_compute_bound_and_grad, x0, method='L-BFGS-B',
                             jac=True, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    if _time_ok():
        try:
            result = minimize(_compute_bound_and_grad, best_coeffs, method='L-BFGS-B',
                             jac=True, options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-15})
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return best_coeffs
