# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.special as sp
from scipy.optimize import minimize

_K = 10
_NPHI = 130
_NTH = 70
_NPTS = _NTH * _NPHI
_NC = 2 * _K + 1

def _build_quadrature():
    """
    Precompute the matrix of real orthonormal spherical harmonics Y_m
    for l=10, m=-10..10 on the specified quadrature grid.
    Returns Y (N_theta * N_phi, 21) and weights W (N_theta * N_phi).
    """
    x, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI
    
    # Weights for the double integral: int_{-1}^1 dx int_0^{2pi} dphi
    # Discretized as sum_i sum_j w_i dphi f(x_i, phi_j)
    W = np.outer(w, np.full(_NPHI, dphi))
    
    Y = np.zeros((_NTH, _NPHI, _NC))
    for idx, m in enumerate(range(-_K, _K + 1)):
        am = abs(m)
        # N_m = sqrt((2K+1)/(4pi) * (K-am)!/(K+am)!)
        # Compute (K-am)!/(K+am)! efficiently
        denom = 1.0
        for n in range(_K - am + 1, _K + am + 1):
            denom *= n
        fact_ratio = 1.0 / denom
        Nm = np.sqrt((2 * _K + 1) / (4.0 * np.pi) * fact_ratio)
        
        P = sp.lpmv(am, _K, x)
        
        if m == 0:
            Y[:, :, idx] = Nm * P[:, None]
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.sin(am * phi)[None, :]
            
    return Y, W

# Precompute at module load to save time in the function call
_Y, _W = _build_quadrature()
_Wflat = _W.ravel()
_Yflat = _Y.reshape(_NPTS, _NC)

def _objective(c):
    """
    Compute R^4 = ||f||_4^4 / ||f||_2^4.
    Minimizing this is equivalent to minimizing R = ||f||_4 / ||f||_2.
    """
    f = _Yflat @ c
    f2 = f * f
    S2 = np.dot(_Wflat, f2)
    if S2 <= 1e-30:
        return 1e30
    f4 = f2 * f2
    S4 = np.dot(_Wflat, f4)
    return S4 / (S2 * S2)

def _gradient(c):
    """
    Analytic gradient of R^4 = S4 / S2^2.
    S2 = <f^2>, S4 = <f^4>.
    d(S4/S2^2) = (4<f^3 Y> S2^2 - S4 * 2 S2 * 2 <f Y>) / S2^4
               = 4<f^3 Y> / S2^2 - 8 S4 <f Y> / S2^3
    """
    f = _Yflat @ c
    f2 = f * f
    f3 = f2 * f
    S2 = np.dot(_Wflat, f2)
    if S2 <= 1e-30:
        return np.zeros_like(c)
    S4 = np.dot(_Wflat, f2 * f2)
    
    term1 = _Yflat.T @ (_Wflat * f3)
    term2 = _Yflat.T @ (_Wflat * f)
    
    grad = (4.0 / (S2 * S2)) * term1 - (8.0 * S4 / (S2 * S2 * S2)) * term2
    return grad

def _optimize_from(c0, deadline):
    """
    Optimize from a given starting point.
    Returns (best_c, best_val) or (None, inf) if failed.
    """
    c = np.asarray(c0, dtype=np.float64).copy()
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return None, np.inf
    c = c / nrm
    
    best_c = c.copy()
    best_val = _objective(c)
    
    # Stage 1: L-BFGS-B
    if time.time() < deadline - 10.0:
        try:
            res = minimize(
                _objective, c, jac=_gradient, method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 1000}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    # Stage 2: Nelder-Mead polish
    if time.time() < deadline - 5.0:
        try:
            res2 = minimize(
                _objective, best_c, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_c = res2.x.copy()
        except Exception:
            pass

    # Stage 3: High-precision polish with BFGS (unconstrained, using analytic gradient)
    # This can help if the previous stages stopped slightly early due to tolerance.
    if time.time() < deadline - 2.0:
        try:
            res3 = minimize(
                _objective, best_c, jac=_gradient, method='BFGS',
                options={'maxiter': 100, 'gtol': 1e-12, 'disp': False}
            )
            if res3.fun < best_val:
                best_val = res3.fun
                best_c = res3.x.copy()
        except Exception:
            pass

    # Normalize result
    nrm_final = np.linalg.norm(best_c)
    if nrm_final > 1e-14:
        best_c = best_c / nrm_final
        # Recalculate value on normalized c to be sure
        best_val = _objective(best_c)
        
    return best_c, best_val

def _make_seeds(rng):
    """
    Generate diverse initial seeds for optimization.
    """
    seeds = []
    
    # 1. All ones (seed program)
    c = np.ones(_NC)
    c /= np.linalg.norm(c)
    seeds.append(c)
    
    # 2. m=0 only
    c = np.zeros(_NC)
    c[_K] = 1.0
    seeds.append(c)
    
    # 3. m=10 cos only
    c = np.zeros(_NC)
    c[_K + 10] = 1.0
    seeds.append(c)
    
    # 4. m=10 sin only
    c = np.zeros(_NC)
    c[_K - 10] = 1.0
    seeds.append(c)
    
    # 5. Even m only
    c = np.zeros(_NC)
    for m in range(-_K, _K + 1, 2):
        c[m + _K] = rng.standard_normal()
    nrm = np.linalg.norm(c)
    if nrm > 1e-14:
        c /= nrm
        seeds.append(c)
        
    # 6. Odd m only
    c = np.zeros(_NC)
    for m in range(-_K + 1, _K + 1, 2):
        c[m + _K] = rng.standard_normal()
    nrm = np.linalg.norm(c)
    if nrm > 1e-14:
        c /= nrm
        seeds.append(c)
        
    # 7. Symmetric (c_m = c_-m)
    c = np.zeros(_NC)
    c[_K] = rng.standard_normal()
    for m in range(1, _K + 1):
        v = rng.standard_normal()
        c[m + _K] = v
        c[-m + _K] = v
    nrm = np.linalg.norm(c)
    if nrm > 1e-14:
        c /= nrm
        seeds.append(c)
        
    # 8. Antisymmetric (c_m = -c_-m, c_0=0)
    c = np.zeros(_NC)
    for m in range(1, _K + 1):
        v = rng.standard_normal()
        c[m + _K] = v
        c[-m + _K] = -v
    nrm = np.linalg.norm(c)
    if nrm > 1e-14:
        c /= nrm
        seeds.append(c)
        
    # 9. Random Gaussian
    c = rng.standard_normal(_NC)
    nrm = np.linalg.norm(c)
    if nrm > 1e-14:
        c /= nrm
        seeds.append(c)
        
    # 10. Random with low-order bias
    c = rng.standard_normal(_NC)
    c[_K] += 0.5
    c[_K + 10] += 0.3
    c[_K - 10] += 0.3
    nrm = np.linalg.norm(c)
    if nrm > 1e-14:
        c /= nrm
        seeds.append(c)
        
    # 11. Random with high-order bias
    c = rng.standard_normal(_NC)
    c[_K + 10] += 0.5
    c[_K - 10] += 0.5
    nrm = np.linalg.norm(c)
    if nrm > 1e-14:
        c /= nrm
        seeds.append(c)
        
    # 12. m=0 and m=10 mix
    c = np.zeros(_NC)
    c[_K] = rng.standard_normal()
    c[_K + 10] = rng.standard_normal()
    c[_K - 10] = rng.standard_normal()
    nrm = np.linalg.norm(c)
    if nrm > 1e-14:
        c /= nrm
        seeds.append(c)
        
    # 13. m=5 mix
    c = np.zeros(_NC)
    c[_K + 5] = rng.standard_normal()
    c[_K - 5] = rng.standard_normal()
    c[_K] = rng.standard_normal()
    nrm = np.linalg.norm(c)
    if nrm > 1e-14:
        c /= nrm
        seeds.append(c)
        
    # 14. Alternating signs
    c = np.zeros(_NC)
    for i in range(_NC):
        c[i] = (-1) ** i
    nrm = np.linalg.norm(c)
    if nrm > 1e-14:
        c /= nrm
        seeds.append(c)
        
    return seeds

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0
    
    rng = np.random.default_rng(42)
    
    seeds = _make_seeds(rng)
    
    best_c = None
    best_val = np.inf
    
    # First pass: optimize from structured seeds
    for c0 in seeds:
        if time.time() > deadline - 30.0:
            break
        c, val = _optimize_from(c0, deadline)
        if c is not None and val < best_val:
            best_val = val
            best_c = c.copy()
            
    # Second pass: random restarts with remaining time
    # Each optimization takes ~0.5-2s. We can fit many more.
    while time.time() < deadline - 10.0:
        c0 = rng.standard_normal(_NC)
        nrm = np.linalg.norm(c0)
        if nrm < 1e-14:
            continue
        c0 /= nrm
        
        c, val = _optimize_from(c0, deadline)
        if c is not None and val < best_val:
            best_val = val
            best_c = c.copy()
            
    # Fallback
    if best_c is None:
        best_c = np.ones(_NC, dtype=np.float64)
        best_c /= np.linalg.norm(best_c)
        
    # Final validation and scaling
    best_c = np.asarray(best_c, dtype=np.float64)
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(_NC, dtype=np.float64)
        best_c /= np.linalg.norm(best_c)
        
    mx = float(np.max(np.abs(best_c)))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    elif mx > 1e5:
        best_c = best_c * (1e5 / mx)
        
    return best_c

# EVOLVE-BLOCK-END
