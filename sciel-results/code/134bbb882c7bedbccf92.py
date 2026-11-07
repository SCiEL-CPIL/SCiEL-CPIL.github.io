"""Structural change: direct root-space search with feasibility projection."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

# Precompute the linear map from (c0,c1,c2) to Q coefficients
# Q(z) = a12*z^5 + a10*z^4 + a8*z^3 + a6*z^2 + a4*z + a2
# c3 = -(c0 + 12*c1 + 1680*c2) / 2032128

def _Q_from_params(c0, c1, c2):
    """Return Q coefficients [a2, a4, a6, a8, a10, a12] from Hermite params."""
    c3 = -(c0 + 12.0*c1 + 1680.0*c2) / 2032128.0
    a2  = -48.0*c1 - 13440.0*c2 - 1290240.0*c3
    a4  =  16.0*c1 + 11520.0*c2 + 2123328.0*c3
    a6  = -3072.0*c2 - 1548288.0*c3
    a8  =  256.0*c2 + 516096.0*c3
    a10 = -73728.0*c3
    a12 =  4096.0*c3
    return np.array([a2, a4, a6, a8, a10, a12])

def _value_from_Q(Q):
    """Compute value = z*/(2*pi) where z* is largest positive sign-changing root of Q."""
    Q = Q.copy()
    if Q[5] < 0:
        Q = -Q
    coeffs = np.array([Q[5], Q[4], Q[3], Q[2], Q[1], Q[0]])
    # Trim leading zeros
    while len(coeffs) > 1 and abs(coeffs[0]) < 1e-14:
        coeffs = coeffs[1:]
    if len(coeffs) <= 1:
        return 1e10
    try:
        roots = np.roots(coeffs)
    except:
        return 1e10
    best_z = 0.0
    for rz in roots:
        if abs(rz.imag) > 1e-9:
            continue
        zr = rz.real
        if zr <= 1e-12:
            continue
        # Check sign change
        eps = 1e-7 * max(1.0, abs(zr))
        qm = np.polyval(coeffs, zr - eps)
        qp = np.polyval(coeffs, zr + eps)
        if qm * qp < 0 and zr > best_z:
            best_z = zr
    if best_z == 0.0:
        # No sign-changing positive root; use largest positive root as fallback
        pos_roots = [rz.real for rz in roots if abs(rz.imag) < 1e-9 and rz.real > 1e-12]
        if pos_roots:
            best_z = max(pos_roots)
        else:
            return 1e10
    return best_z / (2.0 * np.pi)

def _objective(params):
    try:
        Q = _Q_from_params(params[0], params[1], params[2])
        v = _value_from_Q(Q)
        return v if np.isfinite(v) else 1e10
    except:
        return 1e10

def construct():
    t0 = time.time()
    deadline = t0 + 165.0  # 45s safety margin
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _objective(seed)
    best_params = seed.copy()
    
    # Phase 1: Differential evolution for global exploration
    if time.time() < deadline - 60.0:
        try:
            bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.1, 0.1)]
            de_result = differential_evolution(
                _objective, bounds,
                seed=42, maxiter=200, tol=1e-12,
                popsize=30, mutation=(0.3, 1.0), recombination=0.7,
                polish=False, updating='immediate',
                workers=1
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Multi-start local refinement
    np.random.seed(999)
    starts = [best_params.copy()]
    for scale in [0.001, 0.01, 0.05, 0.1, 0.5]:
        for _ in range(6):
            starts.append(best_params + np.random.randn(3) * scale)
    
    for start in starts:
        if time.time() > deadline - 30.0:
            break
        try:
            res = minimize(_objective, start, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-13,
                                   'fatol': 1e-15, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except:
            pass
    
    # Phase 3: High-precision polish
    if time.time() < deadline - 15.0:
        for _ in range(3):
            try:
                res = minimize(_objective, best_params, method='Nelder-Mead',
                              options={'maxiter': 20000, 'xatol': 1e-15,
                                       'fatol': 1e-17, 'adaptive': True})
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x.copy()
            except:
                break
    
    # Phase 4: Final L-BFGS attempt
    if time.time() < deadline - 5.0:
        try:
            res = minimize(_objective, best_params, method='L-BFGS-B',
                          options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except:
            pass
    
    return best_params
# EVOLVE-BLOCK-END
