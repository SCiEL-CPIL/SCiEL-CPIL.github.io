# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

_T0 = time.time()
_DEADLINE = 195.0  # Safety margin below the 210s kill

# ---------- Precompute Hermite Polynomial Coefficients (hardcoded, low->high) ----------
# Physicists' Hermite polynomials:
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12
# H_8(x) = 256x^8 - 4480x^6 + 26880x^4 - 67200x^2 + 40320
# H_12(x) = 4096x^12 - 147456x^10 + 2903040x^8 - 29030400x^6 + 145152000x^4 - 393216000x^2 + 1032160

_H0_coeffs = np.zeros(13)
_H0_coeffs[0] = 1.0

_H4_coeffs = np.zeros(13)
_H4_coeffs[0] = 12.0
_H4_coeffs[2] = -48.0
_H4_coeffs[4] = 16.0

_H8_coeffs = np.zeros(13)
_H8_coeffs[0] = 40320.0
_H8_coeffs[2] = -67200.0
_H8_coeffs[4] = 26880.0
_H8_coeffs[6] = -4480.0
_H8_coeffs[8] = 256.0

_H12_coeffs = np.zeros(13)
_H12_coeffs[0] = 1032160.0
_H12_coeffs[2] = -393216000.0
_H12_coeffs[4] = 145152000.0
_H12_coeffs[6] = -29030400.0
_H12_coeffs[8] = 2903040.0
_H12_coeffs[10] = -147456.0
_H12_coeffs[12] = 4096.0

_H0_0 = _H0_coeffs[0]
_H4_0 = _H4_coeffs[0]
_H8_0 = _H8_coeffs[0]
_H12_0 = _H12_coeffs[0]

def _q_coeffs(c0, c1, c2):
    """
    Compute the coefficients of q(x) = g(x)/x^2.
    g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    c3 = -(c0*H0(0) + c1*H4(0) + c2*H8(0)) / H12(0)
    """
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    
    # g(x) coefficients (degree 0 to 12)
    g = c0 * _H0_coeffs + c1 * _H4_coeffs + c2 * _H8_coeffs + c3 * _H12_coeffs
    
    # q(x) = g(x) / x^2. Since g is even, g[1] is 0.
    # q coefficients are g[2], g[3], ..., g[12] shifted to degree 0 to 10.
    q_full = g[2:] 
    return q_full

def _smallest_positive_sign_changing_root(q_coeffs):
    """
    Find the smallest positive real root of the polynomial q_coeffs (low->high) 
    where the polynomial changes sign.
    """
    # Remove leading zeros to ensure correct degree for np.roots
    qc = q_coeffs.copy()
    while len(qc) > 1 and abs(qc[-1]) < 1e-12:
        qc = qc[:-1]
    
    if len(qc) < 2:
        return np.inf
        
    try:
        roots = np.roots(qc)
    except Exception:
        return np.inf
        
    best_r = np.inf
    for rt in roots:
        # Check if real
        if abs(rt.imag) > 1e-9 * max(1.0, abs(rt.real)):
            continue
        zr = float(rt.real)
        if zr <= 1e-12:
            continue
            
        # Check sign change via derivative non-zero (simple root)
        d_coeffs = np.polyder(np.flip(qc))
        d_val = np.polyval(d_coeffs, zr)
        
        # If derivative is close to zero, it might be a double root (no sign change)
        if abs(d_val) < 1e-8 * max(1.0, abs(zr)):
            continue
            
        # Verify the root is actually a root (residual check)
        q_val = np.polyval(np.flip(qc), zr)
        if abs(q_val) > 1e-6 * max(1.0, abs(zr)):
            continue
            
        if zr < best_r:
            best_r = zr
            
    return best_r

def _objective(params):
    """
    Objective function for optimization.
    Returns value = r^2 / (2*pi). Lower is better.
    """
    c0, c1, c2 = params
    try:
        qc = _q_coeffs(c0, c1, c2)
        r = _smallest_positive_sign_changing_root(qc)
        if not np.isfinite(r) or r <= 0:
            return 1e12
        return r * r / (2.0 * np.pi)
    except Exception:
        return 1e12

def _seed():
    return np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])

def construct():
    global _T0
    try:
        seed = _seed()
        best_val = _objective(seed)
        best_x = seed.copy()
        
        # 1. Differential Evolution for Global Search
        bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
        
        if time.time() - _T0 < _DEADLINE - 40:
            try:
                res = differential_evolution(
                    _objective, 
                    bounds, 
                    seed=42, 
                    maxiter=100, 
                    popsize=25, 
                    tol=1e-10,
                    polish=False
                )
                if np.isfinite(res.fun) and res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # 2. Local Refinement (Nelder-Mead) on the best candidate
        if time.time() - _T0 < _DEADLINE - 15 and best_val < 1e10:
            try:
                res_nm = minimize(
                    _objective, 
                    best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': 800, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
                )
                if np.isfinite(res_nm.fun) and res_nm.fun < best_val:
                    best_val = res_nm.fun
                    best_x = res_nm.x.copy()
            except Exception:
                pass
        
        # 3. Final high-precision polish
        if time.time() - _T0 < _DEADLINE - 5 and best_val < 1e10:
            try:
                res_polish = minimize(
                    _objective, 
                    best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
                )
                if np.isfinite(res_polish.fun) and res_polish.fun < best_val:
                    best_val = res_polish.fun
                    best_x = res_polish.x.copy()
            except Exception:
                pass
        
        # 4. Final Safety Check
        if not np.isfinite(best_val) or best_val >= 1e10:
            best_x = seed
            best_val = _objective(seed)
        
        return np.array([float(best_x[0]), float(best_x[1]), float(best_x[2])])
        
    except Exception:
        seed = _seed()
        return np.array([float(seed[0]), float(seed[1]), float(seed[2])])
# EVOLVE-BLOCK-END
