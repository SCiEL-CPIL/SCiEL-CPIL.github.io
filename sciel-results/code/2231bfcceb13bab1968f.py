# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

_T0 = time.time()
_DEADLINE = 195.0  # Safety margin below the 210s kill

# ---------- Precompute Hermite Polynomial Coefficients (low->high) ----------
# Physicists' Hermite polynomials:
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12
# H_8(x) = 256x^8 - 4480x^6 + 26880x^4 - 67200x^2 + 40320
# H_12(x) = 4096x^12 - 147456x^10 + 2903040x^8 - 29030400x^6 + 145152000x^4 - 393216000x^2 + 1032160

_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 40320.0
_H12_0 = 1032160.0

# Coefficients low->high for each polynomial
_c0v = np.array([1.0])
_c1v = np.array([12.0, 0.0, -48.0, 0.0, 16.0])
_c2v = np.array([40320.0, 0.0, -67200.0, 0.0, 26880.0, 0.0, -4480.0, 0.0, 256.0])
_c3v = np.array([1032160.0, 0.0, -393216000.0, 0.0, 145152000.0, 0.0, -29030400.0,
                 0.0, 2903040.0, 0.0, -147456.0, 0.0, 4096.0])

_deg = 12

def _g_coeffs(c0, c1, c2):
    """Compute g(x) coefficients (low->high, degree 0 to 12)."""
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    g = np.zeros(_deg + 1)
    g += c0 * np.pad(_c0v, (0, _deg + 1 - len(_c0v)))
    g += c1 * np.pad(_c1v, (0, _deg + 1 - len(_c1v)))
    g += c2 * np.pad(_c2v, (0, _deg + 1 - len(_c2v)))
    g += c3 * np.pad(_c3v, (0, _deg + 1 - len(_c3v)))
    return g

def _q_in_z(c0, c1, c2):
    """
    Compute q(x)/x^2 as a polynomial in z=x^2.
    Returns coefficients of polynomial in z (low->high).
    q(x) = g(x)/x^2, and since g is even, q has only even powers of x.
    Let z = x^2, then q = sum a_k z^k where a_k = g[2k+2]
    """
    g = _g_coeffs(c0, c1, c2)
    # q(x) = g(x)/x^2, so q[2k] = g[2k+2] for k=0..5
    # In terms of z=x^2: q(z) = sum_{k=0}^{5} g[2k+2] * z^k
    return g[2::_deg + 1 - 1]  # g[2], g[4], g[6], g[8], g[10], g[12]

def _smallest_positive_simple_root(z_coeffs):
    """
    Return the smallest positive real simple (sign-changing) root of z_coeffs (low->high).
    If none exists, return +inf.
    """
    zc = z_coeffs.copy()
    # Remove leading near-zeros
    while len(zc) > 1 and abs(zc[-1]) < 1e-30:
        zc = zc[:-1]
    
    if len(zc) < 2:
        return np.inf
    
    try:
        roots = np.roots(zc)
    except Exception:
        return np.inf
    
    best = np.inf
    for rt in roots:
        # Check if real
        if abs(rt.imag) > 1e-9 * max(1.0, abs(rt.real)):
            continue
        zr = float(rt.real)
        if zr <= 1e-12:
            continue
        
        # Check it's a simple root (sign change): derivative should be nonzero
        d = np.polyder(np.flip(zc))  # high->low for np.polyval
        dval = np.polyval(d, zr)
        if abs(dval) < 1e-9:
            continue
        
        # Confirm it's actually a root
        qval = np.polyval(np.flip(zc), zr)
        if abs(qval) > 1e-6 * max(1.0, abs(zr)):
            continue
        
        if zr < best:
            best = zr
    
    return best

def _value(c0, c1, c2):
    """Compute value = r^2 / (2*pi) where r is smallest positive sign-changing root."""
    zc = _q_in_z(c0, c1, c2)
    r = _smallest_positive_simple_root(zc)
    if not np.isfinite(r) or r <= 0:
        return np.inf
    return r * r / (2.0 * np.pi)

_BIG = 1e12

def _obj(params):
    """Objective function wrapper with exception handling."""
    try:
        v = _value(float(params[0]), float(params[1]), float(params[2]))
    except Exception:
        return _BIG
    if not np.isfinite(v):
        return _BIG
    return min(v, _BIG)

def _seed():
    return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

def _local_refine(x0, max_iter=500):
    """Nelder-Mead local refinement with tight tolerances."""
    try:
        res = minimize(_obj, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-14,
                                'adaptive': True})
        if np.isfinite(res.fun) and res.fun < _BIG:
            return res.x, res.fun
    except Exception:
        pass
    
    # Try L-BFGS-B as backup
    try:
        res2 = minimize(_obj, x0, method='L-BFGS-B',
                        options={'maxiter': max_iter, 'gtol': 1e-12, 'ftol': 1e-14})
        if np.isfinite(res2.fun) and res2.fun < _BIG:
            return res2.x, res2.fun
    except Exception:
        pass
    
    return x0, _obj(x0)

def _polish_high_precision(x0, max_iter=2000):
    """Very high-precision polish pass."""
    try:
        res = minimize(_obj, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-15, 'fatol': 1e-18,
                                'adaptive': True})
        if np.isfinite(res.fun) and res.fun < _BIG:
            return res.x, res.fun
    except Exception:
        pass
    return x0, _obj(x0)

def construct():
    global _T0
    try:
        rng = np.random.default_rng(42)
        best_x = _seed().copy()
        best_v = _obj(best_x)
        
        if not np.isfinite(best_v):
            best_v = _BIG
        
        # Phase 1: Multi-start local search with diverse initializations
        starts = [best_x.copy()]
        
        # Add random points in various scales
        for scale in [0.5, 1.0, 2.0, 5.0, 10.0]:
            for _ in range(100):
                if time.time() - _T0 > _DEADLINE - 50:
                    break
                p = rng.uniform(-scale, scale, size=3)
                starts.append(p)
        
        # Add points that might create small roots (push coefficients to extremes)
        for a in [0.1, 0.5, 1, 2, 5, 10, 20]:
            for _ in range(50):
                if time.time() - _T0 > _DEADLINE - 50:
                    break
                p = rng.uniform(-a, a, size=3)
                starts.append(p)
        
        # Evaluate all starts and keep top candidates
        scored = []
        for p in starts:
            v = _obj(p)
            scored.append((v, p))
        
        scored.sort(key=lambda t: t[0])
        
        # Take top K distinct candidates for local refinement
        K = min(80, len(scored))
        top = scored[:K]
        
        seen = set()
        cands = []
        for v, p in top:
            key = tuple(np.round(p, 4))
            if key not in seen:
                seen.add(key)
                cands.append((v, p))
        
        # Phase 2: Local refinement of top candidates
        for v, p in cands:
            if time.time() - _T0 > _DEADLINE - 20:
                break
            if v >= best_v:
                continue
            
            refined_x, refined_v = _local_refine(p, max_iter=300)
            if refined_v < best_v:
                best_v = refined_v
                best_x = refined_x.copy()
        
        # Phase 3: Differential evolution for global exploration
        if time.time() - _T0 < _DEADLINE - 40 and best_v < _BIG:
            bounds = [(-20.0, 20.0), (-20.0, 20.0), (-20.0, 20.0)]
            try:
                de_res = differential_evolution(
                    _obj, bounds, seed=42, maxiter=100, popsize=30,
                    tol=1e-8, polish=False, init='sobol'
                )
                if np.isfinite(de_res.fun) and de_res.fun < best_v:
                    best_v = de_res.fun
                    best_x = de_res.x.copy()
            except Exception:
                pass
        
        # Phase 4: High-precision polish
        if time.time() - _T0 < _DEADLINE - 10 and best_v < _BIG:
            polished_x, polished_v = _polish_high_precision(best_x, max_iter=3000)
            if polished_v < best_v:
                best_v = polished_v
                best_x = polished_x
        
        # Phase 5: Final neighborhood search
        if time.time() - _T0 < _DEADLINE - 5 and best_v < _BIG:
            for trial in range(30):
                if time.time() - _T0 > _DEADLINE - 2:
                    break
                perturbation = rng.uniform(-0.05, 0.05, size=3)
                x_trial = best_x + perturbation
                val_trial = _obj(x_trial)
                if val_trial < best_v:
                    best_v = val_trial
                    best_x = x_trial
                    # Quick polish
                    if time.time() - _T0 < _DEADLINE - 5:
                        px, pv = _local_refine(best_x, max_iter=500)
                        if pv < best_v:
                            best_v = pv
                            best_x = px
        
        # Final safety check
        if not np.isfinite(best_v) or best_v >= _BIG:
            best_x = _seed()
            best_v = _obj(best_x)
        
        return np.array([float(best_x[0]), float(best_x[1]), float(best_x[2])])
        
    except Exception:
        seed = _seed()
        return np.array([float(seed[0]), float(seed[1]), float(seed[2])])

# EVOLVE-BLOCK-END
