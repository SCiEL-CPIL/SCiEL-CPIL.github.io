# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize, differential_evolution

_K = 10
_NPHI = 130
_NTH = 70
_NPTS = _NTH * _NPHI
_NC = 2 * _K + 1

# Pre-compute quadrature and basis functions
def _build_quadrature():
    x, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI
    W = np.outer(w, np.full(_NPHI, dphi))

    Y = np.zeros((_NTH, _NPHI, _NC))
    for idx, m in enumerate(range(-_K, _K + 1)):
        am = abs(m)
        # Compute (10-am)! / (10+am)!
        denom = 1.0
        for n in range(_K - am + 1, _K + am + 1):
            denom *= n
        fact_ratio = 1.0 / denom
        Nm = np.sqrt((2 * _K + 1) / (4.0 * np.pi) * fact_ratio)
        
        # Condon-Shortley phase is included in lpmv
        P = lpmv(am, _K, x)
        
        if m == 0:
            Y[:, :, idx] = Nm * P[:, None]
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.sin(am * phi)[None, :]
            
    return Y, W

_Y, _W = _build_quadrature()
_Wflat = _W.ravel()
_Yflat = _Y.reshape(_NPTS, _NC)

def _objective(c):
    """R^4 = <f^4> / <f^2>^2"""
    f = _Yflat @ c
    f2 = f * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return 1e30
    f4 = f2 * f2
    l4sq = float(np.dot(_Wflat, f4))
    return l4sq / (l2sq * l2sq)

def _gradient(c):
    """Analytic gradient of R^4"""
    f = _Yflat @ c
    f2 = f * f
    f3 = f2 * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return np.zeros_like(c)
    l4sq = float(np.dot(_Wflat, f2 * f2))
    
    # d/dc ( <f^4> ) = 4 Y^T (W f^3)
    # d/dc ( <f^2>^2 ) = 4 L2 Y^T (W f)
    # dJ/dc = 4/B^2 Y^T (W f^3) - 4A/B^3 Y^T (W f)
    
    wf3 = _Wflat * f3
    wf = _Wflat * f
    grad_f3 = _Yflat.T @ wf3
    grad_f = _Yflat.T @ wf
    
    B = l2sq
    A = l4sq
    grad = (4.0 / (B * B)) * grad_f3 - (4.0 * A / (B * B * B)) * grad_f
    return grad

def _polish(c0, deadline):
    """Refine c0 using SLSQP with sphere constraint, then L-BFGS-B if needed"""
    c = np.asarray(c0, dtype=np.float64).copy()
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return None
    
    # Normalize initial guess
    c = c / nrm
    
    def fun(x):
        return _objective(x)
        
    def jac(x):
        return _gradient(x)
        
    best_c = c.copy()
    best_val = fun(c)
    
    # 1. SLSQP with sphere constraint
    # This handles the geometry better than raw L-BFGS-B for scale-invariant problems
    try:
        constraint = {'type': 'eq', 'fun': lambda x: np.dot(x, x) - 1.0, 'jac': lambda x: 2.0 * x}
        res = minimize(fun, c, jac=jac, method='SLSQP',
                       constraints=[constraint],
                       options={'maxiter': 500, 'ftol': 1e-14, 'disp': False})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x.copy()
            # Normalize again to be safe
            best_c /= np.linalg.norm(best_c)
    except Exception:
        pass
        
    # 2. Optional: L-BFGS-B polish on the normalized vector?
    # SLSQP usually gets very close. A short L-BFGS-B run might help if we project the gradient.
    # However, L-BFGS-B doesn't support the constraint. 
    # We can do a simple gradient descent step on the sphere if time permits.
    # But SLSQP is generally sufficient for this precision.
    
    # Let's try a quick Nelder-Mead polish on the raw vector, but normalize inside?
    # No, Nelder-Mead doesn't normalize.
    # Let's stick to SLSQP result. It's robust.
    
    nrm = np.linalg.norm(best_c)
    if nrm < 1e-14:
        return None
    # Return normalized c
    return best_c / nrm, float(best_val)

def rudin_problem10() -> np.ndarray:
    start = time.time()
    deadline = start + 625.0  # 5s safety margin
    
    best_c = None
    best_val = np.inf
    
    # 1. Differential Evolution in full 21D space
    # The current best score comes from this approach. We use Sobol initialization
    # and a moderate population to balance exploration and exploitation.
    bounds = [(-10.0, 10.0) for _ in range(_NC)]
    
    try:
        if time.time() < deadline - 100.0:
            de_result = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=300,
                popsize=15,
                tol=1e-12,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=False,
                init='sobol'
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_c = de_result.x.copy()
    except Exception:
        pass

    # 2. Polish the DE result
    if best_c is not None:
        polished = _polish(best_c, deadline)
        if polished:
            c, val = polished
            if val < best_val:
                best_val = val
                best_c = c

    # 3. Additional restarts if time permits
    # We use structured starts (pure modes) and random restarts to cover different basins.
    rng = np.random.default_rng(123)
    
    structured_starts = []
    for m in range(-_K, _K + 1):
        c0 = np.zeros(_NC)
        c0[m + _K] = 1.0
        structured_starts.append(c0)
    
    # Add symmetry-aware combinations: pairs of modes with same |m|
    for m in range(1, _K + 1):
        idx_pos = m + _K
        idx_neg = -m + _K
        for angle in [0.0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]:
            c0 = np.zeros(_NC)
            c0[idx_pos] = np.cos(angle)
            c0[idx_neg] = np.sin(angle)
            structured_starts.append(c0)
    
    # Add combinations of multiple mode pairs
    for m1 in range(1, _K + 1, 2):
        for m2 in range(m1 + 1, _K + 1, 2):
            idx_pos1 = m1 + _K
            idx_neg1 = -m1 + _K
            idx_pos2 = m2 + _K
            idx_neg2 = -m2 + _K
            for a1 in [0.0, np.pi/4, np.pi/2]:
                for a2 in [0.0, np.pi/4, np.pi/2]:
                    c0 = np.zeros(_NC)
                    c0[idx_pos1] = np.cos(a1) * 0.7
                    c0[idx_neg1] = np.sin(a1) * 0.7
                    c0[idx_pos2] = np.cos(a2) * 0.7
                    c0[idx_neg2] = np.sin(a2) * 0.7
                    nrm_c = np.linalg.norm(c0)
                    if nrm_c > 1e-14:
                        c0 /= nrm_c
                        structured_starts.append(c0)
    
    # Also add some random starts
    for _ in range(20):
        c0 = rng.standard_normal(_NC)
        c0 /= np.linalg.norm(c0)
        structured_starts.append(c0)
    
    # Shuffle to mix structured and random
    rng.shuffle(structured_starts)
    
    for c0 in structured_starts:
        if time.time() >= deadline - 30.0:
            break
            
        polished = _polish(c0, deadline)
        if polished:
            c, val = polished
            if val < best_val:
                best_val = val
                best_c = c

    # 4. Final high-precision polish on the best candidate
    if best_c is not None and time.time() < deadline - 10.0:
        # Multiple final polish attempts with different methods
        for _attempt in range(3):
            if time.time() >= deadline - 10.0:
                break
            final_polished = _polish(best_c, deadline)
            if final_polished:
                c, val = final_polished
                if val < best_val:
                    best_val = val
                    best_c = c
                else:
                    break
    
    # 5. Final safety
    if best_c is None:
        # Fallback: pure m=10 mode
        best_c = np.zeros(_NC, dtype=np.float64)
        best_c[_K] = 1.0
        
    best_c = np.asarray(best_c, dtype=np.float64)
    
    # Validation checks
    if not np.all(np.isfinite(best_c)):
        best_c = np.zeros(_NC, dtype=np.float64)
        best_c[_K] = 1.0
        
    mx = float(np.max(np.abs(best_c)))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)
        
    return best_c

# EVOLVE-BLOCK-END
