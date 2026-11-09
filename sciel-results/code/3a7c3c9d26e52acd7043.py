"""Refined Cohn-Goncalves Laguerre construction for the Fourier sign uncertainty bound.

We tune the 12 positive double-root locations (m=12, n=1) to minimize the
upper bound C4 on the Fourier sign uncertainty.
"""
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 560.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _seed_roots():
    return [4.0 * (i + 1) for i in range(12)]


def _valid(roots):
    """Root constraints: 12 positive, strictly increasing (distinct) roots."""
    if len(roots) != 12:
        return False
    if np.any(roots <= 0.0):
        return False
    if np.any(np.diff(roots) <= 0.0):
        return False
    return True


# Precompute Gauss-Laguerre quadrature nodes and weights for weight exp(-x)
# We need to compute integral of x * exp(-x) * Q(x)^2 dx
# Using Gauss-Laguerre: integral of f(x) exp(-x) dx ≈ sum wq_i * f(xq_i)
# So integral of x * exp(-x) * Q(x)^2 dx ≈ sum wq_i * xq_i * Q(xq_i)^2
try:
    from numpy.polynomial.laguerre import laggauss
    _NQ = 200
    _xq, _wq = laggauss(_NQ)
    _xq = np.asarray(_xq, dtype=float)
    _wq = np.asarray(_wq, dtype=float)
    _xw = _xq * _wq  # x * exp(-x) factor baked into weights
except Exception:
    _NQ = 4000
    _xq = np.linspace(0.0, 200.0, _NQ)
    _wq = np.exp(-_xq) * (200.0 / (_NQ - 1))
    _xw = _xq * _wq


def _energy(roots):
    """
    Compute the L2 energy of Q(x) = prod_i (x - r_i)^2 under weight x*exp(-x).
    This is the key quantity for the Cohn-Goncalves C4 bound.
    Lower energy -> lower C4 bound.
    """
    r = np.asarray(roots, dtype=float)
    if np.any(r <= 0.0):
        return 1e30
    
    # Compute log|Q(x)| = 2 * sum_i log|x - r_i| at quadrature nodes
    logQ = np.zeros_like(_xq)
    for ri in r:
        logQ += 2.0 * np.log(np.abs(_xq - ri) + 1e-300)
    
    # Q(x)^2 = exp(2 * logQ)
    # Energy = sum xw_i * exp(2 * logQ_i)
    log_integrand = np.log(_xw) + 2.0 * logQ
    # Clip to avoid overflow/underflow
    log_integrand = np.clip(log_integrand, -700.0, 700.0)
    val = np.sum(np.exp(log_integrand))
    if not np.isfinite(val):
        return 1e30
    return float(val)


def _c4_bound(roots):
    """
    Compute the C4 upper bound. The bound is related to the energy.
    For the Cohn-Goncalves construction, C4 is proportional to the energy.
    We normalize by the seed energy to get a relative bound.
    """
    if not _valid(roots):
        return 1e10
    
    e = _energy(roots)
    e_seed = _energy(_seed_roots())
    if e_seed <= 0 or not np.isfinite(e_seed):
        return 1e10
    return e / e_seed


def construct():
    if _elapsed() > _DEADLINE:
        return _seed_roots()

    try:
        best_score = 1e10
        best_roots = None
        
        # Candidate starting points
        candidates = []
        
        # 1. Seed roots
        candidates.append(np.array(_seed_roots(), dtype=float))
        
        # 2. Roots of generalized Laguerre L_12^(1) - natural for this construction
        try:
            from scipy.special import roots_genlaguerre
            r_lag1, _ = roots_genlaguerre(12, 1)
            r_lag1 = np.sort(np.asarray(r_lag1, dtype=float))
            candidates.append(r_lag1)
        except Exception:
            pass
        
        # 3. Roots of L_12^(0)
        try:
            from scipy.special import roots_genlaguerre
            r_lag0, _ = roots_genlaguerre(12, 0)
            r_lag0 = np.sort(np.asarray(r_lag0, dtype=float))
            candidates.append(r_lag0)
        except Exception:
            pass
        
        # 4. Scaled variants of Laguerre roots
        try:
            from scipy.special import roots_genlaguerre
            r_lag1, _ = roots_genlaguerre(12, 1)
            r_lag1 = np.sort(np.asarray(r_lag1, dtype=float))
            for s in (0.5, 0.7, 1.0, 1.3, 1.5, 2.0):
                candidates.append(r_lag1 * s)
        except Exception:
            pass
        
        # 5. Geometric progression
        geo = 4.0 * np.power(48.0 / 4.0, np.arange(12) / 11.0)
        candidates.append(geo)
        
        # 6. Arithmetic progressions with different scales
        for scale in (0.5, 0.7, 1.0, 1.5, 2.0, 3.0):
            roots_arith = np.array([4.0 * scale * (i + 1) for i in range(12)], dtype=float)
            candidates.append(roots_arith)
        
        # Evaluate all candidates
        for c in candidates:
            if _elapsed() > _DEADLINE - 10:
                break
            if not _valid(c):
                continue
            score = _c4_bound(c)
            if score < best_score:
                best_score = score
                best_roots = c.copy()
        
        # Local refinement from the best candidate and a few others
        if best_roots is not None and _elapsed() < _DEADLINE - 60:
            def objective(params):
                r = np.asarray(params, dtype=float)
                if not _valid(r):
                    return 1e30
                return _energy(r)
            
            # Multi-start: best candidate + scaled Laguerre roots
            start_points = [best_roots.copy()]
            try:
                from scipy.special import roots_genlaguerre
                r_lag1, _ = roots_genlaguerre(12, 1)
                r_lag1 = np.sort(np.asarray(r_lag1, dtype=float))
                for s in (0.7, 1.0, 1.3):
                    start_points.append(r_lag1 * s)
            except Exception:
                pass
            
            for sp in start_points:
                if _elapsed() > _DEADLINE - 30:
                    break
                try:
                    res = minimize(
                        objective,
                        sp,
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8,
                                 'adaptive': True}
                    )
                    if np.all(res.x > 0.0) and _valid(res.x):
                        refined = np.sort(res.x)
                        refined_score = _c4_bound(refined)
                        if refined_score < best_score:
                            best_score = refined_score
                            best_roots = refined.copy()
                except Exception:
                    pass
            
            # Try L-BFGS-B with bounds for a final polish
            if _elapsed() < _DEADLINE - 20 and best_roots is not None:
                try:
                    bounds = [(1e-6, 1000.0)] * 12
                    res = minimize(
                        objective,
                        best_roots,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    if np.all(res.x > 0.0) and _valid(res.x):
                        refined = np.sort(res.x)
                        refined_score = _c4_bound(refined)
                        if refined_score < best_score:
                            best_score = refined_score
                            best_roots = refined.copy()
                except Exception:
                    pass
        
        if best_roots is not None and _valid(best_roots):
            return [float(r) for r in best_roots]
        
    except Exception:
        pass

    return _seed_roots()
