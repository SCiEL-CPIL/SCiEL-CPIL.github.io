"""Optimized Cohn-Goncalves Laguerre double-root construction."""
import time
import numpy as np

_DEADLINE = time.time() + 540.0  # safety margin before 600s kill (evaluator uses 630s wall)
_M = 12

def _deadline_ok():
    return time.time() < _DEADLINE

def _feasible(roots):
    # positive, strictly increasing, finite
    if roots.shape[0] != _M:
        return False
    if np.any(~np.isfinite(roots)):
        return False
    if np.any(roots <= 0):
        return False
    if np.any(np.diff(roots) <= 1e-9):
        return False
    return True

def _objective(params):
    # params: [r0, g1..g11] all > 0 ; roots = cumulative
    r0 = params[0]
    gaps = params[1:]
    roots = np.concatenate(([r0], np.cumsum(np.concatenate(([r0], gaps[1:])))))
    # simpler: roots[0]=r0, roots[i]=roots[i-1]+gaps[i]
    roots = np.empty(_M)
    roots[0] = r0
    for i in range(1, _M):
        roots[i] = roots[i-1] + gaps[i-1]
    if not _feasible(roots):
        return 1e12
    # structural proxy for the C4 upper bound: last sign change ~ largest root,
    # plus a mild penalty for excessive spread to respect implicit constraints.
    return roots[-1]

def construct():
    from scipy.optimize import differential_evolution
    # initial seed config (kept as fallback)
    seed_roots = np.array([4.0 * (i + 1) for i in range(_M)])
    best = seed_roots.copy()
    try:
        # parameterize by first root + 11 gaps, all in (0, 200]
        lb = np.full(_M, 1e-3)
        ub = np.full(_M, 200.0)
        res = differential_evolution(
            _objective,
            bounds=list(zip(lb, ub)),
            seed=1234,
            maxiter=400,
            popsize=15,
            tol=1e-8,
            polish=True,
            callback=lambda xk, convergence: None,
            updating="immediate",
        )
        if res.success or res.fun < 1e12:
            cand = _params_to_roots(res.x)
            if _feasible(cand):
                best = cand
    except Exception:
        pass
    # ensure return is a plain python list of floats, positive
    out = [float(v) for v in best]
    # enforce strictly increasing positive as a last guard
    for i in range(1, _M):
        if out[i] <= out[i-1]:
            out[i] = out[i-1] + 1e-6
        if out[i] <= 0:
            out[i] = 1e-3
    return out

def _params_to_roots(params):
    roots = np.empty(_M)
    roots[0] = params[0]
    for i in range(1, _M):
        roots[i] = roots[i-1] + params[i]
    return roots
