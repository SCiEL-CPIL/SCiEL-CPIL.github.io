"""Suite task: Laguerre double roots (m=12, n=1). Pick z_i to minimize B."""
import time
import math
import itertools
import random
import sympy
from sympy import Rational, Poly, laguerre, diff, real_roots, Symbol, nsimplify, sqrt

# ---- deadline guard ----
_START = time.time()
_DEADLINE = 630.0
_MARGIN = 15.0

def _remaining():
    return _DEADLINE - _MARGIN - (time.time() - _START)

def _timeup():
    return (time.time() - _START) > (_DEADLINE - _MARGIN)

def _to_rational(x):
    return Rational(nsimplify(float(x), rational=True))

# ---- exact score evaluation ----
def _score(zs):
    x = Symbol('x')
    al = Rational(-1, 2)
    zs_r = sorted([_to_rational(z) for z in zs])
    # build 26x26 exact linear system for c_0..c_25
    M = []
    b = []
    for k in range(26):
        L = laguerre(2 * k, x, al)
        M.append([L.coeff(x, j) for j in range(27)])
        b.append(Rational(0))
    # g(0) = 0
    M[0] = [M[0][j] for j in range(26)]
    b[0] = Rational(0)
    # g'(0) = 1
    M[1] = [Rational(0) if j == 0 else M[1][j] * j for j in range(26)]
    b[1] = Rational(1)
    for i, zi in enumerate(zs_r):
        Lf = [laguerre(2 * k, x, al).subs(x, zi) for k in range(26)]
        M.append(Lf)
        b.append(Rational(0))
        Ld = [laguerre(2 * k, x, al).diff(x).subs(x, zi) for k in range(26)]
        M.append(Ld)
        b.append(Rational(0))
    # solve exactly
    A = sympy.Matrix(M)
    bv = sympy.Matrix(b)
    try:
        if A.det() == 0:
            return None
        c = A.LUsolve(bv)
    except Exception:
        return None
    # build g(x)
    g = sum(c[k] * laguerre(2 * k, x, al) for k in range(26))
    # denominator D = x * prod (x - zi)^2
    D = Poly(x, x)
    for zi in zs_r:
        D = D * Poly((x - zi) ** 2, x)
    gP = Poly(g, x)
    try:
        q, rem = gP.div(D)
    except Exception:
        return None
    if q.is_zero:
        return 0.0
    try:
        roots = real_roots(q)
    except Exception:
        return None
    B = 0.0
    for r in roots:
        rv = float(r)
        if rv <= 0:
            continue
        try:
            lo = float(q.evalf(rv - 1e-198))
            hi = float(q.evalf(rv + 1e-198))
        except Exception:
            continue
        if lo == 0.0 or hi == 0.0:
            continue
        if (lo < 0 and hi > 0) or (lo > 0 and hi < 0):
            if rv > B:
                B = rv
    return B / (2.0 * math.pi)

# ---- candidate pool ----
def _build_pool():
    lo = 0.5
    hi = 20.0
    n = 120
    step = (hi - lo) / (n - 1)
    pool = []
    for i in range(n):
        v = lo + i * step
        pool.append(round(v, 6))
    # add some sparse large values
    for v in [25.0, 30.0, 40.0, 50.0, 60.0]:
        pool.append(v)
    return pool

_POOL = _build_pool()
_POOL_SET = set(_POOL)

def _valid(zs):
    if len(zs) != 12:
        return False
    seen = set()
    for z in zs:
        if not math.isfinite(z) or z <= 0:
            return False
        r = _to_rational(z)
        if r in seen:
            return False
        seen.add(r)
    return True

def _local_search(zs):
    # coordinate descent over pool values
    zs = list(zs)
    try:
        cur = _score(zs)
    except Exception:
        return None, None
    if cur is None:
        return None, None
    best_zs = list(zs)
    best = cur
    it = 0
    while _remaining() > 60.0:
        it += 1
        improved = False
        for i in range(12):
            old = zs[i]
            for cand in _POOL:
                if cand == old:
                    continue
                trial = list(zs)
                trial[i] = cand
                if not _valid(trial):
                    continue
                try:
                    s = _score(trial)
                except Exception:
                    continue
                if s is not None and s < best - 1e-12:
                    zs = trial
                    best = s
                    best_zs = list(zs)
                    improved = True
                    break
            if _timeup():
                break
        if not improved:
            break
    return best_zs, best

def _initial_candidates():
    cands = []
    # evenly spaced
    cands.append([0.5 + 1.5 * i for i in range(12)])
    cands.append([1.0 + 1.0 * i for i in range(12)])
    cands.append([0.5 + 0.7 * i for i in range(12)])
    # geometric
    cands.append([0.5 * (1.2 ** i) for i in range(12)])
    cands.append([0.3 * (1.3 ** i) for i in range(12)])
    # small cluster + tail
    cands.append([0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.5, 3.0, 4.0, 5.5, 7.5, 10.0])
    return cands

def _orthogonal_nodes():
    """Compute the 12 positive roots of L_12^(-1/2)(x) using numpy."""
    import numpy as np
    # Use numpy's roots for the Laguerre polynomial
    # L_n^(alpha)(x) can be constructed via numpy.polynomial.laguerre
    from numpy.polynomial.laguerre import laggauss
    # laggauss(n) returns nodes and weights for n-point Gaussian quadrature
    # which are the roots of L_n(x) (alpha=0). We need alpha=-1/2.
    # Instead, construct the polynomial coefficients directly.
    # L_n^(alpha)(x) = sum_{k=0}^{n} (-1)^k * binom(n+alpha, n-k) * x^k / k!
    n = 12
    alpha = -0.5
    # Build coefficients in descending power order for numpy.roots
    coeffs = []
    for k in range(n, -1, -1):
        # coefficient of x^k in L_n^(alpha)(x)
        # = (-1)^k * binom(n+alpha, n-k) / (n-k)!
        # Using scipy.special for exact computation
        from scipy.special import factorial, binom as sp_binom
        coeff = (-1)**k * sp_binom(n + alpha, n - k) / factorial(n - k, exact=True)
        coeffs.append(float(coeff))
    nodes = np.roots(coeffs)
    # Filter to real, positive nodes
    real_nodes = sorted([float(x) for x in nodes if abs(x.imag) < 1e-10 and x.real > 0])
    if len(real_nodes) >= 12:
        return real_nodes[:12]
    return None

def _continuous_optimize(zs, max_iter=50):
    """
    Continuous coordinate descent with adaptive step sizes.
    For each coordinate, try several step sizes in both directions.
    """
    zs = list(zs)
    try:
        cur = _score(zs)
    except Exception:
        return None, None
    if cur is None:
        return None, None
    best_zs = list(zs)
    best = cur
    step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    
    for iteration in range(max_iter):
        if _timeup() or _remaining() < 60.0:
            break
        improved = False
        for i in range(12):
            old = zs[i]
            # Try moving coordinate i in both directions with various step sizes
            for step in step_sizes:
                for direction in [1, -1]:
                    trial = list(zs)
                    trial[i] = old + direction * step
                    if not _valid(trial):
                        continue
                    try:
                        s = _score(trial)
                    except Exception:
                        continue
                    if s is not None and s < best - 1e-12:
                        zs = trial
                        best = s
                        best_zs = list(zs)
                        improved = True
                        # Reset step size to larger one for next iteration
                        step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
                        break
                if improved:
                    break
            if _timeup() or _remaining() < 60.0:
                break
        if not improved:
            # Reduce step sizes if no improvement
            step_sizes = [s * 0.5 for s in step_sizes]
            if step_sizes[0] < 0.001:
                break
    return best_zs, best

def construct():
    best_zs = None
    best = None
    
    # Try orthogonal polynomial nodes first (measured improvement)
    ortho = _orthogonal_nodes()
    if ortho is not None and _valid(ortho):
        try:
            s = _score(ortho)
        except Exception:
            s = None
        if s is not None and (best is None or s < best):
            best = s
            best_zs = list(ortho)
    
    # initial candidates
    for cand in _initial_candidates():
        if _timeup():
            break
        if not _valid(cand):
            continue
        try:
            s = _score(cand)
        except Exception:
            continue
        if s is not None and (best is None or s < best):
            best = s
            best_zs = list(cand)
    
    # Continuous optimization from best
    if best_zs is not None and _remaining() > 90.0:
        cz, cs = _continuous_optimize(best_zs)
        if cz is not None and (best is None or cs < best):
            best = cs
            best_zs = cz
    
    # Also try continuous optimization from orthogonal nodes if we have them
    if ortho is not None and _valid(ortho) and _remaining() > 90.0:
        cz, cs = _continuous_optimize(ortho)
        if cz is not None and (best is None or cs < best):
            best = cs
            best_zs = cz
    
    # Random restarts while time remains
    rng = random.Random(12345)
    while _remaining() > 120.0:
        # Mix of pool-based and continuous random starts
        if rng.random() < 0.5:
            zs = [rng.choice(_POOL) for _ in range(12)]
        else:
            # Continuous random in [0.3, 15]
            zs = [rng.uniform(0.3, 15.0) for _ in range(12)]
        if not _valid(zs):
            continue
        try:
            s = _score(zs)
        except Exception:
            continue
        if s is not None and (best is None or s < best):
            best = s
            best_zs = list(zs)
            if _remaining() > 150.0:
                cz, cs = _continuous_optimize(best_zs)
                if cz is not None and (best is None or cs < best):
                    best = cs
                    best_zs = cz
    
    # Final continuous polish
    if best_zs is not None and _remaining() > 60.0:
        cz, cs = _continuous_optimize(best_zs)
        if cz is not None and (best is None or cs < best):
            best = cs
            best_zs = cz
    
    # fallback
    if best_zs is None:
        best_zs = [4.0 * (i + 1) for i in range(12)]
    # final validation
    if not _valid(best_zs):
        best_zs = [4.0 * (i + 1) for i in range(12)]
    return [float(z) for z in best_zs]
