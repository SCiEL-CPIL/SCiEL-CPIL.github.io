"""Cohn-Goncalves Laguerre double-root construction.

Structural change: constrained nonlinear optimization (SLSQP) of the
refined sign-uncertainty bound over the 12 positive double-root locations,
parameterized as base-root + positive gaps (enforces positivity/ordering).
"""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_genlaguerre

# ---- Precomputation (done once) -------------------------------------------
# Gauss-Laguerre quadrature for the inner product <f,g> = int_0^inf f g e^{-x} dx.
# Integrand P(x)*L_k(x) has degree up to 24+24 = 48 -> 30 nodes exact; use 48.
_NQ = 48
_XQ, _WQ = np.polynomial.laguerre.laggauss(_NQ)
_M = 12                      # m = 12  -> Laguerre combination degrees 0..12
_MAXK = 2 * _M               # P has degree 24
_KFAC = np.array([math.factorial(k) for k in range(_MAXK + 1)], dtype=float)
# Precompute L_k(x_q) for k = 0..24 (shape (25, _NQ))
_LB = np.empty((_MAXK + 1, _NQ))
for _k in range(_MAXK + 1):
    _LB[_k] = eval_genlaguerre(_k, 0.0, _XQ)
# Orthonormalize weights: a_k = <P,L_k>/k!
_WNORM = _WQ / _KFAC[None, :]  # not directly; we divide per k below


def _laguerre_low_coeffs(roots):
    """Return a_0..a_M of P(x)=prod(x-r)^2 in the Laguerre basis L_k (k=0..M)."""
    r = np.asarray(roots, dtype=float)
    # P at quadrature nodes
    val = np.ones_like(_XQ)
    for ri in r:
        d = _XQ - ri
        val = val * d * d
    # a_k = sum_j w_j val_j L_k(x_j) / k!
    out = np.empty(_M + 1)
    for k in range(_M + 1):
        out[k] = np.sum(_WQ * val * _LB[k]) / _KFAC[k]
    return out


def _bound(roots):
    """Smooth surrogate for the Cohn-Goncalves 'last sign change' upper bound C4.

    Combines the coefficient energy (tightens the constant) with a term that
    rewards adjacent same-sign coefficients (fewer / later sign changes).
    """
    a = _laguerre_low_coeffs(roots)
    energy = float(np.sum(a * a))
    # sign-change penalty: -sum(a_k a_{k+1}) is small when adjacent same sign
    sign_term = float(np.sum(a[:-1] * a[1:]))
    return energy - 0.02 * sign_term


# ---- Parameterization: base root + positive gaps ---------------------------
def _roots_from_v(v):
    r1 = v[0]
    gaps = v[1:]
    return r1 + np.concatenate(([0.0], np.cumsum(gaps)))


def _objective_v(v):
    r = _roots_from_v(v)
    return _bound(r)


# Bounds: base root in [0.5, 50]; each gap in [0.1, 12] (keeps roots distinct,
# positive, and the total within a sane range).
_B1 = 0.5
_B2 = 50.0
_G1 = 0.1
_G2 = 12.0
_BOUNDS = [(_B1, _B2)] + [(_G1, _G2)] * (_M - 1)


def _laguerre12_zeros():
    """Zeros of L_12(x): a principled ordered positive starting configuration."""
    try:
        z = np.polynomial.laguerre.Laguerre.basis(_M).roots()
        z = np.sort(np.real(z))
        z = z[z > 0]
        if z.size == _M:
            r1 = float(z[0])
            gaps = np.diff(z)
            v = np.concatenate(([r1], gaps))
            # clip into feasible box
            v[0] = min(max(v[0], _B1), _B2)
            for i in range(1, len(v)):
                v[i] = min(max(v[i], _G1), _G2)
            return v
    except Exception:
        pass
    return None


def _random_v(rng):
    r1 = rng.uniform(_B1, 30.0)
    gaps = rng.uniform(_G1, 8.0, size=_M - 1)
    return np.concatenate(([r1], gaps))


# ---- Main entry point -------------------------------------------------------
def construct():
    t0 = time.time()
    deadline = 630.0 - 45.0          # safety margin before the 630 s kill
    rng = np.random.default_rng(12345)

    # Seed (arithmetic progression) as a guaranteed-valid baseline.
    seed_roots = [4.0 * (i + 1) for i in range(_M)]
    best_roots = list(seed_roots)
    best_val = _bound(np.asarray(seed_roots, dtype=float))

    # Build the list of starting points: Laguerre zeros first, then randoms.
    starts = []
    lz = _laguerre12_zeros()
    if lz is not None:
        starts.append(lz)
    # a few random restarts
    for _ in range(6):
        starts.append(_random_v(rng))

    for v0 in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _objective_v,
                np.asarray(v0, dtype=float),
                method="SLSQP",
                bounds=_BOUNDS,
                options={"maxiter": 250, "ftol": 1e-10},
            )
        except Exception:
            continue
        if res is None or not np.isfinite(res.fun):
            continue
        r = _roots_from_v(res.x)
        # validity: positive, strictly increasing, finite
        if not (np.all(np.isfinite(r)) and np.all(r > 0.0) and np.all(np.diff(r) > 0.0)):
            continue
        val = float(res.fun)
        if val < best_val:
            best_val = val
            best_roots = [float(x) for x in r]

    # Fallback safety: if nothing valid was produced, return the seed.
    if not (len(best_roots) == _M and all(np.isfinite(x) and x > 0 for x in best_roots)):
        best_roots = seed_roots
    return best_roots
# EVOLVE-BLOCK-END
