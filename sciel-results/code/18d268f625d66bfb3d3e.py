# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import lpmv
from math import lgamma

_K = 10
_NPHI = 130
_NTH = 70
_NPTS = _NTH * _NPHI
_NC = 2 * _K + 1
_DEADLINE_OFFSET = 590.0
_SAFETY = 20.0

def _build_quadrature():
    x, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI
    W = np.outer(w, np.full(_NPHI, dphi))

    Y = np.zeros((_NTH, _NPHI, _NC))
    for idx, m in enumerate(range(-_K, _K + 1)):
        am = abs(m)
        log_ratio = lgamma(_K - am + 1) - lgamma(_K + am + 1)
        fact_ratio = np.exp(log_ratio)
        Nm = np.sqrt((2 * _K + 1) / (4.0 * np.pi) * fact_ratio)
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
    """Minimize R^4 = <f^4> / <f^2>^2. Scale-invariant."""
    f = _Yflat @ c
    f2 = f * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return 1e30
    f4 = f2 * f2
    l4sq = float(np.dot(_Wflat, f4))
    return l4sq / (l2sq * l2sq)

def _gradient(c):
    """Analytic gradient of R^4 = <f^4>/<f^2>^2 w.r.t. c."""
    f = _Yflat @ c
    f2 = f * f
    f3 = f2 * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return np.zeros_like(c)
    l4sq = float(np.dot(_Wflat, f2 * f2))
    wf3 = _Wflat * f3
    wf = _Wflat * f
    grad_f3 = _Yflat.T @ wf3
    grad_f = _Yflat.T @ wf
    B = l2sq
    A = l4sq
    grad = (4.0 / (B * B)) * grad_f3 - (4.0 * A / (B * B * B)) * grad_f
    return grad

def _norm_constraint(c):
    return c @ c - 1.0

def _norm_constraint_jac(c):
    return 2.0 * c

def _optimize_from(c0, deadline):
    c = np.asarray(c0, dtype=np.float64).copy()
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return None
    c = c / nrm

    def fun(x):
        return _objective(x)

    def jac(x):
        return _gradient(x)

    constraints = [{'type': 'eq', 'fun': _norm_constraint, 'jac': _norm_constraint_jac}]

    best_c = c.copy()
    best_val = fun(c)

    # Phase 1: SLSQP with unit-norm constraint
    try:
        res = minimize(fun, c, jac=jac, method="SLSQP", constraints=constraints,
                       options={"maxiter": 300, "ftol": 1e-14, "disp": False})
        if res.success or res.fun < best_val:
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
    except Exception:
        pass

    # Re-normalize after SLSQP
    nrm = np.linalg.norm(best_c)
    if nrm < 1e-14:
        return None
    best_c = best_c / nrm
    best_val = fun(best_c)

    if time.time() > deadline - _SAFETY:
        return best_c, best_val

    # Phase 2: Nelder-Mead polish (no constraint, but start on sphere)
    try:
        res2 = minimize(fun, best_c, method="Nelder-Mead",
                        options={"maxiter": 2000, "xatol": 1e-14,
                                 "fatol": 1e-16, "adaptive": True})
        if res2.fun < best_val:
            best_val = res2.fun
            best_c = res2.x.copy()
    except Exception:
        pass

    nrm = np.linalg.norm(best_c)
    if nrm < 1e-14:
        return None
    best_c = best_c / nrm
    return best_c, float(_objective(best_c))

def _make_seeds(rng, n_seeds):
    seeds = []
    for i in range(n_seeds):
        c = np.zeros(_NC)
        if i == 0:
            c[_K] = 1.0
        elif i == 1:
            c[_K + _K] = 1.0
        elif i == 2:
            c[_K - _K] = 1.0
        elif i == 3:
            for m in range(-_K, _K + 1):
                c[m + _K] = np.cos(0.5 * m)
        elif i == 4:
            for m in range(-_K, _K + 1):
                c[m + _K] = (-1) ** m
        elif i == 5:
            c[:] = rng.standard_normal(_NC)
        elif i == 6:
            for m in range(-_K, _K + 1, 2):
                c[m + _K] = rng.standard_normal()
        elif i == 7:
            for m in range(-_K + 1, _K + 1, 2):
                c[m + _K] = rng.standard_normal()
        elif i == 8:
            c[_K] = 1.0
            c[_K + _K] = rng.standard_normal()
            c[_K - _K] = rng.standard_normal()
        elif i == 9:
            c[_K + 5] = rng.standard_normal()
            c[_K - 5] = rng.standard_normal()
        elif i == 10:
            for m in range(0, _K + 1):
                v = rng.standard_normal()
                c[m + _K] = v
                c[-m + _K] = v
        elif i == 11:
            for m in range(1, _K + 1):
                v = rng.standard_normal()
                c[m + _K] = v
                c[-m + _K] = -v
        elif i == 12:
            for m in range(-3, 4):
                c[m + _K] = rng.standard_normal()
        elif i == 13:
            for m in list(range(-10, -6)) + list(range(7, 11)):
                c[m + _K] = rng.standard_normal()
        elif i == 14:
            c[_K] = 1.0
            c[_K + 5] = 0.5 * rng.standard_normal()
            c[_K - 5] = 0.5 * rng.standard_normal()
            c[_K + 10] = 0.1 * rng.standard_normal()
            c[_K - 10] = 0.1 * rng.standard_normal()
        elif i == 15:
            # Uniform magnitude, random phases
            for m in range(-_K, _K + 1):
                c[m + _K] = rng.standard_normal()
        elif i == 16:
            # Emphasize m=0 and m=±1
            c[_K] = 2.0
            c[_K + 1] = rng.standard_normal()
            c[_K - 1] = rng.standard_normal()
            c[_K + 2] = 0.3 * rng.standard_normal()
            c[_K - 2] = 0.3 * rng.standard_normal()
        elif i == 17:
            # Emphasize m=±10
            c[_K + 10] = 2.0
            c[_K - 10] = 2.0
            c[_K + 9] = 0.5 * rng.standard_normal()
            c[_K - 9] = 0.5 * rng.standard_normal()
            c[_K] = 0.3 * rng.standard_normal()
        elif i == 18:
            # Geometric decay from m=0
            for m in range(-_K, _K + 1):
                c[m + _K] = rng.standard_normal() * (0.7 ** abs(m))
        elif i == 19:
            # Bimodal: strong at m=0 and m=±10
            c[_K] = 1.5
            c[_K + 10] = 1.0 * rng.standard_normal()
            c[_K - 10] = 1.0 * rng.standard_normal()
            for m in range(-4, 5):
                if m != 0:
                    c[m + _K] = 0.2 * rng.standard_normal()
        elif i == 20:
            # Random with low-frequency bias
            for m in range(-_K, _K + 1):
                c[m + _K] = rng.standard_normal() / (1.0 + 0.3 * abs(m))
        elif i == 21:
            # Pure cosines only (c_m = c_-m, c_m real)
            for m in range(1, _K + 1):
                v = rng.standard_normal()
                c[m + _K] = v
                c[-m + _K] = v
            c[_K] = rng.standard_normal()
        elif i == 22:
            # Pure sines only (c_m = -c_-m, c_0=0)
            for m in range(1, _K + 1):
                v = rng.standard_normal()
                c[m + _K] = v
                c[-m + _K] = -v
        else:
            c[:] = rng.standard_normal(_NC)
            c[_K] += 0.5
        nrm = np.linalg.norm(c)
        if nrm > 1e-14:
            c /= nrm
        seeds.append(c)
    return seeds

def rudin_problem10() -> np.ndarray:
    start = time.time()
    deadline = start + _DEADLINE_OFFSET
    rng = np.random.default_rng(42)

    seeds = _make_seeds(rng, 24)
    best_c = None
    best_val = np.inf

    for i, c0 in enumerate(seeds):
        if time.time() > deadline - _SAFETY:
            break
        try:
            out = _optimize_from(c0, deadline)
        except Exception:
            continue
        if out is None:
            continue
        c, val = out
        if val < best_val:
            best_val = val
            best_c = c.copy()

    # Use remaining time for more random restarts
    n_extra = 0
    while time.time() < deadline - 50.0:
        c0 = rng.standard_normal(_NC)
        nrm = np.linalg.norm(c0)
        if nrm < 1e-14:
            continue
        c0 /= nrm
        try:
            out = _optimize_from(c0, deadline)
        except Exception:
            continue
        if out is None:
            continue
        c, val = out
        if val < best_val:
            best_val = val
            best_c = c.copy()
        n_extra += 1

    # Final high-precision polish if time permits
    if best_c is not None and time.time() < deadline - 15.0:
        constraints = [{'type': 'eq', 'fun': _norm_constraint, 'jac': _norm_constraint_jac}]
        try:
            res = minimize(_objective, best_c, jac=_gradient, method="SLSQP",
                           constraints=constraints,
                           options={"maxiter": 500, "ftol": 1e-16, "disp": False})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

        # One more Nelder-Mead polish
        if time.time() < deadline - 5.0:
            try:
                res2 = minimize(_objective, best_c, method="Nelder-Mead",
                                options={"maxiter": 3000, "xatol": 1e-15,
                                         "fatol": 1e-17, "adaptive": True})
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_c = res2.x.copy()
            except Exception:
                pass

    if best_c is None:
        best_c = np.ones(_NC, dtype=np.float64)
        best_c /= np.linalg.norm(best_c)

    best_c = np.asarray(best_c, dtype=np.float64)
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(_NC, dtype=np.float64)
        best_c /= np.linalg.norm(best_c)
    mx = float(np.max(np.abs(best_c)))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)
    return best_c
# EVOLVE-BLOCK-END
