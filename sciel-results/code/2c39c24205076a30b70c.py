"""Compact dark-energy growth-index fit; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

_DEADLINE = 82.0


def _now():
    return time.time()


def _Om(a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    wa_a = 1.0 - a
    w = w0 + wa * wa_a
    x = np.maximum(a, 1e-300)
    num = 1.0 + w
    den = 1.0 + 3.0 * w
    E2 = Om0 * np.power(x, 3.0 * den) + (1.0 - Om0) * np.power(x, -3.0 * num)
    return Om0 * np.power(x, 3.0) / E2


def _f_gr(a, Om0, w0, wa):
    Om = _Om(a, Om0, w0, wa)
    a = np.maximum(np.asarray(a, dtype=float), 1e-300)
    dOm = (3.0 * Om - 1.5 * Om * (w0 + wa * (1.0 - a))) / a
    f = 0.5 * Om + 0.6 * np.sqrt(np.maximum(Om * (1.0 - Om), 0.0)) + 0.2 * dOm / (3.0 * a)
    return f


def _D_gr(a, Om0, w0, wa, n=2000):
    a = np.asarray(a, dtype=float)
    xs = np.linspace(np.log(1e-3), 0.0, n)
    f = _f_gr(np.exp(xs), Om0, w0, wa)
    I = np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(xs))))
    out = np.zeros_like(a)
    for i, av in enumerate(a):
        if av <= 1e-3:
            out[i] = 1.0
            continue
        xv = min(max(math.log(av), xs[0]), 0.0)
        out[i] = math.exp(-np.interp(xv, xs, I))
    return out


def _gamma(c, a, Om0, w0, wa):
    q = 1.0 - _Om(a, Om0, w0, wa)
    u = 1.0 + w0 + wa * (1.0 - a)
    return c[0] + c[1] * q + c[2] * q * q + c[3] * u + c[4] * wa * a + c[5] * u * q


def _f_app(c, a, Om0, w0, wa):
    Om = _Om(a, Om0, w0, wa)
    g = _gamma(c, a, Om0, w0, wa)
    g = np.clip(g, 0.1, 1.2)
    return np.power(np.maximum(Om, 1e-300), g)


def _D_app(c, a, Om0, w0, wa, n=2000):
    a = np.asarray(a, dtype=float)
    xs = np.linspace(np.log(1e-3), 0.0, n)
    fa = _f_app(c, np.exp(xs), Om0, w0, wa)
    I = np.concatenate(([0.0], np.cumsum(0.5 * (fa[1:] + fa[:-1]) * np.diff(xs))))
    out = np.zeros_like(a)
    for i, av in enumerate(a):
        if av <= 1e-3:
            out[i] = 1.0
            continue
        xv = min(max(math.log(av), xs[0]), 0.0)
        out[i] = math.exp(-np.interp(xv, xs, I))
    return out


def _build_reference(models, a_eval, a_fit):
    ref_f = []
    ref_D = []
    for (Om0, w0, wa) in models:
        ref_f.append(_f_gr(a_eval, Om0, w0, wa))
        ref_D.append(_D_gr(a_eval, Om0, w0, wa))
    return ref_f, ref_D


def _eval_fast(c, ref_f, ref_D, models, a_eval, a_fit):
    if not np.all(np.isfinite(c)):
        return 1e9
    if np.any(np.abs(c) > 10.0):
        return 1e9

    err_f = 0.0
    err_D = 0.0
    for idx, (Om0, w0, wa) in enumerate(models):
        fa = _f_app(c, a_eval, Om0, w0, wa)
        fg = ref_f[idx]
        rel_f = np.abs(fa - fg) / np.maximum(np.abs(fg), 1e-15)
        err_f = max(err_f, float(np.max(rel_f)))

        Da = _D_app(c, a_eval, Om0, w0, wa)
        Dg = ref_D[idx]
        rel_D = np.abs(Da - Dg) / np.maximum(np.abs(Dg), 1e-15)
        err_D = max(err_D, float(np.max(rel_D)))

    # Check gamma constraint on fit grid
    for (Om0, w0, wa) in models:
        g = _gamma(c, a_fit, Om0, w0, wa)
        if np.any(g < 0.1 - 1e-6) or np.any(g > 1.2 + 1e-6):
            return 1e9

    return 100.0 * max(err_f, err_D)


def _eval_smooth(c, ref_f, ref_D, models, a_eval, a_fit, beta=50.0):
    if not np.all(np.isfinite(c)):
        return 1e9
    if np.any(np.abs(c) > 10.0):
        return 1e9

    all_errors = []
    for idx, (Om0, w0, wa) in enumerate(models):
        fa = _f_app(c, a_eval, Om0, w0, wa)
        fg = ref_f[idx]
        rel_f = np.abs(fa - fg) / np.maximum(np.abs(fg), 1e-15)
        all_errors.extend(rel_f.tolist())

        Da = _D_app(c, a_eval, Om0, w0, wa)
        Dg = ref_D[idx]
        rel_D = np.abs(Da - Dg) / np.maximum(np.abs(Dg), 1e-15)
        all_errors.extend(rel_D.tolist())

    all_errors = np.array(all_errors)
    m = np.max(all_errors)
    if m > 1e9:
        return 1e9
    lse = m + np.log(np.sum(np.exp(all_errors - m))) / beta
    return 100.0 * lse


def construct():
    t0 = _now()

    models = [
        (0.30, -1.0, 0.0),
        (0.30, -0.5, 1.0),
        (0.30, -1.0, 0.5),
        (0.27, -1.0, 0.3),
        (0.30, -0.9, 0.0),
    ]
    a_eval = np.logspace(-2.0, 0.0, 80)
    a_fit = np.logspace(-2.0, 0.0, 50)

    ref_f, ref_D = _build_reference(models, a_eval, a_fit)

    seed_c = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_c = np.array(seed_c)
    best_v = _eval_fast(best_c, ref_f, ref_D, models, a_eval, a_fit)

    def _try(c):
        nonlocal best_c, best_v
        c = np.clip(np.asarray(c, dtype=float), -10.0, 10.0)
        if not np.all(np.isfinite(c)):
            return
        v = _eval_fast(c, ref_f, ref_D, models, a_eval, a_fit)
        if v < best_v:
            best_v = v
            best_c = c.copy()

    # Phase 1: Differential evolution for global search
    elapsed = _now() - t0
    if elapsed < _DEADLINE - 35.0:
        try:
            bounds = [(-10.0, 10.0)] * 6
            de_res = differential_evolution(
                lambda x: _eval_smooth(np.clip(x, -10.0, 10.0), ref_f, ref_D, models, a_eval, a_fit, beta=40.0),
                bounds=bounds,
                seed=42,
                maxiter=40,
                popsize=12,
                tol=1e-4,
                mutation=(0.4, 1.0),
                recombination=0.8,
                polish=False,
            )
            _try(de_res.x)
        except Exception:
            pass

    # Phase 2: Multi-start Nelder-Mead polish
    rng = np.random.default_rng(12345)
    n_starts = 12
    for i in range(n_starts):
        if _now() - t0 > _DEADLINE - 8.0:
            break

        if i == 0:
            start = best_c.copy()
        elif i == 1:
            start = np.array(seed_c)
        else:
            start = best_c + rng.normal(0.0, 0.12, 6)
            start = np.clip(start, -10.0, 10.0)

        try:
            res = minimize(
                lambda x: _eval_smooth(np.clip(x, -10.0, 10.0), ref_f, ref_D, models, a_eval, a_fit, beta=80.0),
                np.clip(start, -10.0, 10.0),
                method="Nelder-Mead",
                options={"maxiter": 400, "maxfev": 600,
                         "xatol": 1e-7, "fatol": 1e-8},
            )
            _try(res.x)
        except Exception:
            pass

    # Phase 3: Final tight polish
    if _now() - t0 < _DEADLINE - 4.0:
        try:
            res = minimize(
                lambda x: _eval_fast(np.clip(x, -10.0, 10.0), ref_f, ref_D, models, a_eval, a_fit),
                best_c.copy(),
                method="Nelder-Mead",
                options={"maxiter": 200, "maxfev": 400,
                         "xatol": 1e-8, "fatol": 1e-9},
            )
            _try(res.x)
        except Exception:
            pass

    # Final validation
    best_c = np.clip(best_c, -10.0, 10.0)
    valid = True
    for (Om0, w0, wa) in models:
        g = _gamma(best_c, a_fit, Om0, w0, wa)
        if g.min() < 0.1 or g.max() > 1.2:
            valid = False
            break
    if not valid:
        best_c = np.array(seed_c)

    return [float(x) for x in best_c]
# EVOLVE-BLOCK-END
