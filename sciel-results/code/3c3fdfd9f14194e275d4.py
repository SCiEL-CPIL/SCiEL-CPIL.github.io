import time
import math
import numpy as np
from scipy.optimize import least_squares

_DEADLINE = time.time() + 85.0  # safety margin vs 90s wall-clock

def _om(a, om, w0, wa):
    e2 = om * a ** (-3) + (1 - om) * a ** (-3 * (1 + w0 + wa * (1 - a)))
    return e2 / e2  # placeholder
def _om(a, om, w0, wa):
    e2 = om * a ** (-3) + (1 - om) * a ** (-3 * (1 + w0 + wa * (1 - a)))
    return e2 / e2  # (redefined below for clarity)

def _gamma(a, om, w0, wa, c):
    c0, c1, c2, c3, c4, c5 = c
    q = 1.0 - om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c0 + c1 * q + c2 * q * q + c3 * u + c4 * wa * a + c5 * u * q

def _f(a, om, w0, wa, c):
    g = _gamma(a, om, w0, wa, c)
    return om ** g

def _Dnorm(a, om, w0, wa, c):
    # normalized growth factor via log-integral
    xs = np.linspace(math.log(a), 0.0, 200)
    aa = np.exp(xs)
    ee = om * aa ** (-3) + (1 - om) * aa ** (-3 * (1 + w0 + wa * (1 - aa)))
    oo = ee / ee
    oo = (om * aa ** (-3)) / ee
    ff = oo ** _gamma(aa, om, w0, wa, c)
    return math.exp(-np.trapz(ff, xs))

def _ref_f(a, om, w0, wa):
    ee = om * a ** (-3) + (1 - om) * a ** (-3 * (1 + w0 + wa * (1 - a)))
    return 0.5 * om + (5.0 / 7.0) * om - (3.0 / 7.0) * om * (1 - om) / ee

def _ref_D(a, om, w0, wa):
    xs = np.linspace(math.log(a), 0.0, 400)
    aa = np.exp(xs)
    ff = _ref_f(aa, om, w0, wa)
    return math.exp(-np.trapz(ff, xs))

# validation grid
_A = np.linspace(0.05, 1.0, 60)
_MODELS = [
    (0.30, -1.0, 0.0),
    (0.27, -1.0, 0.0),
    (0.30, -1.0, 0.3),
    (0.30, -0.5, 0.5),
    (0.30, -1.3, 0.5),
    (0.25, -1.0, 0.2),
    (0.35, -1.0, 0.0),
    (0.30, -0.9, 0.1),
    (0.30, -1.2, 0.4),
    (0.28, -1.0, 0.0),
]

def construct():
    t0 = time.time()
    best = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_err = np.inf

    # precompute reference data
    ref = []
    for om, w0, wa in _MODELS:
        for a in _A:
            ref.append((a, om, w0, wa, _ref_f(a, om, w0, wa), _ref_D(a, om, w0, wa)))

    def err(c):
        # relative error on f and D, weighted; also gamma range penalty
        s = 0.0
        for a, om, w0, wa, rf, rd in ref:
            g = _gamma(a, om, w0, wa, c)
            if g < 0.1 or g > 1.2:
                s += (min(abs(g - 0.1), abs(g - 1.2))) * 10.0
            f = om ** g
            s += abs(f - rf) / rf
            d = _Dnorm(a, om, w0, wa, c)
            s += abs(d - rd) / rd
        return s

    def res(c):
        out = []
        for a, om, w0, wa, rf, rd in ref:
            g = _gamma(a, om, w0, wa, c)
            if g < 0.1 or g > 1.2:
                out.append(np.sign(g - 0.5) * 5.0)
            f = om ** g
            out.append((f - rf) / rf)
            d = _Dnorm(a, om, w0, wa, c)
            out.append((d - rd) / rd)
        return out

    # local refinement from the seed
    try:
        sol = least_squares(res, best, method='trf', bounds=([-10]*6, [10]*6),
                            xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=400)
        if sol.x is not None and np.all(np.isfinite(sol.x)):
            e = err(sol.x)
            if e < best_err:
                best, best_err = sol.x, e
    except Exception:
        pass

    if time.time() > _DEADLINE:
        return [float(x) for x in best]

    # multi-start to escape local minima
    rng = np.random.default_rng(0)
    starts = [best.copy()]
    for _ in range(6):
        s = np.array([rng.uniform(0.3, 0.8), rng.uniform(-0.5, 0.5),
                      rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5),
                      rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)])
        starts.append(s)
    for s in starts:
        if time.time() > _DEADLINE - 5:
            break
        try:
            sol = least_squares(res, s, method='trf', bounds=([-10]*6, [10]*6),
                                xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=300)
            if sol.x is not None and np.all(np.isfinite(sol.x)):
                e = err(sol.x)
                if e < best_err:
                    best, best_err = sol.x, e
        except Exception:
            continue

    # clamp to [-10,10]
    best = np.clip(best, -10.0, 10.0)
    return [float(x) for x in best]
