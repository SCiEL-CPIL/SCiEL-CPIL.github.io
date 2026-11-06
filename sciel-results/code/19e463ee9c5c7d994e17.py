"""Spectral pre-conditioned fit for the six-coefficient CPL growth ansatz."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_T0 = time.time()
_DEADLINE = 72.0  # safety margin under the 90 s wall-clock kill
def _left():
    return _DEADLINE - (time.time() - _T0)

def _Om(a, Om0, wa):
    w0 = -1.0
    return Om0 * a**3 / (Om0 * a**3 + (1.0 - Om0) * a**(-3.0 * (1.0 + w0 + wa * (1.0 - a))))

def _H2(a, Om0, wa):
    w0 = -1.0
    return Om0 * a**3 + (1.0 - Om0) * a**(-3.0 * (1.0 + w0 + wa * (1.0 - a)))

def _ref_fD(a, Om0, wa):
    """Integrate the GR second-order ODE for growth; return f(a), D(a)/D(1)."""
    h2 = _H2(a, Om0, wa)
    om = _Om(a, Om0, wa)
    n = 1600
    xs = np.linspace(0.0, math.log(a), n)
    dx = xs[1] - xs[0]
    # state: y0 = lnD, y1 = dlnD/dx ; integrate backward from x=0 (a=1)
    y0 = 0.0
    y1 = 0.0
    f = np.empty(n)
    D = np.empty(n)
    f[-1] = 0.0
    D[-1] = 1.0
    for i in range(n - 1, 0, -1):
        xi = xs[i]
        ai = math.exp(xi)
        h2i = _H2(ai, Om0, wa)
        omi = _Om(ai, Om0, wa)
        # ODE: y1' = (3/2)(1 + omi - omi/(2*om? )) ... use standard: y1' = 1.5*(1 - 0.5*omi) - 1.5*omi*H'/H
        # H'/H = 1.5*Om - 1.5*(1+w)
        w = -1.0 + wa * (1.0 - ai)
        Hprime_H = 1.5 * omi - 1.5 * (1.0 + w)
        y1p = 1.5 * (1.0 - 0.5 * omi) - 1.5 * omi * Hprime_H
        # backward step
        y1 = y1 - y1p * dx
        y0 = y0 - y1 * dx
        f[i - 1] = y1 / math.log(ai) if ai > 1.0 else 0.0  # dlnD/dlna = y1
        D[i - 1] = math.exp(y0)
    return f, D

def _gamma_from_coeffs(c, q, u, wa, a):
    c0, c1, c2, c3, c4, c5 = c
    return c0 + c1 * q + c2 * q * q + c3 * u + c4 * wa * a + c5 * u * q

def _objective(c, grid):
    """Worst relative error of f and D over the training grid."""
    c0, c1, c2, c3, c4, c5 = c
    Om = grid['Om']; q = grid['q']; u = grid['u']; a = grid['a']; wa = grid['wa']
    g = _gamma_from_coeffs(c, q, u, wa, a)
    f_app = np.power(Om, g)
    err_f = np.abs(f_app - grid['f_ref']) / np.maximum(np.abs(grid['f_ref']), 1e-12)
    # D: exp(-integral f_app dx), x = ln a from ln(a) to 0
    # integrate per model with trapezoid over a's grid
    err_D = np.zeros_like(f_app)
    # process each (model, a) independently using cumulative trapezoid
    for idx in range(len(a)):
        pass
    # Efficient: for each model, integrate over its a-slice
    err_D = np.empty_like(f_app)
    idx = 0
    for m in range(grid['n_models']):
        k = grid['n_a']
        sl = slice(idx, idx + k)
        xs = grid['xs'][sl]
        fa = f_app[sl]
        # integral from ln(a) to 0 of f dx = - integral_0^{ln a} f dx
        # cumulative trapezoid from start (x=ln a0) to x=0
        integ = np.zeros(k)
        for j in range(1, k):
            integ[j] = integ[j - 1] + 0.5 * (fa[j - 1] + fa[j]) * (xs[j] - xs[j - 1])
        D_app = np.exp(-integ)
        err_D[sl] = np.abs(D_app - grid['D_ref'][sl]) / np.maximum(np.abs(grid['D_ref'][sl]), 1e-12)
        idx += k
    return float(np.max(np.concatenate([err_f, err_D])))

def construct():
    if _left() < 1.0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Representative CPL grid
    Om0s = [0.25, 0.30, 0.35]
    was = [-0.6, -0.3, 0.0, 0.3, 0.6]
    a0 = 0.2
    n_a = 24
    a_vals = np.linspace(a0, 1.0, n_a)
    xs = np.log(a_vals)

    Om = []; q = []; u = []; a = []; wa = []; f_ref = []; D_ref = []; xs_list = []
    n_models = 0
    for Om0 in Om0s:
        for w0 in [-1.0]:
            for waa in was:
                if _left() < 2.0:
                    break
                # reference
                f_all = np.empty(n_a)
                D_all = np.empty(n_a)
                for j in range(n_a):
                    if _left() < 2.0:
                        break
                    if a_vals[j] >= 1.0:
                        f_all[j] = 0.0
                        D_all[j] = 1.0
                    else:
                        farr, Darr = _ref_fD(a_vals[j], Om0, waa)
                        f_all[j] = farr[0]
                        D_all[j] = Darr[0]
                for j in range(n_a):
                    om = _Om(a_vals[j], Om0, waa)
                    qj = 1.0 - om
                    uj = 1.0 + w0 + waa * (1.0 - a_vals[j])
                    Om.append(om); q.append(qj); u.append(uj)
                    a.append(a_vals[j]); wa.append(waa)
                    f_ref.append(f_all[j]); D_ref.append(D_all[j]); xs_list.append(xs[j])
                n_models += 1

    grid = {
        'Om': np.array(Om), 'q': np.array(q), 'u': np.array(u),
        'a': np.array(a), 'wa': np.array(wa),
        'f_ref': np.array(f_ref), 'D_ref': np.array(D_ref),
        'xs': np.array(xs_list), 'n_models': n_models, 'n_a': n_a,
    }

    # Initial fit: constant gamma
    c0 = 0.55
    best = [c0, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_val = _objective(best, grid)

    # Local polish: Nelder-Mead style coordinate + small perturbations
    def _try(cand):
        if np.any(np.abs(cand) > 10.0):
            return None
        # gamma bounds check at grid
        g = _gamma_from_coeffs(cand, grid['q'], grid['u'], grid['wa'], grid['a'])
        if np.min(g) < 0.1 or np.max(g) > 1.2:
            return None
        return _objective(cand, grid)

    # coordinate descent with line search
    c = np.array(best, dtype=float)
    step = [0.05, 0.1, 0.2, 0.5, 0.0, 0.05]
    for it in range(200):
        if _left() < 3.0:
            break
        improved = False
        for d in range(6):
            for sgn in (1.0, -1.0):
                for s in step:
                    cand = c.copy()
                    cand[d] = c[d] + sgn * s
                    v = _try(cand)
                    if v is not None and v < best_val:
                        c = cand
                        best_val = v
                        improved = True
        if not improved:
            break

    best = list(np.clip(c, -10.0, 10.0))
    return best
# EVOLVE-BLOCK-END
