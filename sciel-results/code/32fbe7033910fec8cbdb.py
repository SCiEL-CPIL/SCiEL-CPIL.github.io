"""Self-contained optimizer for the six-coefficient CPL growth-index ansatz."""
import time
import math
import numpy as np

_START = time.time()
_DEADLINE = _START + 82.0

def _budget_ok():
    return time.time() < _DEADLINE

# ---- Cosmology helpers -----------------------------------------------------
def _Om(a, Om, w0, wa):
    a = np.asarray(a, dtype=float)
    return Om * a**(-3) / (Om * a**(-3) + (1.0 - Om) * np.exp(-3.0 * (1.0 + w0 + wa) * (1.0 - a) - 1.5 * wa * (1.0 - a**2)))

def _f_ref(x, Om, w0, wa):
    a = np.exp(x)
    O = _Om(a, Om, w0, wa)
    H2 = O + (1.0 - O) * np.exp(-3.0 * (1.0 + w0 + wa) * (1.0 - a) - 1.5 * wa * (1.0 - a**2))
    q = 0.5 * O - 0.5 * H2
    return 0.5 * O - 1.5 * q

def _D_ref(x, Om, w0, wa):
    # D(a)/D(1), x = ln a
    f = _f_ref(x, Om, w0, wa)
    n = len(x)
    if n < 2:
        return np.ones_like(x)
    cum = np.empty(n)
    cum[0] = 0.0
    for i in range(1, n):
        h = x[i] - x[i-1]
        cum[i] = cum[i-1] + 0.5 * h * (f[i] + f[i-1])
    return np.exp(-cum)

# ---- Approximation ---------------------------------------------------------
def _gamma(c, a, Om, w0, wa):
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c[0] + c[1]*q + c[2]*q*q + c[3]*u + c[4]*wa*a + c[5]*u*q

def _f_approx(c, a, Om, w0, wa):
    g = _gamma(c, a, Om, w0, wa)
    return Om**g

def _D_approx(c, x, Om, w0, wa):
    a = np.exp(x)
    f = _f_approx(c, a, Om, w0, wa)
    n = len(x)
    if n < 2:
        return np.ones_like(x)
    cum = np.empty(n)
    cum[0] = 0.0
    for i in range(1, n):
        h = x[i] - x[i-1]
        cum[i] = cum[i-1] + 0.5 * h * (f[i] + f[i-1])
    return np.exp(-cum)

# ---- Objective (smooth max relative error, with gamma soft-constraint) -----
def _objective(c, models, xgrid, beta):
    errs = []
    for (Om, w0, wa) in models:
        fref = _f_ref(xgrid, Om, w0, wa)
        Dref = _D_ref(xgrid, Om, w0, wa)
        fa = _f_approx(c, np.exp(xgrid), Om, w0, wa)
        Da = _D_approx(c, xgrid, Om, w0, wa)
        errs.append(np.max(np.abs(fa - fref) / fref))
        errs.append(np.max(np.abs(Da - Dref) / Dref))
    e = np.max(errs)
    # soft gamma penalty
    gmin = 1.0
    gmax = -1.0
    for (Om, w0, wa) in models:
        g = _gamma(c, np.exp(xgrid), Om, w0, wa)
        gmin = min(gmin, float(np.min(g)))
        gmax = max(gmax, float(np.max(g)))
    pen = 0.0
    if gmin < 0.10:
        pen += (0.10 - gmin) ** 2
    if gmax > 1.20:
        pen += (gmax - 1.20) ** 2
    if beta > 0 and pen > 0:
        e = e + math.log(1.0 + beta * pen) / beta
    return e

def _grad(c, models, xgrid, beta, h=1e-7):
    g = np.zeros(6)
    base = _objective(c, models, xgrid, beta)
    for i in range(6):
        cp = c.copy(); cp[i] += h
        cm = c.copy(); cm[i] -= h
        g[i] = (_objective(cp, models, xgrid, beta) - _objective(cm, models, xgrid, beta)) / (2.0 * h)
    return base, g

def _nelder_mead(c0, models, xgrid, beta, tol=1e-9, maxit=400):
    c = np.array(c0, dtype=float)
    # simplex
    S = [c.copy()]
    for i in range(6):
        d = c.copy()
        d[i] += 0.05
        S.append(d)
    S = [np.array(s, dtype=float) for s in S]
    fvals = [_objective(s, models, xgrid, beta) for s in S]
    for it in range(maxit):
        if not _budget_ok():
            break
        order = np.argsort(fvals)
        S = [S[i] for i in order]
        fvals = [fvals[i] for i in order]
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        centroid = np.mean(S[:-1], axis=0)
        xr = centroid + 1.0 * (centroid - S[-1])
        fr = _objective(xr, models, xgrid, beta)
        if fr < fvals[0]:
            xe = centroid + 2.0 * (centroid - S[-1])
            fe = _objective(xe, models, xgrid, beta)
            if fe < fr:
                S[-1] = xe; fvals[-1] = fe
            else:
                S[-1] = xr; fvals[-1] = fr
        elif fr < fvals[-2]:
            S[-1] = xr; fvals[-1] = fr
        else:
            xc = centroid + 0.5 * (S[-1] - centroid)
            fc = _objective(xc, models, xgrid, beta)
            if fc < fvals[-1]:
                S[-1] = xc; fvals[-1] = fc
            else:
                for j in range(1, len(S)):
                    S[j] = 0.5 * (S[j] + S[0])
                    fvals[j] = _objective(S[j], models, xgrid, beta)
    best = int(np.argmin(fvals))
    return S[best], fvals[best]

def _grad_descent(c0, models, xgrid, beta, maxit=200):
    c = np.array(c0, dtype=float)
    f, g = _grad(c, models, xgrid, beta)
    lr = 0.02
    for it in range(maxit):
        if not _budget_ok():
            break
        ng = np.linalg.norm(g)
        if ng < 1e-10:
            break
        c = c - lr * g
        f2, g = _grad(c, models, xgrid, beta)
        if f2 < f:
            f = f2
        else:
            lr *= 0.5
    return c, f

def construct():
    # Training models (smooth CPL) and integration grid.
    models = [
        (0.30, -1.0, 0.0),
        (0.27, -1.0, 0.0),
        (0.30, -0.5, 0.5),
        (0.30, -1.0, 0.5),
        (0.25, -1.2, 0.3),
        (0.35, -0.9, 0.2),
    ]
    xgrid = np.linspace(-6.0, 0.0, 80)  # a from ~0.0025 to 1

    beta = 100.0
    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_f = _objective(best_c, models, xgrid, beta)

    # Multi-start Nelder-Mead
    rng = np.random.default_rng(12345)
    starts = [best_c.copy()]
    for _ in range(6):
        s = best_c.copy() + rng.normal(0.0, 0.15, size=6)
        starts.append(s)

    for s in starts:
        if not _budget_ok():
            break
        c, f = _nelder_mead(s, models, xgrid, beta, tol=1e-9, maxit=300)
        if f < best_f:
            best_f = f
            best_c = c

    # Gradient polish
    if _budget_ok():
        c, f = _grad_descent(best_c, models, xgrid, beta, maxit=150)
        if f < best_f:
            best_f = f
            best_c = c

    # Final tight Nelder-Mead polish (small steps)
    if _budget_ok():
        c, f = _nelder_mead(best_c, models, xgrid, beta, tol=1e-12, maxit=300)
        if f < best_f:
            best_f = f
            best_c = c

    # Clamp to [-10,10]
    best_c = np.clip(best_c, -10.0, 10.0)
    return [float(x) for x in best_c]
