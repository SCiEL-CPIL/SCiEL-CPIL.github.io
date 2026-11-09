"""Compact dark-energy growth approximation: optimize 6 coefficients vs GR ODE."""
import time
import math
import numpy as np

DEADLINE = 82.0  # safety margin under 90s wall-clock

def _Om(a, Om0, w0, wa):
    return Om0 * a**(-3.0) / (Om0 * a**(-3.0) + (1.0 - Om0) * (a / 1.0)**(-3.0 * (1.0 + w0 + wa)))

def _gamma(c, a, Om0, w0, wa):
    Om = _Om(a, Om0, w0, wa)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c[0] + c[1]*q + c[2]*q*q + c[3]*u + c[4]*wa*a + c[5]*u*q

def _f_approx(c, a, Om0, w0, wa):
    Om = _Om(a, Om0, w0, wa)
    return Om ** _gamma(c, a, Om0, w0, wa)

def _D_norm_approx(c, a, Om0, w0, wa, N=256):
    # D(a)/D(1) = exp(- integral_{ln a}^{0} f(exp x) dx )
    if a >= 1.0:
        return 1.0
    x0 = math.log(a)
    xs = np.linspace(x0, 0.0, N)
    fs = np.array([_f_approx(c, math.exp(x), Om0, w0, wa) for x in xs])
    val = -np.trapz(fs, xs)
    return math.exp(val)

def _solve_ode(a, Om0, w0, wa, N=2000):
    # y'' + (3/H + H'/H) y' - (3/2) Om(a) y = 0, y(1)=1, y'(1)=f(1)
    def Hinv(a_):
        return math.sqrt(1.0 / (_Om(a_, Om0, w0, wa) + (1.0 - Om0) * (a_ / 1.0)**(-3.0 * (1.0 + w0 + wa))))
    def f_exact(a_):
        Om = _Om(a_, Om0, w0, wa)
        H = 1.0 / Hinv(a_)
        H2 = H * H
        H2p = 1.5 * H * (Om - (1.0 + w0 + wa * (1.0 - a_)) * (1.0 - Om))
        return 0.5 * Om - 1.5 * H2p / (2.0 * H2)
    # integrate in ln a from 0 to ln a using RK4
    if a >= 1.0:
        return 1.0
    ln_a = math.log(a)
    dx = ln_a / N  # negative step
    x = 0.0
    y = 1.0
    yp = f_exact(1.0)
    for i in range(N):
        a_ = math.exp(x)
        Hinv_ = Hinv(a_)
        H = 1.0 / Hinv_
        H2 = H * H
        H2p = 1.5 * H * (_Om(a_, Om0, w0, wa) - (1.0 + w0 + wa * (1.0 - a_)) * (1.0 - _Om(a_, Om0, w0, wa)))
        p = 3.0 * H + H2p / H2
        q2 = 1.5 * _Om(a_, Om0, w0, wa)
        def deriv(y_, yp_):
            return (yp_, -p * yp_ - q2 * y_)
        k1y, k1yp = deriv(y, yp)
        k2y, k2yp = deriv(y + 0.5*dx*k1y, yp + 0.5*dx*k1yp)
        k3y, k3yp = deriv(y + 0.5*dx*k2y, yp + 0.5*dx*k2yp)
        k4y, k4yp = deriv(y + dx*k3y, yp + dx*k3yp)
        y = y + (dx/6.0)*(k1y + 2*k2y + 2*k3y + k4y)
        yp = yp + (dx/6.0)*(k1yp + 2*k2yp + 2*k3yp + k4yp)
        x += dx
    return y

def _rel_err(x, ref):
    if ref == 0.0:
        return abs(x)
    return abs(x - ref) / abs(ref)

def _score(c, models, avals, f_cache, D_cache):
    worst = 0.0
    for m in range(len(models)):
        Om0, w0, wa = models[m]
        for a in avals:
            f_a = f_cache[m][a]
            D_a = D_cache[m][a]
            if f_a is not None:
                err = _rel_err(_f_approx(c, a, Om0, w0, wa), f_a)
                if err > worst:
                    worst = err
            if D_a is not None:
                err = _rel_err(_D_norm_approx(c, a, Om0, w0, wa), D_a)
                if err > worst:
                    worst = err
    return worst

def _gamma_range_ok(c, models, avals):
    for Om0, w0, wa in models:
        for a in avals:
            g = _gamma(c, a, Om0, w0, wa)
            if g < 0.1 or g > 1.2:
                return False
    return True

def construct():
    t0 = time.time()
    # validation-like cosmologies (disjoint from seed; use a spread)
    models = [
        (0.30, -1.0, 0.0),
        (0.30, -1.0, 0.5),
        (0.28, -0.9, 0.3),
        (0.32, -1.1, -0.3),
        (0.25, -1.0, 0.0),
        (0.35, -0.8, 0.4),
    ]
    avals = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    # build reference caches (f and D) once
    f_cache = []
    D_cache = []
    for (Om0, w0, wa) in models:
        fc = {}
        Dc = {}
        for a in avals:
            # f reference from exact ODE derivative at a (or use ODE y')
            # cheaper: integrate ODE and record y' at a
            # We'll just compute f via the exact formula using H, H'
            def Hinv(a_):
                return math.sqrt(1.0 / (_Om(a_, Om0, w0, wa) + (1.0 - Om0) * (a_ / 1.0)**(-3.0 * (1.0 + w0 + wa))))
            def f_exact(a_):
                Om = _Om(a_, Om0, w0, wa)
                H = 1.0 / Hinv(a_)
                H2 = H * H
                H2p = 1.5 * H * (Om - (1.0 + w0 + wa * (1.0 - a_)) * (1.0 - Om))
                return 0.5 * Om - 1.5 * H2p / (2.0 * H2)
            fc[a] = f_exact(a)
            Dc[a] = _solve_ode(a, Om0, w0, wa)
        f_cache.append(fc)
        D_cache.append(Dc)

    # start from constant-gamma seed
    best_c = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_score = _score(best_c, models, avals, f_cache, D_cache)

    # coordinate / Nelder-Mead style local refinement
    def nm(c0, max_iter=200, tol=1e-6):
        c = list(c0)
        # initial simplex
        pts = [list(c)]
        for i in range(6):
            cc = list(c)
            cc[i] += 0.05
            pts.append(cc)
        scores = [_score(p, models, avals, f_cache, D_cache) for p in pts]
        for it in range(max_iter):
            if time.time() > DEADLINE:
                break
            order = np.argsort(scores)
            pts = [pts[i] for i in order]
            scores = [scores[i] for i in order]
            if abs(scores[-1] - scores[0]) < tol:
                break
            centroid = np.mean(pts[:-1], axis=0)
            worst = pts[-1]
            # reflect
            r = centroid + 1.0 * (centroid - np.array(worst))
            sr = _score(r, models, avals, f_cache, D_cache)
            if sr < scores[0]:
                # expand
                e = centroid + 2.0 * (r - centroid)
                se = _score(e, models, avals, f_cache, D_cache)
                if se < sr:
                    pts[-1] = list(e); scores[-1] = se
                else:
                    pts[-1] = list(r); scores[-1] = sr
            elif sr < scores[-2]:
                pts[-1] = list(r); scores[-1] = sr
            else:
                # contract
                ctc = centroid + 0.5 * (np.array(worst) - centroid)
                sc = _score(ctc, models, avals, f_cache, D_cache)
                if sc < scores[-1]:
                    pts[-1] = list(ctc); scores[-1] = sc
                else:
                    # shrink
                    base = pts[0]
                    for i in range(1, 7):
                        pts[i] = [base[j] + 0.5 * (pts[i][j] - base[j]) for j in range(6)]
                        scores[i] = _score(pts[i], models, avals, f_cache, D_cache)
        order = np.argsort(scores)
        return pts[order[0]], scores[order[0]]

    # a few random restarts + polish
    rng = np.random.default_rng(0)
    for restart in range(20):
        if time.time() > DEADLINE:
            break
        c0 = [0.55] + list(rng.uniform(-0.15, 0.15, 5))
        # clamp gamma range during search by soft projection
        c0 = [max(-10.0, min(10.0, x)) for x in c0]
        c_new, s_new = nm(c0, max_iter=120)
        # enforce coefficient bounds
        c_new = [max(-10.0, min(10.0, x)) for x in c_new]
        if not _gamma_range_ok(c_new, models, avals):
            # nudge back toward seed if gamma out of range
            c_new = [0.5 * x + 0.5 * best_c[i] for i, x in enumerate(c_new)]
            c_new = [max(-10.0, min(10.0, x)) for x in c_new]
            if not _gamma_range_ok(c_new, models, avals):
                continue
        s_new = _score(c_new, models, avals, f_cache, D_cache)
        if s_new < best_score:
            best_score = s_new
            best_c = c_new

    # final high-precision polish from best
    if time.time() < DEADLINE - 2:
        c_pol, s_pol = nm(best_c, max_iter=300, tol=1e-8)
        c_pol = [max(-10.0, min(10.0, x)) for x in c_pol]
        if _gamma_range_ok(c_pol, models, avals):
            s_pol = _score(c_pol, models, avals, f_cache, D_cache)
            if s_pol < best_score:
                best_score = s_pol
                best_c = c_pol

    # ensure finite and in range
    out = []
    for x in best_c:
        if not math.isfinite(x):
            x = 0.55
        out.append(max(-10.0, min(10.0, float(x))))
    return out
