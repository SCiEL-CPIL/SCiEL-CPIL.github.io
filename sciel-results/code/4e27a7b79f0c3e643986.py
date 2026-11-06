"""Self-optimizing growth-index fitter. SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 85.0          # safety margin before the 90s kill
    GMIN, GMAX = 0.1, 1.2

    def gamma_of(a, c, Om, w0, wa):
        q = 1.0 - Om
        u = 1.0 + w0 + wa * (1.0 - a)
        g = (c[0] + c[1] * q + c[2] * q * q + c[3] * u
             + c[4] * wa * a + c[5] * u * q)
        if g < GMIN:
            g = GMIN
        elif g > GMAX:
            g = GMAX
        return g

    def H2_of_x(x, Om0, w0, wa):
        a = np.exp(x)
        Om = Om0 * a ** (-3.0)
        Ode = (1.0 - Om0) * a ** (-3.0 * (1.0 + w0 + wa))
        return Om + Ode

    def solve_growth(Om0, w0, wa, xg):
        # RK4 backward from x=0 (a=1) to xg (a<1)
        n = xg.size
        D = np.empty(n)
        f = np.empty(n)
        D[0] = 1.0
        f[0] = Om0 ** 0.55
        h = xg[1] - xg[0]
        for i in range(1, n):
            x = xg[i - 1]
            dv = f
            df = -0.5 * H2_of_x(x, Om0, w0, wa) * f - 1.5 * Om0 * np.exp(-3.0 * x) * D
            k1D, k1f = dv, df
            xh = x + 0.5 * h
            k2D, k2f = f + 0.5 * h * k1f, (f + 0.5 * h * k1f) - 0.5 * h * H2_of_x(xh, Om0, w0, wa) * (f + 0.5 * h * k1f) - 1.5 * Om0 * np.exp(-3.0 * xh) * (D + 0.5 * h * k1D)
            xh = x + 0.5 * h
            k3D, k3f = f + 0.5 * h * k2f, (f + 0.5 * h * k2f) - 0.5 * h * H2_of_x(xh, Om0, w0, wa) * (f + 0.5 * h * k2f) - 1.5 * Om0 * np.exp(-3.0 * xh) * (D + 0.5 * h * k2D)
            xh = x + h
            k4D, k4f = f + h * k3f, (f + h * k3f) - h * H2_of_x(xh, Om0, w0, wa) * (f + h * k3f) - 1.5 * Om0 * np.exp(-3.0 * xh) * (D + h * k3D)
            D[i] = D + (h / 6.0) * (k1D + 2.0 * k2D + 2.0 * k3D + k4D)
            f[i] = f + (h / 6.0) * (k1f + 2.0 * k2f + 2.0 * k3f + k4f)
        return D, f

    def D_from_f(fv, xg):
        D = np.empty(fv.size)
        D[-1] = 1.0
        h = xg[1] - xg[0]
        for i in range(fv.size - 2, -1, -1):
            D[i] = D[i + 1] * np.exp(h * 0.5 * (fv[i] + fv[i + 1]))
        return D

    # Validation cosmologies (CPL dark energy)
    models = [(0.30, 0.0, 0.0), (0.30, -1.0, 0.0), (0.30, -1.0, 1.0),
              (0.30, -1.0, -0.5), (0.30, -1.0, 0.5), (0.25, -1.0, 0.5),
              (0.35, -1.0, 1.0), (0.30, -1.2, 0.5)]
    agrid = np.linspace(0.1, 1.0, 25)
    xg = np.log(agrid)
    ref = {}
    for m in models:
        Om0, w0, wa = m
        Dref, fref = solve_growth(Om0, w0, wa, xg)
        ref[m] = (Dref, fref)

    def objective(c):
        c = np.clip(c, -10.0, 10.0)
        worst = 0.0
        for m in models:
            Om0, w0, wa = m
            Om = Om0 * agrid ** (-3.0)
            g = gamma_of(agrid, c, Om, w0, wa)
            f = Om ** g
            D = D_from_f(f, xg)
            Dref, fref = ref[m]
            e1 = np.max(np.abs(f - fref) / np.abs(fref))
            e2 = np.max(np.abs(D - Dref) / np.abs(Dref))
            e = e1 if e1 > e2 else e2
            if e > worst:
                worst = e
        return worst

    def feasible(c):
        for v in c:
            if v < -10.0 or v > 10.0:
                return False
        return True

    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best = c0.copy()
    bestval = objective(best)

    try:
        def objwrap(x):
            x = np.clip(x, -10.0, 10.0)
            return objective(x)

        res = minimize(objwrap, c0, method='Nelder-Mead',
                       options={'maxiter': 4000, 'xatol': 1e-5, 'fatol': 1e-7,
                                'adaptive': True})
        cand = np.clip(res.x, -10.0, 10.0)
        cv = objective(cand)
        if cv < bestval:
            best, bestval = cand, cv

        # local polish
        for _ in range(3):
            if time.time() > DEADLINE:
                break
            r2 = minimize(objwrap, best, method='Nelder-Mead',
                          options={'maxiter': 1500, 'xatol': 1e-6, 'fatol': 1e-8,
                                   'adaptive': True})
            c2 = np.clip(r2.x, -10.0, 10.0)
            v2 = objective(c2)
            if v2 < bestval:
                best, bestval = c2, v2

        if not feasible(best):
            best = c0
    except Exception:
        pass

    out = [float(v) for v in best]
    for i in range(6):
        if not np.isfinite(out[i]):
            out[i] = 0.0
        if out[i] > 10.0:
            out[i] = 10.0
        elif out[i] < -10.0:
            out[i] = -10.0
    return out
# EVOLVE-BLOCK-END
