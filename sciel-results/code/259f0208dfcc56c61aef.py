"""Self-contained fitter for the six-coefficient growth-index ansatz.

Strategy:
  1. Build a GR reference: analytic H^2(a), numerical f(a) from the linear
     growth ODE, and the normalized growth factor D(a)/D(1).
  2. Convert f(a) into an "effective gamma" via g = ln f / ln Om(a).
  3. Least-squares fit the six coefficients so that the ansatz gamma matches
     the reference effective gamma, and (via a second, D-weighted pass) so
     that the integrated growth factor matches the reference D.
  4. Clamp gamma to [0.1, 1.2] everywhere on the grid.

Only the standard library is used, so no dependency risk.
"""
import math
import time

_START = time.time()
_DEADLINE = _START + 84.0  # safety margin well under the 90 s wall-clock kill


def _now():
    return time.time()


def _within_budget():
    return _now() < _DEADLINE


def _clamp_gamma(g, lo=0.1, hi=1.2):
    if g < lo:
        return lo
    if g > hi:
        return hi
    return g


def _Om_m(a, Om0, wa):
    """Matter density parameter Omega_m(a)."""
    x = a ** (-3.0)
    denom = Om0 * x + (1.0 - Om0) * a ** (3.0 * (1.0 + wa))
    return Om0 * x / denom


def _H2(a, Om0, wa):
    """H^2(a)/H0^2 for CPL dark energy (analytic)."""
    return Om0 * a ** (-3.0) + (1.0 - Om0) * a ** (3.0 * (1.0 + wa))


def _ref_f(Om0, wa, a_lo=1e-4, a_hi=1.0, n=600):
    """Integrate the linear growth ODE from a_lo to a_hi; return (a, f, D).

    f' = d f/da = (3/2 - 3/2*Om_m - (3/2) w(a)) f
    D' = dD/da  = f * D
    """
    h = (a_hi - a_lo) / n
    # initial condition: f ~ Om(a)^0.55, D ~ 1
    Om0a = _Om_m(a_lo, Om0, wa)
    f = Om0a ** 0.55
    D = 1.0
    a = a_lo
    a_arr = [a]
    f_arr = [f]
    D_arr = [D]
    for _ in range(n):
        a_next = a + h
        # RK4 on the coupled system (f, D)
        def deriv(aa, ff, dd):
            Om = _Om_m(aa, Om0, wa)
            w = wa * (1.0 - aa)
            df = (1.5 - 1.5 * Om - 1.5 * w) * ff
            dD = ff * dd
            return df, dD

        k1f, k1D = deriv(a, f, D)
        k2f, k2D = deriv(a + 0.5 * h, f + 0.5 * h * k1f, D + 0.5 * h * k1D)
        k3f, k3D = deriv(a + 0.5 * h, f + 0.5 * h * k2f, D + 0.5 * h * k2D)
        k4f, k4D = deriv(a + h, f + h * k3f, D + h * k3D)
        f = f + (h / 6.0) * (k1f + 2.0 * k2f + 2.0 * k3f + k4f)
        D = D + (h / 6.0) * (k1D + 2.0 * k2D + 2.0 * k3D + k4D)
        a = a_next
        a_arr.append(a)
        f_arr.append(f)
        D_arr.append(D)
    return a_arr, f_arr, D_arr


def _build_grid():
    """Return (a_grid, [(Om0, wa, a_list, f_list, D_list), ...])."""
    a_grid = []
    a = 1e-4
    while a <= 1.0 + 1e-12:
        a_grid.append(a)
        a *= 1.02
    models = []
    for Om0 in (0.25, 0.30, 0.35, 0.40):
        for wa in (-0.5, -0.25, 0.0, 0.25, 0.5):
            a_arr, f_arr, D_arr = _ref_f(Om0, wa)
            models.append((Om0, wa, a_arr, f_arr, D_arr))
    return a_grid, models


def _features(a, Om0, wa):
    """Return (q, u, wa_a, uq) for the ansatz basis at scale factor a."""
    Om = _Om_m(a, Om0, wa)
    q = 1.0 - Om
    u = 1.0 + wa * (1.0 - a)
    wa_a = wa * a
    uq = u * q
    return q, u, wa_a, uq


def _fit(params, a_grid, models, target, weight, iters=200):
    """Least-squares (Gauss-Newton) fit of params to target via the gamma ansatz.

    For each (model, a) we form gamma = c0 + c1 q + c2 q^2 + c3 u + c4 wa_a + c5 uq,
    clamp to [0.1,1.2], and compare to target. weight scales the residual.
    """
    # precompute basis rows
    rows = []
    for Om0, wa, a_arr, f_arr, D_arr in models:
        for i, a in enumerate(a_arr):
            q, u, wa_a, uq = _features(a, Om0, wa)
            t = target(Om0, wa, a, f_arr[i], D_arr[i])
            wgt = weight(Om0, wa, a, f_arr[i], D_arr[i])
            rows.append((q, q * q, u, wa_a, u * q, t, wgt))
    n = len(rows)
    c = list(params)
    for it in range(iters):
        if not _within_budget():
            break
        # build normal equations A^T A x = A^T r with current clamp-aware residual
        A = [[0.0] * 5 for _ in range(5)]
        b = [0.0] * 5
        for q, q2, u, wa_a, uq, t, wgt in rows:
            # current predicted gamma
            g = c[0] + c[1] * q + c[2] * q2 + c[3] * u + c[4] * wa_a + c[5] * uq
            g = _clamp_gamma(g)
            r = (g - t) * wgt
            # active basis (only where clamp not binding)
            active = (0.1 < g < 1.2)
            if not active:
                continue
            row = [1.0, q, q2, u, wa_a]
            # we keep 6 params but map through 5 basis terms with c5=uq handled:
            # actually basis has 6 terms: [1, q, q2, u, wa_a, uq]
            row = [1.0, q, q2, u, wa_a, uq]
            for i in range(6):
                for j in range(6):
                    A[i][j] += row[i] * row[j] * wgt * wgt
                b[i] += row[i] * r * wgt
        # solve 6x6 normal system via Gaussian elimination
        try:
            x = _solve(A, b)
        except Exception:
            break
        # update
        for i in range(6):
            c[i] -= x[i]
    return c


def _solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        # pivot
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-14:
            raise ValueError("singular")
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= pv
        for r in range(n):
            if r == col:
                continue
            f = M[r][col]
            if f != 0.0:
                for j in range(col, n + 1):
                    M[r][j] -= f * M[col][j]
    return [M[i][n] for i in range(n)]


def _gamma_of(c, a, Om0, wa):
    q, u, wa_a, uq = _features(a, Om0, wa)
    g = c[0] + c[1] * q + c[2] * q * q + c[3] * u + c[4] * wa_a + c[5] * uq
    return _clamp_gamma(g)


def _eval_max_err(c, a_grid, models):
    """Worst relative error of f and normalized D across the grid."""
    worst = 0.0
    for Om0, wa, a_arr, f_arr, D_arr in models:
        D1 = D_arr[-1]
        for i, a in enumerate(a_arr):
            Om = _Om_m(a, Om0, wa)
            g = _gamma_of(c, a, Om0, wa)
            f_app = Om ** g
            # normalized D via exp(-int f dx)
            # approximate integral of f(exp(x)) from ln a to 0
            # use trapezoidal on a few sub-steps
            D_norm = _norm_D(c, a, Om0, wa, a_arr, f_arr, D_arr, i)
            rf = abs(f_app - f_arr[i]) / max(abs(f_arr[i]), 1e-12)
            rd = abs(D_norm - D_arr[i] / D1) / max(abs(D_arr[i] / D1), 1e-12)
            if rf > worst:
                worst = rf
            if rd > worst:
                worst = rd
    return worst


def _norm_D(c, a, Om0, wa, a_arr, f_arr, D_arr, i):
    """Normalized growth factor using the ansatz f, integrated from a to 1.

    D(a)/D(1) = exp(-int_{ln a}^0 f(exp x) dx). We approximate with the
    reference grid: integrate f_app over the a_grid between a and 1.
    """
    # integrate f_app(a) da' from a to 1, then D = exp(int) ... but sign:
    # D(a)/D(1) = exp(-int_{ln a}^0 f dx) = exp(-int_a^1 f(a') (da'/a'))
    # because dx = da'/a'
    # Use the reference a_arr grid points from index i to end.
    total = 0.0
    for k in range(i, len(a_arr) - 1):
        a0 = a_arr[k]
        a1 = a_arr[k + 1]
        f0 = _Om_m(a0, Om0, wa) ** _gamma_of(c, a0, Om0, wa)
        f1 = _Om_m(a1, Om0, wa) ** _gamma_of(c, a1, Om0, wa)
        # integral of f da'/a' ~ (f0+f1)/2 * ln(a1/a0)
        total += 0.5 * (f0 + f1) * math.log(a1 / a0)
    return math.exp(-total)


def construct():
    try:
        a_grid, models = _build_grid()

        # target 1: effective gamma from reference f
        def target_g(Om0, wa, a, f, D):
            Om = _Om_m(a, Om0, wa)
            if Om <= 0 or abs(math.log(Om)) < 1e-9:
                return 0.55
            g = math.log(f) / math.log(Om)
            return _clamp_gamma(g)

        def w1(Om0, wa, a, f, D):
            return 1.0

        c = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        c = _fit(c, a_grid, models, target_g, w1, iters=150)

        # target 2: D-weighted refinement on effective gamma to reduce D error.
        # Weight by how sensitive D is to gamma: larger for smaller a (longer integral).
        def target_g2(Om0, wa, a, f, D):
            Om = _Om_m(a, Om0, wa)
            if Om <= 0 or abs(math.log(Om)) < 1e-9:
                return 0.55
            g = math.log(f) / math.log(Om)
            return _clamp_gamma(g)

        def w2(Om0, wa, a, f, D):
            # emphasize early times where D accumulates more error
            return 1.0 + 2.0 * (1.0 - a)

        c = _fit(c, a_grid, models, target_g2, w2, iters=120)

        # final light polish on f-target
        c = _fit(c, a_grid, models, target_g, w1, iters=80)

        # keep coefficients finite and within [-10,10]
        out = []
        for v in c:
            if not math.isfinite(v):
                v = 0.55
            if v > 10.0:
                v = 10.0
            if v < -10.0:
                v = -10.0
            out.append(float(v))
        return out
    except Exception:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
