# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import factorial, lpmv
from scipy.optimize import minimize

_K = 10
_NGL = 70
_NPHI = 130
_M = _NGL * _NPHI
_TOL = 1e-12


def _build_basis():
    """Build the exact evaluator quadrature basis.

    Returns (B, w) with B of shape (M, 21) and w of shape (M,).
    f_vals = B @ c ;  norm4 = sqrt(sqrt(sum(w * f_vals**4)))
    """
    # 70-point Gauss-Legendre in x = cos(theta)
    x, wx = np.polynomial.legendre.leggauss(_NGL)
    # phi grid (equally spaced); sin(m phi) for m>0 uses sin(m*phi)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    mpos = np.arange(1, _K + 1)
    cosm = np.cos(np.outer(mpos, phi))          # (10, NPHI)
    sinm = np.sin(np.outer(mpos, phi))          # (10, NPHI)

    N = np.sqrt(21.0 / (4.0 * np.pi) *
                np.array([float(factorial(_K - m) / factorial(_K + m))
                          for m in range(_K + 1)]))
    N0 = N[0]

    B = np.empty((_M, 2 * _K + 1), dtype=np.float64)
    for m in range(_K + 1):
        Pm = lpmv(m, _K, x) * N[m]             # (NGL,)
        # m == 0
        B[:, m] = np.repeat(Pm, _NPHI)
        if m > 0:
            s2 = np.sqrt(2.0)
            B[:, _K + m] = np.tile(s2 * Pm[:, None] * cosm[m - 1], (_NGL, 1)).ravel()
            B[:, _K - m] = np.tile(s2 * Pm[:, None] * sinm[m - 1], (_NGL, 1)).ravel()

    w = np.repeat(wx, _NPHI)
    return B, w


_B, _W = _build_basis()
_WB = _W[:, None] * _B          # (M, 21) precomputed weighted basis (fast gradient)


def _norm4(c):
    f = _B @ c
    return np.sqrt(np.sqrt(np.sum(_W * f ** 4)))


def _norm4_grad(c):
    f = _B @ c
    # d/dc sum(w f^4) = 4 sum_m f^3 (w B_m) ;  d sqrt(sqrt(s)) = 0.5 s^{-1/4} * 0.5 s^{-1} = 0.25 s^{-3/4}
    s = np.sum(_W * f ** 4)
    return 2.0 * (f ** 3) @ _WB / (s ** 0.75)


def _project(c):
    n = np.linalg.norm(c)
    if n < 1e-300:
        return c
    return c / n


def _polish(c):
    """Sphere-constrained gradient descent with backtracking on ||f||4."""
    c = _project(c.astype(np.float64).copy())
    best = _norm4(c)
    d = 0
    while d < 20000:
        g = _norm4_grad(c)
        # project gradient onto tangent of the sphere
        g = g - np.dot(g, c) * c
        gn = np.linalg.norm(g)
        if gn < 1e-10:
            break
        d = d + 1
        p = -g / gn
        lr = 0.5
        improved = False
        for _ in range(40):
            nc = _project(c + lr * p)
            v = _norm4(nc)
            if v < best - 1e-15:
                c = nc
                best = v
                improved = True
                break
            lr *= 0.5
        if not improved:
            break
    return c, best


def _lbfgs(c0, deadline, maxit):
    c0 = _project(c0)
    try:
        res = minimize(_norm4, c0, jac=_norm4_grad, method='L-BFGS-B',
                       options={'maxiter': maxit, 'gtol': 1e-10, 'maxls': 20})
        if res.success or res.fun < _norm4(c0):
            c0 = res.x
    except Exception:
        pass
    return _project(c0)


def _valid(c):
    if c.shape != (2 * _K + 1,):
        return False
    if not np.all(np.isfinite(c)):
        return False
    mx = float(np.max(np.abs(c)))
    return (mx >= 1e-3) and (mx <= 1e5)


def rudin_problem10() -> np.ndarray:
    t0 = time.time()
    deadline = t0 + 600.0

    # ---- seed candidates (diverse) ----
    seeds = [np.ones(2 * _K + 1, dtype=np.float64)]
    rng = np.random.default_rng(12345)
    for _ in range(60):
        seeds.append(rng.standard_normal(2 * _K + 1))
    # symmetric cosine-only
    cs = np.zeros(2 * _K + 1)
    cs[_K] = 1.0
    cs[_K + 2] = 1.0
    cs[_K + 6] = 1.0
    seeds.append(cs)
    # sine-only
    cs2 = np.zeros(2 * _K + 1)
    cs2[_K - 1] = 1.0
    cs2[_K - 3] = 1.0
    cs2[_K - 7] = 1.0
    seeds.append(cs2)
    # low-m bias
    cl = np.zeros(2 * _K + 1)
    for m in range(1, 6):
        cl[_K + m] = 1.0
        cl[_K - m] = 1.0
    cl[_K] = 1.0
    seeds.append(cl)

    best_c = None
    best_val = np.inf
    for s in seeds:
        if time.time() > deadline - 30.0:
            break
        c = _lbfgs(s, deadline, maxit=400)
        c, v = _polish(c)
        if v < best_val:
            best_val = v
            best_c = c

    # ---- extra budget: more diverse starts + deeper polish ----
    n_extra = 0
    while time.time() < deadline - 60.0 and n_extra < 40:
        n_extra += 1
        s = rng.standard_normal(2 * _K + 1)
        if n_extra % 5 == 0:
            # triplet-ish structured start
            idx = rng.choice(2 * _K + 1, size=3, replace=False)
            s = np.zeros(2 * _K + 1)
            s[idx] = rng.choice([-1.0, 1.0], size=3)
        c = _lbfgs(s, deadline, maxit=200)
        c, v = _polish(c)
        if v < best_val:
            best_val = v
            best_c = c

    # ---- final high-precision polish on the best ----
    if best_c is not None and time.time() < deadline - 15.0:
        for _ in range(3):
            if time.time() > deadline - 10.0:
                break
            best_c, best_val = _polish(best_c)

    out = np.asarray(best_c, dtype=np.float64)
    if not _valid(out):
        out = np.ones(2 * _K + 1, dtype=np.float64)
    return out


# EVOLVE-BLOCK-END
