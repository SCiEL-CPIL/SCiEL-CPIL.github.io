# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize


def _build_harmonics():
    """Return (mu, phi, Y) where Y has shape (21, nmu, nphi).
    Y[i] corresponds to m = i - 10."""
    k = 10
    nmu, nphi = 70, 130
    mu, wmu = np.polynomial.legendre.leggauss(nmu)
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    MU, PHI = np.meshgrid(mu, phi, indexing='ij')  # (nmu, nphi)
    cosphi = np.cos(PHI)
    sinphi = np.sin(PHI)

    Y = np.zeros((2 * k + 1, nmu, nphi), dtype=np.float64)
    for m in range(-k, k + 1):
        am = abs(m)
        # associated Legendre P_l^m(cos theta) with Condon-Shortley phase
        P = lpmv(am, k, mu)  # (nmu,)
        P2 = np.broadcast_to(P[:, None], (nmu, nphi))
        norm = np.sqrt((2 * k + 1) / (4.0 * np.pi) *
                       float(np.math.factorial(k - am)) /
                       float(np.math.factorial(k + am)))
        if m == 0:
            Y[m + k] = norm * P2
        elif m > 0:
            Y[m + k] = np.sqrt(2.0) * norm * P2 * cosphi ** 0 * (
                np.cos(m * PHI) if False else np.cos(m * PHI))
            Y[m + k] = np.sqrt(2.0) * norm * P2 * np.cos(m * PHI)
        else:
            Y[m + k] = np.sqrt(2.0) * norm * P2 * np.sin(am * PHI)
    return mu, wmu, Y


def _ratio(coeffs, Y, wmu):
    """R = ||f||_4 / ||f||_2 over the grid with surface weights."""
    f = np.tensordot(coeffs, Y, axes=(0, 0))  # (nmu, nphi)
    w = wmu[:, None]
    f2 = f * f
    l2 = np.sum(f2 * w)
    if l2 <= 0.0:
        return np.inf
    l4 = np.sum(f2 * f2 * w)
    return (l4 / l2) ** 0.5  # (||f||_4^4/||f||_2^2)^{1/2} = ||f||_4/||f||_2


def rudin_problem10() -> np.ndarray:
    k = 10
    mu, wmu, Y = _build_harmonics()

    # objective: minimize R (scale invariant). Use a smooth surrogate:
    # minimize log-like via minimizing (L4/L2) directly.
    def obj(c):
        return _ratio(c, Y, wmu)

    # Several diverse starts.
    rng = np.random.default_rng(12345)
    starts = []
    starts.append(np.ones(2 * k + 1, dtype=np.float64))
    for _ in range(12):
        v = rng.standard_normal(2 * k + 1)
        starts.append(v / np.linalg.norm(v))
    # A few structured "flat" candidates: emphasize low |m| (more isotropic)
    for w0 in (0.5, 1.0, 2.0):
        v = np.array([np.exp(-w0 * (m - 0) ** 2) for m in range(-k, k + 1)])
        starts.append(v / np.linalg.norm(v))

    best_c = None
    best_val = np.inf
    for s in starts:
        try:
            res = minimize(obj, s, method='L-BFGS-B',
                           options={'maxiter': 400, 'gtol': 1e-10})
            c = res.x
            val = obj(c)
            if val < best_val:
                best_val = val
                best_c = c
        except Exception:
            continue

    # Polish with Nelder-Mead from the best
    if best_c is not None:
        try:
            res2 = minimize(obj, best_c, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-12,
                                     'fatol': 1e-14})
            c = res2.x
            val = obj(c)
            if val < best_val:
                best_val = val
                best_c = c
        except Exception:
            pass

    if best_c is None:
        best_c = np.ones(2 * k + 1, dtype=np.float64)

    c = np.asarray(best_c, dtype=np.float64)
    # Sanitize: ensure finite and within magnitude bounds.
    if not np.all(np.isfinite(c)):
        c = np.ones(2 * k + 1, dtype=np.float64)
    mx = np.max(np.abs(c))
    if mx < 1e-3:
        c = c / mx * 1e-1 if mx > 0 else np.ones(2 * k + 1)
        mx = np.max(np.abs(c))
    if mx > 1e5:
        c = c / mx * 1e5
    return c


# EVOLVE-BLOCK-END
