# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv


def _build_quadrature():
    """70-pt Gauss-Legendre in x=cos(theta) x 130 uniform phi, with the
    surface-measure weights (total mass 4*pi)."""
    n_x, n_p = 70, 130
    x, w = np.polynomial.legendre.leggauss(n_x)
    phi = (np.arange(n_p) + 0.5) * (2.0 * np.pi / n_p)
    X, PH = np.meshgrid(x, phi, indexing='ij')
    W = (w[:, None] * (2.0 * np.pi / n_p))[None, :, :]
    W = np.broadcast_to(W, (n_x, n_p, n_p)).copy()
    W = W * np.sqrt(1.0 - X**2)
    return X, PH, W


def _build_basis(X, PH):
    """Real orthonormal degree-10 harmonics, returned as (21, n_x, n_p)."""
    k = 10
    N = np.array([np.sqrt((2 * k + 1) / (4.0 * np.pi) *
                          float(np.math.factorial(k - m) / np.math.factorial(k + m)))
                  for m in range(k + 1)])
    P = np.empty((k + 1, X.shape[0], X.shape[1]))
    for m in range(k + 1):
        P[m] = lpmv(m, k, X) * N[m]
    B = np.empty((2 * k + 1, X.shape[0], X.shape[1]))
    for m in range(0, k + 1):
        B[m] = P[m]
    for m in range(1, k + 1):
        B[k + m] = np.sqrt(2.0) * P[m] * np.cos(m * PH)
        B[k - m] = np.sqrt(2.0) * P[m] * np.sin(m * PH)
    return B


def _compute_ratio(c, B, W):
    """R = ||f||_4 / ||f||_2  (and the mean of f^2, f^4)."""
    f = c @ B
    s2 = float(np.sum(W * f * f))
    s4 = float(np.sum(W * f * f * f * f))
    if s2 <= 0.0:
        return np.inf, s2, s4
    return np.sqrt(s4) / s2, s2, s4


def _optimize(c0, B, W, deadline):
    """Gauss-Newton (Levenberg-Marquardt) minimizer of R^2 = <f^4>/<f^2>^2."""
    n = c0.size
    c = c0.astype(np.float64).copy()
    f = c @ B
    s2 = float(np.sum(W * f * f))
    s4 = float(np.sum(W * f * f * f * f))
    if s2 <= 0.0:
        return c0
    mean2 = s2 / 4.0
    best = c.copy()
    best_R = np.sqrt(s4) / s2
    lam = 1e-3
    for _ in range(800):
        if time.time() > deadline:
            break
        f2 = f * f
        g = (2.0 / mean2) * np.einsum('ij,ijk->k', B, W * (f2 - mean2) * f2)
        H = 4.0 * np.einsum('ikj,ijk->ij', B, W * f2)
        rhs = H @ c - g
        A = H + lam * np.diag(np.diag(H))
        try:
            step = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue
        trial = c + step
        ft = trial @ B
        s2t = float(np.sum(W * ft * ft))
        if s2t <= 0.0:
            lam *= 10.0
            continue
        Rt = np.sqrt(float(np.sum(W * ft * ft * ft * ft))) / s2t
        if Rt < best_R:
            c = trial
            f = ft
            s2 = s2t
            best = c.copy()
            best_R = Rt
            lam = max(lam * 0.3, 1e-12)
        else:
            lam *= 10.0
            if lam > 1e15:
                break
        if np.max(np.abs(step)) < 1e-14 * (np.max(np.abs(c)) + 1e-30):
            break
    return best


def rudin_problem10() -> np.ndarray:
    start = time.time()
    deadline = start + 615.0
    X, PH, W = _build_quadrature()
    B = _build_basis(X, PH)
    k = 10
    n = 2 * k + 1
    # seed: all ones (the reference baseline)
    c_seed = np.ones(n, dtype=np.float64)
    best = c_seed
    best_R = _compute_ratio(c_seed, B, W)[0]
    rng = np.random.default_rng(101)
    while time.time() < deadline:
        c0 = rng.normal(0.0, 1.0, size=n)
        c0 = c0 / (np.linalg.norm(c0) + 1e-30)
        cand = _optimize(c0, B, W, deadline)
        R = _compute_ratio(cand, B, W)[0]
        if R < best_R:
            best_R = R
            best = cand
    out = best.astype(np.float64)
    if not np.isfinite(out).all():
        out = c_seed
    return out


# EVOLVE-BLOCK-END
