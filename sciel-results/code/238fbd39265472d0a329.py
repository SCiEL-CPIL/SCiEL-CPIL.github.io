# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 620.0  # safety margin under 630 s kill
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _precompute():
    """Build the 70x130 quadrature grid and the 21 real orthonormal degree-10 basis."""
    k = 10
    n_theta = 70
    n_phi = 130

    # Gauss-Legendre nodes/weights on [-1, 1]
    nodes, weights = np.polynomial.legendre.leggauss(n_theta)
    mu = nodes            # cos theta
    w_theta = weights     # integrates over mu in [-1,1]

    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    w_phi = 1.0 / n_phi   # integrates over phi in [0, 2pi)

    # Associated Legendre P_l^m with Condon-Shortley phase, via recurrence
    # P_0^0=1, P_1^0=x; P_l^0 from three-term; P_l^1 = -(1-x^2)^{1/2} P_{l-1}' ;
    # then P_l^{m+1} = (2l-1) x P_l^m - (l+m)(l-m+1) P_l^{m-1}  (includes CS phase)
    def P_table(l, x):
        # returns array P[m, :] for m=0..l
        P = np.zeros((l + 1, x.size))
        P[0, :] = 1.0
        if l >= 1:
            P[1, :] = x
            for ll in range(2, l + 1):
                P[ll, :] = ((2 * ll - 1) * x * P[ll - 1, :] - (ll - 1) * P[ll - 2, :]) / ll
        # P_l^1
        Pp = np.zeros((l + 1, x.size))  # derivatives
        if l >= 1:
            Pp[1, :] = 1.0
            for ll in range(2, l + 1):
                Pp[ll, :] = ((2 * ll - 1) * (x * Pp[ll - 1, :] - P[ll - 1, :]) - (ll - 1) * Pp[ll - 2, :]) / ll
        P[1, :] = -np.sqrt(np.maximum(0.0, 1.0 - x * x)) * Pp[l, :]
        for m in range(1, l):
            P[m + 1, :] = (2 * l - 1) * x * P[m, :] - (l + m) * (l - m + 1) * P[m - 1, :]
        return P

    P = P_table(k, mu)  # P[m, :] for m=0..k

    # Normalization N_m = sqrt( (2l+1)/(4pi) * (l-m)!/(l+m)! )
    def N(m):
        return np.sqrt((2 * k + 1) / (4.0 * np.pi) *
                       float(np.math.factorial(k - m)) / float(np.math.factorial(k + m)))

    # Build basis: index i -> m = i - k
    basis = np.zeros((2 * k + 1, n_theta, n_phi))
    for i in range(2 * k + 1):
        m = i - k
        if m == 0:
            basis[i] = N(0) * P[0, :][:, None] * np.ones((1, n_phi))
        elif m > 0:
            basis[i] = np.sqrt(2.0) * N(m) * P[m, :][:, None] * np.cos(m * phi)[None, :]
        else:
            mm = -m
            basis[i] = np.sqrt(2.0) * N(mm) * P[mm, :][:, None] * np.sin(mm * phi)[None, :]

    # Surface measure weights: dOmega = dmu dphi
    w = (w_theta[:, None] * w_phi[None, :]).ravel()  # size n_theta*n_phi
    basis_flat = basis.reshape(2 * k + 1, -1)  # (21, n)

    return basis_flat, w


_BASIS, _W = _precompute()
_NPTS = _BASIS.shape[1]


def _R_and_grad(c):
    """Return (R, grad) where R = ||f||_4/||f||_2 and grad = dR/dc."""
    f = _BASIS @ c                      # (n,)
    absf = np.abs(f)
    n4 = np.sum(w * absf ** 4)
    n2 = np.sum(w * f * f)
    if n4 <= 0 or n2 <= 0:
        return np.inf, np.zeros_like(c)
    R4 = n4 ** 0.25
    R2 = np.sqrt(n2)
    R = R4 / R2

    # d/dx (x^4) = 4 x^3 ; d/dx (x^2) = 2 x
    # dR = (1/2) R4^{-1} R2^{-1} [ 4 <w |f|^3 f> R2 - 2 <w |f|> R4^2 * (R4/R2) ] / R2
    # Simplify: R = R4 R2^{-1}
    # dR = R4^{-1} R2^{-1} * (1/2) [ 4 A R2 - 2 B R4^2 (R4/R2) ]  where A=<w|f|^3 f>, B=<w|f|>
    #     = R4^{-1} R2^{-1} * [ 2 A R2 - B R4^2 (R4/R2) ]
    A = np.sum(w * absf ** 3 * f)
    B = np.sum(w * absf)
    dR = R4 ** (-1) * R2 ** (-1) * (2.0 * A * R2 - B * R4 ** 2 * (R4 / R2))
    # Actually recompute carefully:
    # R = R4 / R2
    # dR = (dR4 * R2 - R4 * dR2) / R2^2
    # dR4 = (1/4) R4^{-3} * 4 A = A / R4^3
    # dR2 = (1/2) R2^{-1} * 2 B = B / R2
    # dR = ( (A/R4^3) R2 - R4 (B/R2) ) / R2^2
    #     = A R2 / (R4^3 R2^2) - R4 B / (R2 R2^2)
    #     = A / (R4^3 R2) - R4 B / R2^3
    dR = A / (R4 ** 3 * R2) - R4 * B / (R2 ** 3)
    grad = dR * (_BASIS.T @ f)
    return R, grad


def _valid(c):
    if c.size != 21:
        return False
    if not np.all(np.isfinite(c)):
        return False
    m = np.max(np.abs(c))
    return (m >= 1e-3) and (m <= 1e5)


def _line_search(c, g, R0, max_iter=60):
    """Backtracking line search minimizing R."""
    step = 0.5
    for _ in range(max_iter):
        c_new = c - step * g
        R_new, _ = _R_and_grad(c_new)
        if np.isfinite(R_new) and R_new < R0:
            return c_new, R_new, step
        step *= 0.5
    return c, R0, 0.0


def rudin_problem10() -> np.ndarray:
    k = 10
    # Start from a non-degenerate point
    rng = np.random.default_rng(12345)
    c = rng.normal(size=2 * k + 1)
    c = np.maximum(np.abs(c), 1e-2)  # avoid near-zero
    c = np.sign(c) * np.maximum(np.abs(c), 1e-2)

    R, g = _R_and_grad(c)
    best_c = c.copy()
    best_R = R

    # Gradient descent with adaptive step
    lr = 0.5
    stall = 0
    it = 0
    while _time_left() > 30.0:
        if not np.isfinite(R) or not np.all(np.isfinite(g)):
            break
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-12:
            break
        g = g / g_norm  # normalize to control step size
        c_new, R_new, step = _line_search(c, g, R, max_iter=40)
        if step == 0.0:
            stall += 1
            lr *= 0.5
            if stall > 15:
                # perturb to escape
                c_new = c + 0.05 * rng.normal(size=c.size)
                R_new, _ = _R_and_grad(c_new)
                if not np.isfinite(R_new):
                    break
                stall = 0
            else:
                c_new = c - lr * g
                R_new, _ = _R_and_grad(c_new)
                if not np.isfinite(R_new):
                    break
        else:
            stall = 0
        if R_new < best_R:
            best_R = R_new
            best_c = c_new.copy()
        c = c_new
        R = R_new
        it += 1
        if it % 200 == 0 and _time_left() < 60.0:
            break

    # Final polish: tighter local solve via more iterations
    c = best_c.copy()
    R, g = _R_and_grad(c)
    for _ in range(2000):
        if _time_left() < 10.0:
            break
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-14:
            break
        g = g / g_norm
        c_new, R_new, step = _line_search(c, g, R, max_iter=60)
        if step == 0.0:
            break
        c = c_new
        R = R_new
        if R < best_R:
            best_R = R
            best_c = c.copy()

    out = best_c
    if not _valid(out):
        # fallback to a valid point
        out = np.ones(2 * k + 1, dtype=np.float64)
    return out.astype(np.float64)


# EVOLVE-BLOCK-END
