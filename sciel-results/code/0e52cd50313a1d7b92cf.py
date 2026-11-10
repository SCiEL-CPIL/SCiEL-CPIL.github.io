# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 600.0
_TIME_MARGIN = 30.0  # seconds before hard limit to stop


def _precompute_basis():
    """Precompute quadrature weights and the 21 degree-10 real orthonormal
    spherical harmonic values on the 70x130 grid."""
    from scipy.special import lpmv

    n_theta = 70
    n_phi = 130
    k = 10

    # Gauss-Legendre in u = cos(theta) in [-1, 1]
    u, wu = np.polynomial.legendre.leggauss(n_theta)
    theta = np.arccos(u)
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi

    # weights: dOmega = du dphi, total mass 4*pi
    # integrate over phi with 1/n_phi factor (uniform), over u with wu
    # So weight for each (theta,phi) point = wu[i] * (2*pi / n_phi)
    w_phi = 2.0 * np.pi / n_phi
    # Full weight matrix (n_theta, n_phi)
    W = np.outer(wu, np.full(n_phi, w_phi))
    W = W.ravel()  # length N = n_theta * n_phi
    N = W.size

    # Build basis Y_m for m = -10..10, flatten to (N, 21)
    # Y_0 = N_0 * P_10^0(cos theta)
    # Y_m = sqrt(2) N_m P_10^m(cos theta) cos(m phi) for m>0
    # Y_m = sqrt(2) N_|m| P_10^{|m|}(cos theta) sin(|m| phi) for m<0
    # N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
    # lpmv includes Condon-Shortley phase.
    Y = np.zeros((N, 21), dtype=np.float64)

    # precompute per-m angular part and normalization
    for m in range(-10, 11):
        idx = m + 10
        am = abs(m)
        # N_m
        from math import factorial, pi
        N_m = np.sqrt(21.0 / (4.0 * pi) * factorial(10 - am) / factorial(10 + am))
        # P_10^am(cos theta) with Condon-Shortley (lpmv includes it)
        P = lpmv(am, 10, theta)  # shape (n_theta,)
        # angular
        if m == 0:
            ang = np.ones(n_phi)
            scale = N_m
        elif m > 0:
            ang = np.cos(m * phi)
            scale = np.sqrt(2.0) * N_m
        else:
            ang = np.sin(am * phi)
            scale = np.sqrt(2.0) * N_m
        # Y on grid: outer(P * scale, ang) -> (n_theta, n_phi), ravel
        Y[:, idx] = (scale * P[:, None] * ang[None, :]).ravel()
    return W, Y


_W, _Y = _precompute_basis()
_N = _W.size
_YT = _Y.T  # (21, N) for gradient computation


def _objective_and_grad(c):
    """R(c) = (sum w f^4)^{1/4} / (sum w f^2)^{1/2} and its gradient."""
    f = _Y @ c  # (N,)
    f2 = f * f
    f4 = f2 * f2
    A = np.dot(_W, f4)
    B = np.dot(_W, f2)
    if A <= 0 or B <= 0:
        return 1e30, np.zeros_like(c)
    R = A ** 0.25 / np.sqrt(B)
    # gradient
    # dA_dc = 4 * sum w f^3 Y_i
    # dB_dc = 2 * sum w f Y_i
    wY = _W[:, None] * _Y  # (N, 21)
    dA_dc = 4.0 * np.dot(f ** 3 * _W, _Y)  # (21,)
    dB_dc = 2.0 * np.dot(f * _W, _Y)       # (21,)
    grad = 0.25 * A ** (-0.75) * B ** (-0.5) * dA_dc \
           - 0.5 * A ** 0.25 * B ** (-1.5) * dB_dc
    return R, grad


def _is_valid(c):
    if c is None or c.shape != (21,):
        return False
    if not np.all(np.isfinite(c)):
        return False
    m = np.max(np.abs(c))
    if m < 1e-3 or m > 1e5:
        return False
    return True


def rudin_problem10() -> np.ndarray:
    global _DEADLINE
    best_c = np.ones(21, dtype=np.float64)
    best_R = _objective_and_grad(best_c)[0]

    rng = np.random.default_rng(42)

    # Generate a set of diverse starting points
    starts = []
    # all ones
    starts.append(np.ones(21, dtype=np.float64))
    # random unit vectors
    for _ in range(12):
        v = rng.standard_normal(21)
        v /= np.linalg.norm(v)
        starts.append(v)
    # some structured: single mode, pairs
    for m in range(-10, 11):
        v = np.zeros(21)
        v[m + 10] = 1.0
        starts.append(v)
    # alternating signs
    alt = np.ones(21)
    alt[::2] = -1.0
    starts.append(alt)

    for s in starts:
        if time.time() > _DEADLINE - _TIME_MARGIN:
            break
        if not _is_valid(s):
            continue
        # normalize to reasonable scale
        s = s.copy()
        nrm = np.linalg.norm(s)
        if nrm > 0:
            s = s / nrm * 1.0
        try:
            res = minimize(
                _objective_and_grad,
                s,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12,
                         'maxls': 50, 'maxfun': 2000}
            )
        except Exception:
            continue
        if res.success or res.x is not None:
            xc = res.x
            if _is_valid(xc):
                Rv, _ = _objective_and_grad(xc)
                if Rv < best_R:
                    best_R = Rv
                    best_c = xc.copy()
        if time.time() > _DEADLINE - _TIME_MARGIN:
            break

    # Final polish: a few more L-BFGS-B iterations from the best point
    if time.time() < _DEADLINE - _TIME_MARGIN:
        try:
            res = minimize(
                _objective_and_grad,
                best_c,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-14,
                         'maxls': 80, 'maxfun': 5000}
            )
            if res.x is not None and _is_valid(res.x):
                Rv, _ = _objective_and_grad(res.x)
                if Rv < best_R:
                    best_R = Rv
                    best_c = res.x.copy()
        except Exception:
            pass

    # Ensure validity
    if not _is_valid(best_c):
        best_c = np.ones(21, dtype=np.float64)
    return best_c


# EVOLVE-BLOCK-END
