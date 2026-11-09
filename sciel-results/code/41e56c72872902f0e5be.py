# EVOLVE-BLOCK-START
import numpy as np
import time

# ---------------- Precompute the quadrature grid and basis matrix ----------
_K = 10
_NTH = 70
_NPHI = 130
_T0 = time.time()

def _build_grid_and_basis():
    """Return (Y, w) where Y is (NTH*NPHI, 21) with Y[i, j] = Y_m(x_i) for
    m = j-10, and w is the per-point surface weight (sum to 4*pi)."""
    # Gauss-Legendre nodes/weights on [-1,1]
    nodes, weights = np.polynomial.legendre.leggauss(_NTH)
    cos_th = nodes
    sin_th = np.sqrt(np.clip(1.0 - cos_th**2, 0.0, 1.0))
    phi = 2.0 * np.pi * np.arange(_NPHI) / _NPHI

    # Associated Legendre P_10^m with Condon-Shortley phase
    # scipy.special.lpmn returns (P, dP) with the Condon-Shortley phase included
    from scipy.special import lpmn
    P = np.zeros((_NPHI, _NTH, 2 * _K + 1), dtype=np.float64)
    for m in range(0, _K + 1):
        Pm, _ = lpmn(m, _K, cos_th)  # Pm has shape (len(cos_th),)
        P[:, :, m] = Pm[None, :]
        P[:, :, -m] = Pm[None, :]  # store |m| value; sign handled by sin/cos

    # Normalization N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
    # Use log-gamma for stability
    from scipy.special import gammaln
    logN = 0.5 * (np.log(21.0) - np.log(4.0 * np.pi)
                  + gammaln(_K + 1 - np.arange(_K + 1))
                  - gammaln(_K + 1 + np.arange(_K + 1)))
    N = np.exp(logN)  # N[m] for m = 0..K

    # Build Y_m(x) for m = -K..K
    Y = np.zeros((_NPHI * _NTH, 2 * _K + 1), dtype=np.float64)
    # m = 0
    Y[:, 0] = N[0] * P[:, :, 0]  # broadcast over phi
    # m > 0: sqrt(2) N_m P^m cos(m phi); m < 0: sqrt(2) N_|m| P^|m| sin(|m| phi)
    sqrt2 = np.sqrt(2.0)
    for m in range(1, _K + 1):
        base = sqrt2 * N[m] * P[:, :, m]  # (NPHI, NTH)
        Y[:, _K + m] = base * np.cos(m * phi)[:, None]
        Y[:, _K - m] = base * np.sin(m * phi)[:, None]

    # Surface weights: dOmega = sin_th dphi d(cos_th) ; quadrature weight per point
    w = np.zeros(_NPHI * _NTH, dtype=np.float64)
    dphi = 2.0 * np.pi / _NPHI
    for i in range(_NPHI):
        w[i * _NTH:(i + 1) * _NTH] = sin_th * weights * dphi
    # sum(w) should be 4*pi
    return Y, w


_Y, _W = _build_grid_and_basis()
_W2 = _W * _W  # not needed; keep _W
_NPTS = _Y.shape[0]


def _norm4_of(c):
    """Return ||f||_4 where f = Y @ c (quadrature approx, unnormalized measure)."""
    f = _Y @ c
    f2 = f * f
    return np.sqrt(np.sum(f2 * f2 * _W))  # (sum |f|^4 w)^(1/4)


def _norm2_of(c):
    f = _Y @ c
    return np.sqrt(np.sum(f * f * _W))


def _objective_and_grad(c):
    """Minimize ||f||_4 subject to ||f||_2 = 1 (we normalize c externally).
    Here we minimize J = (1/4) sum |f|^4 w  (monotone in ||f||_4, no 4th root
    -> smoother). Gradient: dJ/dc = sum (f^3 w) Y = Y^T (f^3 w)."""
    f = _Y @ c
    f3w = f * f * f * _W
    J = 0.25 * np.sum(f3w * f)
    g = _Y.T @ f3w
    return J, g


def _optimize_from(c0, deadline):
    """Run L-BFGS-B on the normalized problem: minimize J(c) s.t. ||c||_2=1
    handled by re-normalizing each step via a penalty-free reparam: we optimize
    on the sphere by projecting. Simpler: optimize J on the unit sphere using
    scipy minimize with a custom step that renormalizes. We instead minimize
    J(c) - lambda*(||c||_2^2 - 1) via augmented? Simpler and robust: minimize
    J over the sphere using scipy.optimize.minimize with the constraint
    ||c||_2^2 = 1 (SLSQP) is slow. Use projected gradient / L-BFGS with
    re-normalization: define a wrapper that renormalizes c after each gradient
    step is not directly supported. Instead: minimize J(c) with constraint
    using scipy L-BFGS-B on the *angle* parametrization is messy.

    Cleanest: minimize J(c) subject to ||c||_2^2 = 1 using scipy.optimize.minimize
    with method='SLSQP' and one equality constraint. SLSQP is slow but the
    problem is 21-dim; with a tight maxiter and deadline guard it's fine.
    """
    from scipy.optimize import minimize

    def obj(c):
        J, _ = _objective_and_grad(c)
        return J

    def obj_grad(c):
        J, g = _objective_and_grad(c)
        return J, g

    cons = ({'type': 'eq',
             'fun': lambda c: np.dot(c, c) - 1.0,
             'jac': lambda c: 2.0 * c})

    res = minimize(obj_grad, c0, method='SLSQP', jac=True, constraints=cons,
                   options={'maxiter': 400, 'ftol': 1e-12, 'disp': False})
    return res


def _ratio(c):
    n4 = _norm4_of(c)
    n2 = _norm2_of(c)
    if n2 <= 0:
        return np.inf
    return n4 / n2


def rudin_problem10() -> np.ndarray:
    deadline = _T0 + 560.0  # safety margin under 630s
    rng = np.random.default_rng(12345)

    # Seed candidates
    best_c = np.ones(2 * _K + 1, dtype=np.float64)
    best_c = best_c / np.linalg.norm(best_c)
    best_R = _ratio(best_c)

    # A few structured initial guesses
    candidates = [best_c.copy()]
    # pure m=0
    for m0 in range(-_K, _K + 1):
        v = np.zeros(2 * _K + 1)
        v[m0 + _K] = 1.0
        candidates.append(v)
    # random
    for _ in range(8):
        v = rng.standard_normal(2 * _K + 1)
        candidates.append(v)

    for c0 in candidates:
        if time.time() > deadline - 30.0:
            break
        c0 = c0 / np.linalg.norm(c0)
        try:
            res = _optimize_from(c0, deadline)
        except Exception:
            continue
        c = res.x
        nrm = np.linalg.norm(c)
        if nrm <= 0:
            continue
        c = c / nrm
        R = _ratio(c)
        if R < best_R:
            best_R = R
            best_c = c.copy()

    # Final safety: ensure valid
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(2 * _K + 1)
    nrm = np.linalg.norm(best_c)
    if nrm <= 0:
        best_c = np.ones(2 * _K + 1)
    else:
        best_c = best_c / nrm
    # max|c| sanity: scale to be within (1e-3, 1e5) -- normalization gives
    # max|c| ~ O(1), well within range.
    return best_c.astype(np.float64)


# EVOLVE-BLOCK-END
