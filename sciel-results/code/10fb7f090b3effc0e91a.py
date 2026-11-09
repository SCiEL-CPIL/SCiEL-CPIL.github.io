# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize

_K = 10
_NPHI = 130
_NTH = 70

def _build_quadrature():
    """Build Gauss-Legendre nodes/weights in x=cos(theta) and phi grid,
    plus the real orthonormal degree-K harmonics evaluated on the grid."""
    x, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI
    W = np.outer(w, np.full(_NPHI, dphi))
    
    Y = np.zeros((_NTH, _NPHI, 2 * _K + 1))
    for idx, m in enumerate(range(-_K, _K + 1)):
        am = abs(m)
        denom = 1.0
        for n in range(_K - am + 1, _K + am + 1):
            denom *= n
        fact_ratio = 1.0 / denom
        Nm = np.sqrt((2 * _K + 1) / (4.0 * np.pi) * fact_ratio)
        P = lpmv(am, _K, x)
        if m == 0:
            Y[:, :, idx] = Nm * P[:, None]
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.sin(am * phi)[None, :]
    return Y, W

_Y, _W = _build_quadrature()
_Wflat = _W.ravel()
_Yflat = _Y.reshape(-1, 2 * _K + 1)


def _build_quadrature():
    """Build Gauss-Legendre nodes/weights in x=cos(theta) and phi grid,
    plus the real orthonormal degree-K harmonics evaluated on the grid."""
    x, w = np.polynomial.legendre.leggauss(_NTH)          # x in (-1,1), weights sum to 2
    # surface element dOmega = dphi dx (since dOmega = sin(theta) dtheta dphi = -dx dphi)
    # total mass = 4*pi, and sum_i w_i = 2, sum_j (dphi) = 2*pi  -> product = 4*pi. Good.
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI

    # 2D weights: W[i,j] = w_i * dphi
    W = np.outer(w, np.full(_NPHI, dphi))

    # Evaluate real orthonormal harmonics Y_m for m = -K..K on the grid.
    # Y_0 = N_0 P^K(cos theta)
    # Y_m = sqrt(2) N_m P^|m|(cos theta) cos(m phi)  for m>0
    # Y_m = sqrt(2) N_m P^|m|(cos theta) sin(|m| phi) for m<0
    # N_m = sqrt( (2K+1)/(4 pi) * (K-m)!/(K+m)! )
    Y = np.zeros((_NTH, _NPHI, 2 * _K + 1))
    for idx, m in enumerate(range(-_K, _K + 1)):
        am = abs(m)
        fact_ratio = 1.0
        for n in range(am):
            fact_ratio *= (1.0 / (n + 1.0))
        # (K-am)!/(K+am)! = product over n=am..K of 1/(n+1)  ... careful
        # (K-am)!/(K+am)! = 1 / [ (K-am+1)(K-am+2)...(K+am) ]
        denom = 1.0
        for n in range(_K - am + 1, _K + am + 1):
            denom *= n
        fact_ratio = 1.0 / denom
        Nm = np.sqrt((2 * _K + 1) / (4.0 * np.pi) * fact_ratio)
        P = lpmv(am, _K, x)  # includes Condon-Shortley phase
        if m == 0:
            Y[:, :, idx] = Nm * P[:, None]
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.sin(am * phi)[None, :]
    return Y, W


_Y, _W = _build_quadrature()
_W2 = _W * _W  # not needed; keep W
_Wflat = _W.ravel()
_Yflat = _Y.reshape(-1, 2 * _K + 1)  # (N, 21)


def _ratio_from_coeffs(c):
    """Compute R = ||f||4/||f||2 for coefficients c (length 21)."""
    c = np.asarray(c, dtype=np.float64)
    f = _Yflat @ c
    f2 = f * f
    f4 = f2 * f2
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 0.0:
        return np.inf
    l4sq = float(np.dot(_Wflat, f4))
    if l4sq <= 0.0:
        return 1.0
    l2 = np.sqrt(l2sq)
    l4 = np.sqrt(l4sq)
    return l4 / l2


def _objective(c):
    c = np.asarray(c, dtype=np.float64)
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return 1e30
    c = c / nrm
    return _ratio_from_coeffs(c)

def _negE(c):
    """Negative L4^4 / (4pi)^2 for L2-normalized c, to maximize flatness."""
    c = np.asarray(c, dtype=np.float64)
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return -1e30
    c = c / nrm
    f = _Yflat @ c
    f2 = f * f
    l4sq = float(np.dot(_Wflat, f2 * f2))
    if l4sq <= 0.0:
        return -1.0
    # E = ||f||_4^4 / (4pi)^2 = l4sq / (4pi)^2
    return -l4sq / ((4.0 * np.pi) ** 2)


def _gradient(c):
    """Analytic gradient of R w.r.t. c."""
    c = np.asarray(c, dtype=np.float64)
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return np.zeros_like(c)
    cn = c / nrm
    f = _Yflat @ cn
    f2 = f * f
    l2sq = float(np.dot(_Wflat, f2))
    l4sq = float(np.dot(_Wflat, f2 * f2))
    l2 = np.sqrt(l2sq)
    l4 = np.sqrt(l4sq)
    Yt_f = _Yflat.T @ (f * _Wflat)
    Yt_f3 = _Yflat.T @ (f * f2 * _Wflat)
    dl2_dc = (2.0 / l2) * Yt_f
    dl4_dc = (2.0 / l4) * Yt_f3
    dR_dc = (dl4_dc * l2 - dl2_dc * l4) / (l2 * l2)
    dR_dc = dR_dc - (np.dot(dR_dc, cn) / nrm) * c
    return dR_dc

def _negE_grad(c):
    """Gradient of -E where E = ||f||_4^4/(4pi)^2 for L2-normalized c."""
    c = np.asarray(c, dtype=np.float64)
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return np.zeros_like(c)
    cn = c / nrm
    f = _Yflat @ cn
    f2 = f * f
    l4sq = float(np.dot(_Wflat, f2 * f2))
    # d/dc of -l4sq/(4pi)^2 = -(1/(4pi)^2) * 2 * sum W f^3 Y * 4 (since d(f^4)/df = 4f^3)
    # Actually: E = sum W f^4 / (4pi)^2, dE/dc = (4/(4pi)^2) sum W f^3 Y
    Yt_f3 = _Yflat.T @ (f * f2 * _Wflat)
    dE_dc = (4.0 / ((4.0 * np.pi) ** 2)) * Yt_f3
    # Project to tangent
    dE_dc = dE_dc - (np.dot(dE_dc, cn) / nrm) * c
    return -dE_dc


def _optimize_from(c0, deadline):
    c0 = np.asarray(c0, dtype=np.float64)
    nrm = np.linalg.norm(c0)
    if nrm < 1e-14:
        c0 = np.random.randn(2 * _K + 1)
        nrm = np.linalg.norm(c0)
    c0 = c0 / nrm
    
    # Phase 1: Optimize R directly
    res = minimize(_objective, c0, jac=_gradient, method="L-BFGS-B",
                   options={"maxiter": 300, "maxfun": 600, "gtol": 1e-10,
                            "ftol": 1e-14})
    c = np.asarray(res.x, dtype=np.float64)
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return None
    c = c / nrm
    val1 = float(_ratio_from_coeffs(c))
    
    # Phase 2: Refine using -E objective (maximize flatness)
    if time.time() < deadline - 10.0:
        res2 = minimize(_negE, c, jac=_negE_grad, method="L-BFGS-B",
                       options={"maxiter": 200, "maxfun": 400, "gtol": 1e-10,
                                "ftol": 1e-14})
        c2 = np.asarray(res2.x, dtype=np.float64)
        nrm2 = np.linalg.norm(c2)
        if nrm2 < 1e-14:
            return c, val1
        c2 = c2 / nrm2
        val2 = float(_ratio_from_coeffs(c2))
        if val2 < val1:
            return c2, val2
    return c, val1


def rudin_problem10() -> np.ndarray:
    start = time.time()
    deadline = start + 580.0  # safety margin before 630s kill
    rng = np.random.default_rng(2024)

    best_c = None
    best_val = np.inf

    def _candidate_structured(i):
        """Structured initialization encouraging flatness."""
        c = np.zeros(2 * _K + 1)
        # Use symmetric patterns with varying phases
        for m in range(-_K, _K + 1):
            idx = m + _K
            # Mix of cosine and sine patterns with different frequencies
            c[idx] = np.cos(i * 0.5 + m * 0.4) * np.cos(m * 0.2 + i * 0.1)
            c[idx] += 0.2 * rng.standard_normal()
        return c

    def _candidate_random():
        """Pure random initialization."""
        c = rng.standard_normal(2 * _K + 1)
        return c

    def _candidate_symmetric():
        """Symmetric initialization (even/odd m patterns)."""
        c = np.zeros(2 * _K + 1)
        for m in range(-_K, _K + 1):
            idx = m + _K
            if m == 0:
                c[idx] = 1.0
            else:
                # Symmetric: c[-m] = c[m] or c[-m] = -c[m]
                base = rng.standard_normal()
                c[idx] = base if rng.random() > 0.5 else -base
        # Ensure symmetry
        for m in range(1, _K + 1):
            if rng.random() > 0.5:
                c[_K + m] = c[_K - m]
            else:
                c[_K + m] = -c[_K - m]
        return c

    # Phase 1: Multiple restarts with different strategies
    n_restarts = 35
    for i in range(n_restarts):
        if time.time() > deadline - 60.0:
            break
        # Alternate between different initialization strategies
        if i % 3 == 0:
            c0 = _candidate_structured(i)
        elif i % 3 == 1:
            c0 = _candidate_random()
        else:
            c0 = _candidate_symmetric()
        
        try:
            out = _optimize_from(c0, deadline)
        except Exception:
            continue
        if out is None:
            continue
        c, val = out
        if val < best_val:
            best_val = val
            best_c = c.copy()
    
    # Phase 2: Fine-tuning from best result
    if best_c is not None and time.time() < deadline - 30.0:
        # Try small perturbations around best
        for trial in range(5):
            if time.time() > deadline - 15.0:
                break
            perturbation = 0.01 * rng.standard_normal(2 * _K + 1)
            c0 = best_c + perturbation
            try:
                out = _optimize_from(c0, deadline)
            except Exception:
                continue
            if out is None:
                continue
            c, val = out
            if val < best_val:
                best_val = val
                best_c = c.copy()

    if best_c is None:
        best_c = np.ones(2 * _K + 1, dtype=np.float64)

    best_c = np.asarray(best_c, dtype=np.float64)
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(2 * _K + 1, dtype=np.float64)
    mx = float(np.max(np.abs(best_c)))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)
    return best_c


# EVOLVE-BLOCK-END
