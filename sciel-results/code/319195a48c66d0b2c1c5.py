# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE_SECONDS = 620.0  # safety margin under the 630 s kill


def _build_harmonic_matrix():
    """
    Build the real orthonormal degree-10 spherical harmonic evaluation matrix.
    Returns H of shape (Ntheta, Nphi, 21) where column j corresponds to m = j-10.
    Uses 70-point Gauss-Legendre in cos(theta) and 130 equally spaced phi.
    """
    from scipy.special import eval_legendre

    k = 10
    Ntheta = 70
    Nphi = 130

    # Gauss-Legendre nodes and weights on [-1, 1]
    x, w = np.polynomial.legendre.leggauss(Ntheta)  # x in [-1,1], x = cos(theta)
    # theta = arccos(x); dOmega = dphi d(cos theta). Surface measure weight per (theta,phi) = w_i * (dphi)
    # We integrate over phi with uniform spacing: dphi = 2*pi/Nphi
    phi = np.linspace(0.0, 2.0 * np.pi, Nphi, endpoint=False)
    dphi = 2.0 * np.pi / Nphi

    # Condon-Shortley associated Legendre P_l^m(x) via scipy (includes (-1)^m factor)
    # For real harmonics:
    #   Y_0 = N_0 P_10^0
    #   Y_m (m>0) = sqrt(2) N_m P_10^m cos(m phi)
    #   Y_m (m<0) = sqrt(2) N_|m| P_10^{|m|} sin(|m| phi)
    # N_m = sqrt( (2l+1)/(4 pi) * (l-m)!/(l+m)! )
    H = np.zeros((Ntheta, Nphi, 2 * k + 1), dtype=np.float64)

    # Precompute P_10^m(x) for m = 0..10
    P = np.zeros((Ntheta, k + 1), dtype=np.float64)
    for m in range(k + 1):
        P[:, m] = eval_legendre(k, m, x)

    # Normalization constants
    def N_const(m):
        m = abs(int(m))
        # (10-m)!/(10+m)!
        num = 1.0
        den = 1.0
        # compute ratio of factorials
        # (10-m)!/(10+m)! = 1 / prod_{j=10-m+1}^{10+m} j
        lo = k - m + 1
        hi = k + m
        if hi >= lo:
            for j in range(lo, hi + 1):
                den *= j
        return np.sqrt((2 * k + 1) / (4.0 * np.pi) * (num / den))

    for m in range(-k, k + 1):
        j = m + k  # column index
        if m == 0:
            H[:, :, j] = N_const(0) * P[:, 0][:, None] * np.ones((1, Nphi))
        elif m > 0:
            H[:, :, j] = np.sqrt(2.0) * N_const(m) * P[:, m][:, None] * np.cos(m * phi)[None, :]
        else:
            am = -m
            H[:, :, j] = np.sqrt(2.0) * N_const(am) * P[:, am][:, None] * np.sin(am * phi)[None, :]

    # Quadrature weight per grid point: w_i (theta) * dphi (phi)
    # Total mass should be 4*pi. Check: sum_i w_i = 2 (Gauss-Legendre on [-1,1]), sum over phi of dphi = 2*pi.
    # So total = 2 * 2*pi = 4*pi. Good.
    weights = np.outer(w, np.full(Nphi, dphi))  # shape (Ntheta, Nphi)
    return H, weights


def _ratio_from_coeffs(c, H, weights):
    """Compute R = ||f||_4 / ||f||_2 for coefficients c (length 21)."""
    f = np.tensordot(c, H, axes=([0], [2]))  # shape (Ntheta, Nphi)
    # ||f||_2^2 = sum f^2 * weights
    f2 = f * f
    norm2_sq = np.sum(f2 * weights)
    # ||f||_4^4 = sum f^4 * weights
    f4 = f2 * f2
    norm4_4 = np.sum(f4 * weights)
    if norm2_sq <= 0.0 or norm4_4 <= 0.0:
        return np.inf
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    return norm4 / norm2


def _objective(c, H, weights):
    """Minimize R = ||f||_4/||f||_2. Return R (scalar)."""
    return _ratio_from_coeffs(c, H, weights)


def _make_initializations(k, rng):
    """Generate diverse starting points for the optimizer."""
    n = 2 * k + 1
    inits = []

    # 1) Single modes (each m alone)
    for m in range(-k, k + 1):
        v = np.zeros(n)
        v[m + k] = 1.0
        inits.append(v.copy())

    # 2) Pairs of +/- m (cos/sin combinations) to reduce directional concentration
    for m in range(1, k + 1):
        v = np.zeros(n)
        v[m + k] = 1.0
        v[-m + k] = 1.0
        inits.append(v.copy())
        v2 = np.zeros(n)
        v2[m + k] = 1.0
        v2[-m + k] = -1.0
        inits.append(v2.copy())

    # 3) Symmetric: only even m, equal magnitudes (reduces phi modulation)
    v = np.zeros(n)
    for m in range(-k, k + 1, 2):
        v[m + k] = 1.0
    inits.append(v.copy())

    # 4) Only m=0 and m=±10 (maximal + minimal azimuthal)
    v = np.zeros(n)
    v[0 + k] = 1.0
    v[10 + k] = 1.0
    v[-10 + k] = 1.0
    inits.append(v.copy())

    # 5) Random initializations
    for _ in range(40):
        v = rng.normal(size=n)
        v = v / (np.linalg.norm(v) + 1e-12)
        inits.append(v)

    # 6) Sparse random (few nonzero)
    for _ in range(20):
        v = np.zeros(n)
        idx = rng.choice(n, size=rng.integers(2, 6), replace=False)
        v[idx] = rng.normal(size=len(idx))
        v = v / (np.linalg.norm(v) + 1e-12)
        inits.append(v)

    return inits


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m (m = -10..10) of a degree-10 real spherical
    harmonic f = sum_m c_m Y_m that MINIMIZES the norm ratio
        R = ||f||_{L^4(S^2)} / ||f||_{L^2(S^2)}
    (surface measure of total mass 4*pi). Lower R = flatter = better.

    Returns:
        coeffs: np.ndarray of shape (21,) with c_m ordered m = -10..10.
    """
    start_time = time.time()
    k = 10
    n = 2 * k + 1

    H, weights = _build_harmonic_matrix()

    rng = np.random.default_rng(12345)
    inits = _make_initializations(k, rng)

    # Baseline: all ones
    best_c = np.ones(n, dtype=np.float64)
    best_R = _ratio_from_coeffs(best_c, H, weights)

    def remaining():
        return _DEADLINE_SECONDS - (time.time() - start_time)

    # Multi-start L-BFGS-B
    from scipy.optimize import minimize

    for i, c0 in enumerate(inits):
        if remaining() < 20.0:
            break
        try:
            res = minimize(
                _objective,
                c0,
                args=(H, weights),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10},
            )
            c_res = res.x
            R_res = _ratio_from_coeffs(c_res, H, weights)
            if np.isfinite(R_res) and R_res < best_R:
                best_R = R_res
                best_c = c_res.copy()
        except Exception:
            continue

    # Polishing: a few more L-BFGS-B restarts from the best point with tiny perturbations
    if remaining() > 30.0:
        for _ in range(5):
            if remaining() < 20.0:
                break
            pert = best_c + rng.normal(size=n) * 1e-3
            try:
                res = minimize(
                    _objective,
                    pert,
                    args=(H, weights),
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12},
                )
                c_res = res.x
                R_res = _ratio_from_coeffs(c_res, H, weights)
                if np.isfinite(R_res) and R_res < best_R:
                    best_R = R_res
                    best_c = c_res.copy()
            except Exception:
                continue

    # Validate: ensure finite and within magnitude bounds
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(n, dtype=np.float64)
    mx = np.max(np.abs(best_c))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)

    return best_c


# EVOLVE-BLOCK-END
