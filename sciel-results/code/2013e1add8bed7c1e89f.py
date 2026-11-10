# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE_SECONDS = 600.0  # safety margin under the 630 s kill


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
    phi = np.linspace(0.0, 2.0 * np.pi, Nphi, endpoint=False)
    dphi = 2.0 * np.pi / Nphi

    H = np.zeros((Ntheta, Nphi, 2 * k + 1), dtype=np.float64)

    # Precompute P_10^m(x) for m = 0..10
    P = np.zeros((Ntheta, k + 1), dtype=np.float64)
    for m in range(k + 1):
        P[:, m] = eval_legendre(k, m, x)

    # Normalization constants
    def N_const(m):
        m = abs(int(m))
        lo = k - m + 1
        hi = k + m
        den = 1.0
        if hi >= lo:
            for j in range(lo, hi + 1):
                den *= j
        return np.sqrt((2 * k + 1) / (4.0 * np.pi) * (1.0 / den))

    for m in range(-k, k + 1):
        j = m + k
        if m == 0:
            H[:, :, j] = N_const(0) * P[:, 0][:, None] * np.ones((1, Nphi))
        elif m > 0:
            H[:, :, j] = np.sqrt(2.0) * N_const(m) * P[:, m][:, None] * np.cos(m * phi)[None, :]
        else:
            am = -m
            H[:, :, j] = np.sqrt(2.0) * N_const(am) * P[:, am][:, None] * np.sin(am * phi)[None, :]

    weights = np.outer(w, np.full(Nphi, dphi))
    return H, weights


def _compute_ratio(c, H, weights):
    """Compute R = ||f||_4 / ||f||_2 for coefficients c (length 21)."""
    f = np.tensordot(c, H, axes=([0], [2]))
    f2 = f * f
    norm2_sq = np.sum(f2 * weights)
    f4 = f2 * f2
    norm4_4 = np.sum(f4 * weights)
    if norm2_sq <= 0.0 or norm4_4 <= 0.0:
        return np.inf
    return norm4_4 ** 0.25 / np.sqrt(norm2_sq)


def _variance_surrogate(c, H, weights):
    """
    Smooth surrogate: minimize variance of f^2 (equivalent to minimizing E[f^4] - E[f^2]^2).
    This is smooth (quartic polynomial in c) and scale-invariant after normalization.
    We minimize E[f^4] subject to ||f||_2^2 = 1, which is equivalent to minimizing variance.
    """
    f = np.tensordot(c, H, axes=([0], [2]))
    f2 = f * f
    E_f4 = np.sum(f2 * f2 * weights) / (4.0 * np.pi)
    E_f2 = np.sum(f2 * weights) / (4.0 * np.pi)
    # Variance of f^2 = E[f^4] - (E[f^2])^2
    return E_f4 - E_f2 * E_f2


def _ratio_from_var(c, H, weights):
    """Compute actual R for scoring."""
    return _compute_ratio(c, H, weights)


def _make_initializations(k, rng):
    """Generate diverse starting points."""
    n = 2 * k + 1
    inits = []

    # 1) Single modes
    for m in range(-k, k + 1):
        v = np.zeros(n)
        v[m + k] = 1.0
        inits.append(v.copy())

    # 2) Pairs +/-m
    for m in range(1, k + 1):
        v = np.zeros(n)
        v[m + k] = 1.0
        v[-m + k] = 1.0
        inits.append(v.copy())
        v2 = np.zeros(n)
        v2[m + k] = 1.0
        v2[-m + k] = -1.0
        inits.append(v2.copy())

    # 3) Symmetric even m
    v = np.zeros(n)
    for m in range(-k, k + 1, 2):
        v[m + k] = 1.0
        v[m + k] /= np.sqrt(2 * k // 2 + 1)
    inits.append(v.copy())

    # 4) m=0 and m=±10
    v = np.zeros(n)
    v[0 + k] = 1.0
    v[10 + k] = 1.0
    v[-10 + k] = 1.0
    v /= np.linalg.norm(v)
    inits.append(v.copy())

    # 5) Only m=0
    v = np.zeros(n)
    v[k] = 1.0
    inits.append(v.copy())

    # 6) Only m=±10 (maximal azimuthal)
    v = np.zeros(n)
    v[-10 + k] = 1.0
    v[10 + k] = 1.0
    v /= np.linalg.norm(v)
    inits.append(v.copy())

    # 7) m=0 + m=±5
    v = np.zeros(n)
    v[k] = 1.0
    v[5 + k] = 1.0
    v[-5 + k] = 1.0
    v /= np.linalg.norm(v)
    inits.append(v.copy())

    # 8) m=±10 + m=±8
    v = np.zeros(n)
    v[-10 + k] = 1.0
    v[10 + k] = 1.0
    v[-8 + k] = 1.0
    v[8 + k] = 1.0
    v /= np.linalg.norm(v)
    inits.append(v.copy())

    # 9) Random full
    for _ in range(50):
        v = rng.normal(size=n)
        v /= (np.linalg.norm(v) + 1e-12)
        inits.append(v)

    # 10) Sparse random
    for _ in range(30):
        v = np.zeros(n)
        nnz = rng.integers(2, 7)
        idx = rng.choice(n, size=nnz, replace=False)
        v[idx] = rng.normal(size=nnz)
        v /= (np.linalg.norm(v) + 1e-12)
        inits.append(v)

    # 11) Concentrated high-m only
    for _ in range(20):
        v = np.zeros(n)
        idx = rng.choice([0, 1, 19, 20, 2, 18, 3, 17], size=rng.integers(2, 5), replace=False)
        v[idx] = rng.normal(size=len(idx))
        v /= (np.linalg.norm(v) + 1e-12)
        inits.append(v)

    return inits


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m (m = -10..10) of a degree-10 real spherical
    harmonic that MINIMIZES R = ||f||_{L^4} / ||f||_{L^2}.
    """
    start_time = time.time()
    k = 10
    n = 2 * k + 1

    H, weights = _build_harmonic_matrix()
    total_mass = 4.0 * np.pi

    rng = np.random.default_rng(42)
    inits = _make_initializations(k, rng)

    best_c = np.ones(n, dtype=np.float64)
    best_R = _compute_ratio(best_c, H, weights)

    def remaining():
        return _DEADLINE_SECONDS - (time.time() - start_time)

    from scipy.optimize import minimize

    def objective(c, H, weights):
        """Objective: minimize variance of f^2 = E[f^4] - E[f^2]^2."""
        f = np.tensordot(c, H, axes=([0], [2]))
        f2 = f * f
        E_f4 = np.sum(f2 * f2 * weights) / total_mass
        E_f2 = np.sum(f2 * weights) / total_mass
        return E_f4 - E_f2 * E_f2

    # Multi-start with L-BFGS-B on the variance surrogate
    for i, c0 in enumerate(inits):
        if remaining() < 15.0:
            break
        try:
            res = minimize(
                objective,
                c0,
                args=(H, weights),
                method='L-BFGS-B',
                options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-12},
            )
            c_res = res.x
            R_res = _compute_ratio(c_res, H, weights)
            if np.isfinite(R_res) and R_res < best_R:
                best_R = R_res
                best_c = c_res.copy()
        except Exception:
            continue

    # Phase 2: SLSQP with explicit unit-norm constraint for tighter convergence
    if remaining() > 40.0:
        def obj_sl(c):
            return objective(c, H, weights)

        def grad_sl(c):
            # Numerical gradient
            eps = 1e-8
            g = np.zeros_like(c)
            f0 = obj_sl(c)
            for j in range(n):
                cp = c.copy()
                cp[j] += eps
                g[j] = (obj_sl(cp) - f0) / eps
            return g

        cons = [{'type': 'eq', 'fun': lambda c: np.sum(c * c) - 1.0}]

        for _ in range(10):
            if remaining() < 20.0:
                break
            pert = best_c + rng.normal(size=n) * 0.01
            pert /= (np.linalg.norm(pert) + 1e-12)
            try:
                res = minimize(
                    obj_sl,
                    pert,
                    jac=grad_sl,
                    method='SLSQP',
                    constraints=cons,
                    options={'maxiter': 500, 'ftol': 1e-14},
                )
                c_res = res.x
                c_res = c_res / (np.linalg.norm(c_res) + 1e-12)
                R_res = _compute_ratio(c_res, H, weights)
                if np.isfinite(R_res) and R_res < best_R:
                    best_R = R_res
                    best_c = c_res.copy()
            except Exception:
                continue

    # Phase 3: Nelder-Mead on normalized sphere for final polish
    if remaining() > 30.0:
        from scipy.optimize import minimize

        def obj_nm(c):
            c_norm = c / (np.linalg.norm(c) + 1e-12)
            return objective(c_norm, H, weights)

        for _ in range(5):
            if remaining() < 15.0:
                break
            c0 = best_c.copy()
            try:
                res = minimize(
                    obj_nm,
                    c0,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12},
                )
                c_res = res.x
                c_res = c_res / (np.linalg.norm(c_res) + 1e-12)
                R_res = _compute_ratio(c_res, H, weights)
                if np.isfinite(R_res) and R_res < best_R:
                    best_R = R_res
                    best_c = c_res.copy()
            except Exception:
                continue

    # Final polishing with very tight L-BFGS-B
    if remaining() > 10.0:
        try:
            res = minimize(
                objective,
                best_c,
                args=(H, weights),
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14},
            )
            c_res = res.x
            R_res = _compute_ratio(c_res, H, weights)
            if np.isfinite(R_res) and R_res < best_R:
                best_R = R_res
                best_c = c_res.copy()
        except Exception:
            pass

    # Validate
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(n, dtype=np.float64)
    mx = np.max(np.abs(best_c))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)

    return best_c


# EVOLVE-BLOCK-END
