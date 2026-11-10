# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE_SECONDS = 615.0  # safety margin under the 630 s kill


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

    x, w = np.polynomial.legendre.leggauss(Ntheta)
    phi = np.linspace(0.0, 2.0 * np.pi, Nphi, endpoint=False)
    dphi = 2.0 * np.pi / Nphi

    H = np.zeros((Ntheta, Nphi, 2 * k + 1), dtype=np.float64)

    P = np.zeros((Ntheta, k + 1), dtype=np.float64)
    for m in range(k + 1):
        P[:, m] = eval_legendre(k, m, x)

    def N_const(m):
        m = abs(int(m))
        lo = k - m + 1
        hi = k + m
        den = 1.0
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


def _ratio_and_grad(c, H, weights):
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient w.r.t. c.
    f = H @ c  (shape Ntheta x Nphi)
    R = (sum f^4 * w)^(1/4) / (sum f^2 * w)^(1/2)

    Let A = sum f^4 w, B = sum f^2 w.
    R = A^(1/4) / B^(1/2)
    dR/dc = (1/4) A^(-3/4) * 4 f^3 (dw/d c) / B^(1/2) - A^(1/4) * (1/2) B^(-3/2) * 2 f (dw/d c)
         = A^(-3/4) B^(-1/2) f^3 w  -  A^(1/4) B^(-3/2) f w
         = (f w) [ A^(-3/4) f^2 B^(-1/2) - A^(1/4) B^(-3/2) ]
    where dw/d c for each component j is H_j (the j-th harmonic).

    So grad_j = sum_{i} f[i] * w[i] * H[i,j] * (A^(-3/4) f[i]^2 B^(-1/2) - A^(1/4) B^(-3/2))
    """
    f = np.tensordot(c, H, axes=([0], [2]))  # (Ntheta, Nphi)
    f2 = f * f
    f4 = f2 * f2
    A = np.sum(f4 * weights)  # ||f||_4^4
    B = np.sum(f2 * weights)  # ||f||_2^2

    if A <= 0 or B <= 0:
        return np.inf, np.zeros_like(c)

    R = A ** 0.25 / np.sqrt(B)

    # Gradient: grad_j = sum_i f_i w_i H_ij * [ A^(-3/4) f_i^2 B^(-1/2) - A^(1/4) B^(-3/2) ]
    A_neg34 = A ** (-0.75)
    B_neg12 = B ** (-0.5)
    A_pos14 = A ** 0.25
    B_neg32 = B ** (-1.5)

    coeff = A_neg34 * f2 * B_neg12 - A_pos14 * B_neg32  # shape (Ntheta, Nphi)
    # grad_j = sum_i (f * w * coeff)_i * H_ij
    fw_coeff = f * weights * coeff  # (Ntheta, Nphi)
    grad = np.tensordot(fw_coeff, H, axes=([0, 1], [0, 1]))  # (21,)

    return R, grad


def _power_iteration_restart(H, weights, n, rng, n_iter=50):
    """
    Spectral restart: find the eigenvector of the matrix M = H^T W H (Gram with weights),
    but more relevantly, we want to maximize the kurtosis-like functional.
    Instead, use a simple power iteration on the "f^2-weighted" operator:
    Start with random v, compute f = H@v, then update v = H^T (f^2 * w * H @ v) / ||...||
    This is a fixed-point for maximizing ||f||_4^4 / ||f||_2^2 approximately.
    """
    v = rng.normal(size=n)
    v /= (np.linalg.norm(v) + 1e-15)

    for _ in range(n_iter):
        f = np.tensordot(v, H, axes=([0], [2]))
        f2 = f * f
        # New direction: H^T (f^2 * w * f) = H^T (f^3 w)
        # Actually for maximizing A/B we want the gradient direction.
        # Let's just do: v_new = H^T (f^2 * w * H @ v)  -- this is the "natural" quadratic form
        # But to target quartic, use f^3:
        vec = f * f2 * weights  # f^3 * w
        v_new = np.tensordot(vec, H, axes=([0, 1], [0, 1]))
        norm = np.linalg.norm(v_new)
        if norm < 1e-15:
            break
        v_new /= norm
        v = v_new

    return v


def _make_initializations(k, rng, H, weights):
    """Generate diverse starting points."""
    n = 2 * k + 1
    inits = []

    # Single modes
    for m in range(-k, k + 1):
        v = np.zeros(n)
        v[m + k] = 1.0
        inits.append(v.copy())

    # Pairs
    for m in range(1, k + 1):
        v = np.zeros(n)
        v[m + k] = 1.0
        v[-m + k] = 1.0
        inits.append(v.copy())

    # All ones
    inits.append(np.ones(n))

    # Even m only
    v = np.zeros(n)
    for m in range(-k, k + 1, 2):
        v[m + k] = 1.0
    inits.append(v.copy())

    # Random
    for _ in range(30):
        v = rng.normal(size=n)
        v /= (np.linalg.norm(v) + 1e-15)
        inits.append(v)

    # Sparse
    for _ in range(15):
        v = np.zeros(n)
        idx = rng.choice(n, size=rng.integers(2, 6), replace=False)
        v[idx] = rng.normal(size=len(idx))
        v /= (np.linalg.norm(v) + 1e-15)
        inits.append(v)

    return inits


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m (m = -10..10) of a degree-10 real spherical
    harmonic f = sum_m c_m Y_m that MINIMIZES the norm ratio
        R = ||f||_{L^4(S^2)} / ||f||_{L^2(S^2)}
    Returns:
        coeffs: np.ndarray of shape (21,) with c_m ordered m = -10..10.
    """
    start_time = time.time()
    k = 10
    n = 2 * k + 1

    H, weights = _build_harmonic_matrix()
    rng = np.random.default_rng(42)

    def remaining():
        return _DEADLINE_SECONDS - (time.time() - start_time)

    # Precompute H as contiguous array for speed
    H_flat = H.reshape(-1, n)  # (Ntheta*Nphi, 21)
    w_flat = weights.reshape(-1)  # (Ntheta*Nphi,)

    def _ratio_grad_fast(c):
        f = H_flat @ c  # (N,)
        f2 = f * f
        f4 = f2 * f2
        A = np.dot(f4, w_flat)
        B = np.dot(f2, w_flat)
        if A <= 0 or B <= 0:
            return np.inf, np.zeros(n)
        R = A ** 0.25 / np.sqrt(B)
        A_neg34 = A ** (-0.75)
        B_neg12 = B ** (-0.5)
        A_pos14 = A ** 0.25
        B_neg32 = B ** (-1.5)
        coeff = A_neg34 * f2 * B_neg12 - A_pos14 * B_neg32
        fw_coeff = f * w_flat * coeff
        grad = H_flat.T @ fw_coeff
        return R, grad

    inits = _make_initializations(k, rng, H, weights)

    best_c = np.ones(n, dtype=np.float64)
    best_R = _ratio_grad_fast(best_c)[0]

    # Gradient descent with adaptive step + spectral restarts
    lr = 0.01
    n_steps_per_restart = 200
    n_restarts = 0

    for init_idx, c0 in enumerate(inits):
        if remaining() < 10.0:
            break

        c = c0.copy()
        # Normalize to unit L2 norm initially for stability
        c_norm = np.linalg.norm(c)
        if c_norm > 1e-15:
            c = c / c_norm

        # Gradient descent with backtracking
        for step in range(n_steps_per_restart):
            if remaining() < 5.0:
                break

            R, grad = _ratio_grad_fast(c)
            if not np.isfinite(R):
                break

            if R < best_R:
                best_R = R
                best_c = c.copy()

            # Check convergence
            if np.linalg.norm(grad) < 1e-12:
                break

            # Adaptive step size: try to reduce R
            g_norm = np.linalg.norm(grad)
            if g_norm < 1e-15:
                break

            # Backtracking line search
            step_size = lr
            improved = False
            for bt in range(20):
                c_trial = c - step_size * grad
                R_trial, _ = _ratio_grad_fast(c_trial)
                if np.isfinite(R_trial) and R_trial < R:
                    c = c_trial
                    improved = True
                    break
                step_size *= 0.5
            if not improved:
                break

        # Spectral restart: perturb best point and try power iteration
        n_restarts += 1
        if n_restarts % 5 == 0 and remaining() > 30.0:
            # Power iteration from best_c
            v = best_c + rng.normal(size=n) * 0.1
            v /= (np.linalg.norm(v) + 1e-15)
            for _ in range(30):
                if remaining() < 10.0:
                    break
                f = H_flat @ v
                f2 = f * f
                vec = f * f2 * w_flat
                v_new = H_flat.T @ vec
                nv = np.linalg.norm(v_new)
                if nv < 1e-15:
                    break
                v_new /= nv
                v = v_new

            # Gradient descent from power iteration result
            c = v.copy()
            for step in range(100):
                if remaining() < 5.0:
                    break
                R, grad = _ratio_grad_fast(c)
                if not np.isfinite(R):
                    break
                if R < best_R:
                    best_R = R
                    best_c = c.copy()
                g_norm = np.linalg.norm(grad)
                if g_norm < 1e-12:
                    break
                step_size = 0.01
                improved = False
                for bt in range(20):
                    c_trial = c - step_size * grad
                    R_trial, _ = _ratio_grad_fast(c_trial)
                    if np.isfinite(R_trial) and R_trial < R:
                        c = c_trial
                        improved = True
                        break
                    step_size *= 0.5
                if not improved:
                    break

    # Final polishing: more gradient steps from best
    if remaining() > 20.0:
        c = best_c.copy()
        for step in range(500):
            if remaining() < 10.0:
                break
            R, grad = _ratio_grad_fast(c)
            if not np.isfinite(R):
                break
            if R < best_R:
                best_R = R
                best_c = c.copy()
            g_norm = np.linalg.norm(grad)
            if g_norm < 1e-14:
                break
            step_size = 0.005
            improved = False
            for bt in range(25):
                c_trial = c - step_size * grad
                R_trial, _ = _ratio_grad_fast(c_trial)
                if np.isfinite(R_trial) and R_trial < R:
                    c = c_trial
                    improved = True
                    break
                step_size *= 0.5
            if not improved:
                break

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
