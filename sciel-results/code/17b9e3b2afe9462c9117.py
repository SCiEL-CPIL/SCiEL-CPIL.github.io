# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import eval_legendre, gammaln
from scipy.optimize import minimize


def _precompute():
    """Precompute quadrature nodes, weights, and spherical harmonic basis."""
    k = 10

    # Gauss-Legendre quadrature for cos(theta) in [-1, 1]
    n_theta = 70
    x, w = np.polynomial.legendre.leggauss(n_theta)

    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi

    # Precompute P_10^m(cos(theta)) for m = 0..10 at all quadrature nodes
    P = np.zeros((11, n_theta))
    for m in range(11):
        P[m, :] = eval_legendre(10, m, x)

    # Normalization constants
    N = np.zeros(11)
    for m in range(11):
        log_N = 0.5 * (np.log(21.0 / (4 * np.pi)) + gammaln(11 - m) - gammaln(11 + m))
        N[m] = np.exp(log_N)

    # Build basis matrix: B[m_idx, i, j] where m_idx=0..20 corresponds to m=-10..10
    B = np.zeros((21, n_theta, n_phi))

    for m_idx in range(21):
        m = m_idx - 10
        if m == 0:
            B[m_idx, :, :] = N[0] * P[0, :]
        elif m > 0:
            B[m_idx, :, :] = np.sqrt(2) * N[m] * P[m, :][:, np.newaxis] * np.cos(m * phi)[np.newaxis, :]
        else:
            am = abs(m)
            B[m_idx, :, :] = np.sqrt(2) * N[am] * P[am, :][:, np.newaxis] * np.sin(am * phi)[np.newaxis, :]

    # Quadrature weights
    W = np.outer(w, np.ones(n_phi)) * dphi

    return B, W, x, phi, n_theta, n_phi


def _evaluate_ratio(coeffs, B, W):
    """Compute ||f||_4 / ||f||_2 using precomputed basis and weights."""
    f = np.tensordot(coeffs, B, axes=(0, 0))
    f2 = f * f
    norm2_sq = np.sum(W * f2)
    f4 = f2 * f2
    norm4_4 = np.sum(W * f4)

    if norm2_sq <= 0:
        return np.inf

    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))

    return norm4 / norm2


def _objective_ratio(c, B, W):
    """
    Objective: minimize R = ||f||_4 / ||f||_2 directly.
    Since R is scale-invariant, we work on the unit sphere ||c||=1.
    On the unit sphere, ||f||_2 = 1 (by orthonormality), so R = ||f||_4.
    But to be safe, we compute the ratio directly.
    """
    c_norm = np.linalg.norm(c)
    if c_norm < 1e-15:
        return 1e10, np.zeros_like(c)
    c_unit = c / c_norm
    f = np.tensordot(c_unit, B, axes=(0, 0))
    f2 = f * f
    norm2_sq = np.sum(W * f2)
    f4 = f2 * f2
    norm4_4 = np.sum(W * f4)
    
    if norm2_sq <= 0:
        return 1e10, np.zeros_like(c)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    R = norm4 / norm2
    
    # Gradient of R w.r.t. c on the sphere
    # R = ||f||_4 / ||f||_2
    # dR/dc_m = (1/||f||_2) * d||f||_4/dc_m - (||f||_4/||f||_2^2) * d||f||_2/dc_m
    # d||f||_4/dc_m = (1/4) * ||f||_4^{-3} * 4 * sum W f^3 Y_m = ||f||_4^{-3} * sum W f^3 Y_m
    # d||f||_2/dc_m = (1/2) * ||f||_2^{-1} * 2 * sum W f Y_m = ||f||_2^{-1} * sum W f Y_m
    # So dR/dc_m = (1/||f||_2) * ||f||_4^{-3} * sum W f^3 Y_m - (||f||_4/||f||_2^2) * ||f||_2^{-1} * sum W f Y_m
    #            = ||f||_4^{-3} * sum W f^3 Y_m / ||f||_2 - ||f||_4 * sum W f Y_m / ||f||_2^3
    #            = (1/||f||_2^3) * (||f||_2^2 * sum W f^3 Y_m / ||f||_4^3 - ||f||_4 * sum W f Y_m)
    
    # Simpler: use the fact that on the sphere, we can minimize ||f||_4^4
    # and the gradient is 4 * sum W f^3 Y_m
    # But let's compute the actual ratio gradient for better convergence
    
    grad_f4 = 4.0 * np.tensordot(W * f**3, B, axes=([0, 1], [1, 2]))
    grad_f2 = 2.0 * np.tensordot(W * f, B, axes=([0, 1], [1, 2]))
    
    # R = N4 / N2 where N4 = ||f||_4, N2 = ||f||_2
    # dR = (N2 * dN4 - N4 * dN2) / N2^2
    # dN4/dc = (1/(4*N4^3)) * grad_f4  [since N4^4 = sum W f^4, dN4^4 = 4 sum W f^3 df, so dN4 = (1/(4*N4^3)) * 4 sum W f^3 df = (1/N4^3) sum W f^3 df]
    # Actually: N4 = (sum W f^4)^(1/4), dN4 = (1/4)(sum W f^4)^(-3/4) * 4 sum W f^3 df = N4^{-3} sum W f^3 df
    # N2 = (sum W f^2)^(1/2), dN2 = (1/2)(sum W f^2)^(-1/2) * 2 sum W f df = N2^{-1} sum W f df
    
    dN4_dc = grad_f4 / (4.0 * norm4**3) if norm4 > 0 else np.zeros_like(c)
    dN2_dc = grad_f2 / (2.0 * norm2) if norm2 > 0 else np.zeros_like(c)
    
    grad_R = (norm2 * dN4_dc - norm4 * dN2_dc) / (norm2**2)
    
    # Project to tangent space of sphere
    grad_R_proj = grad_R - np.dot(grad_R, c_unit) * c_unit
    
    return R, grad_R_proj


def _objective_f4(c, B, W):
    """Minimize ||f||_4^4 on the unit sphere (simpler, equivalent when ||c||=1)."""
    c_norm = np.linalg.norm(c)
    if c_norm < 1e-15:
        return 1e10, np.zeros_like(c)
    c_unit = c / c_norm
    f = np.tensordot(c_unit, B, axes=(0, 0))
    f4 = f**4
    obj = np.sum(W * f4)
    grad = 4.0 * np.tensordot(W * f**3, B, axes=([0, 1], [1, 2]))
    # Project to tangent space
    grad_proj = grad - np.dot(grad, c_unit) * c_unit
    return obj, grad_proj


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m for m = -10..10 of a real degree-10 spherical harmonic
    that minimize the norm ratio ||f||_4 / ||f||_2 on the unit sphere.
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin

    B, W, x, phi, n_theta, n_phi = _precompute()

    dim = 21

    def check_time():
        return time.time() < deadline - 5

    best_coeffs = None
    best_ratio = np.inf

    def make_objective(c):
        """Return objective value for L-BFGS-B (uses normalized c)."""
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-15:
            return 1e10
        c_unit = c / c_norm
        f = np.tensordot(c_unit, B, axes=(0, 0))
        f4 = f**4
        return np.sum(W * f4)

    def make_grad(c):
        """Return gradient for L-BFGS-B."""
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-15:
            return np.zeros(dim)
        c_unit = c / c_norm
        f = np.tensordot(c_unit, B, axes=(0, 0))
        grad = 4.0 * np.tensordot(W * f**3, B, axes=([0, 1], [1, 2]))
        # Project to tangent space of sphere
        grad = grad - np.dot(grad, c_unit) * c_unit
        return grad

    # Generate diverse initializations
    np.random.seed(12345)
    initializations = []

    # 1. All ones
    initializations.append(np.ones(dim) / np.sqrt(dim))

    # 2. Only m=0
    c0 = np.zeros(dim)
    c0[10] = 1.0
    initializations.append(c0.copy())

    # 3. Random unit vectors
    for trial in range(40):
        c = np.random.randn(dim)
        c /= np.linalg.norm(c)
        initializations.append(c)

    # 4. Alternating signs
    c_alt = np.ones(dim)
    c_alt[::2] = -1.0
    initializations.append(c_alt / np.linalg.norm(c_alt))

    # 5. Reduced m=0
    c_low_m0 = np.ones(dim)
    c_low_m0[10] = 0.1
    initializations.append(c_low_m0 / np.linalg.norm(c_low_m0))

    # 6. Pure high-m modes
    for m_val in [3, 5, 7, 8, 10]:
        for sign in [1, -1]:
            c = np.zeros(dim)
            idx_pos = m_val + 10
            idx_neg = -m_val + 10
            c[idx_pos] = sign
            c[idx_neg] = sign
            initializations.append(c / np.linalg.norm(c))

    # 7. Mix of modes with random weights, downweight m=0
    for trial in range(30):
        c = np.random.randn(dim)
        c[10] *= 0.3  # downweight m=0
        c /= np.linalg.norm(c)
        initializations.append(c)

    # 8. Structured: emphasize even m or odd m
    c_even = np.zeros(dim)
    for i in range(dim):
        m = i - 10
        if m % 2 == 0:
            c_even[i] = 1.0
    initializations.append(c_even / np.linalg.norm(c_even))
    
    c_odd = np.zeros(dim)
    for i in range(dim):
        m = i - 10
        if m % 2 != 0:
            c_odd[i] = 1.0
    initializations.append(c_odd / np.linalg.norm(c_odd))

    # 9. Symmetric pairs with random phases
    for trial in range(15):
        c = np.zeros(dim)
        for m_val in range(1, 11):
            idx_pos = m_val + 10
            idx_neg = -m_val + 10
            weight = np.random.randn()
            c[idx_pos] = weight
            c[idx_neg] = weight * np.random.choice([1, -1])
        c[10] = np.random.randn() * 0.5
        c /= np.linalg.norm(c)
        initializations.append(c)

    # Run L-BFGS-B from each initialization
    for i, c_init in enumerate(initializations):
        if not check_time():
            break

        try:
            result = minimize(
                make_objective,
                c_init,
                jac=make_grad,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )

            c_opt = result.x
            c_norm = np.linalg.norm(c_opt)
            if c_norm < 1e-15:
                continue
            c_opt = c_opt / c_norm

            ratio = _evaluate_ratio(c_opt, B, W)

            if ratio < best_ratio:
                best_ratio = ratio
                best_coeffs = c_opt.copy()

        except Exception:
            continue

    # Additional refinement with more iterations on the best solution
    if best_coeffs is not None and check_time():
        try:
            result = minimize(
                make_objective,
                best_coeffs,
                jac=make_grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
            )
            c_opt = result.x
            c_norm = np.linalg.norm(c_opt)
            if c_norm >= 1e-15:
                c_opt = c_opt / c_norm
                ratio = _evaluate_ratio(c_opt, B, W)
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_coeffs = c_opt.copy()
        except Exception:
            pass

    # Try even more random restarts if time permits
    if check_time():
        for trial in range(80):
            if not check_time():
                break
            c = np.random.randn(dim)
            c /= np.linalg.norm(c)
            try:
                result = minimize(
                    make_objective,
                    c,
                    jac=make_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
                c_opt = result.x
                c_norm = np.linalg.norm(c_opt)
                if c_norm >= 1e-15:
                    c_opt = c_opt / c_norm
                    ratio = _evaluate_ratio(c_opt, B, W)
                    if ratio < best_ratio:
                        best_ratio = ratio
                        best_coeffs = c_opt.copy()
            except Exception:
                continue

    # Final polishing pass with very tight tolerance
    if best_coeffs is not None and check_time():
        try:
            result = minimize(
                make_objective,
                best_coeffs,
                jac=make_grad,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12, 'disp': False}
            )
            c_opt = result.x
            c_norm = np.linalg.norm(c_opt)
            if c_norm >= 1e-15:
                c_opt = c_opt / c_norm
                ratio = _evaluate_ratio(c_opt, B, W)
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_coeffs = c_opt.copy()
        except Exception:
            pass

    if best_coeffs is None:
        best_coeffs = np.ones(dim) / np.sqrt(dim)

    # Ensure validity: max|c| in [1e-3, 1e5]
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = best_coeffs * (1e-3 / max_abs)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs * (1e5 / max_abs)

    # Remove any NaN or inf
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(dim)

    return best_coeffs.astype(np.float64)


# EVOLVE-BLOCK-END
