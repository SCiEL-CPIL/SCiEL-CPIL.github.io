# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _assoc_legendre_P(n, m, x):
    """Associated Legendre polynomial P_n^m(x) with Condon-Shortley phase."""
    m = abs(m)
    if m > n:
        return np.zeros_like(x)
    # Compute P_m^m first
    Pmm = np.ones_like(x)
    if m > 0:
        somedef = np.sqrt((1 - x) * (1 + x))
        for i in range(1, m + 1):
            Pmm = -Pmm * somedef * (2 * i - 1)
            somedef *= (1 - x * x)
    if n == m:
        return Pmm
    Pmmp1 = x * (2 * m + 1) * Pmm
    if n == m + 1:
        return Pmmp1
    Pmm2 = np.zeros_like(x)
    for i in range(m + 2, n + 1):
        Pmm2 = ((2 * i - 1) * x * Pmmp1 - (i + m - 1) * Pmm) / (i - m)
        Pmm = Pmmp1
        Pmmp1 = Pmm2
    return Pmmp1


def _build_quadrature(n_theta=70, n_phi=130):
    """Build quadrature grid on the sphere."""
    # Gauss-Legendre nodes and weights in [-1, 1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_theta)
    # Map to theta: cos(theta) = x_gl, so dOmega = sin(theta) dtheta dphi
    # Actually, the integral over sphere: ∫ f dΩ = ∫_0^π ∫_0^{2π} f(θ,φ) sinθ dφ dθ
    # With u = cosθ, du = -sinθ dθ, so ∫_0^π g(θ) sinθ dθ = ∫_{-1}^{1} g(arccos(u)) du
    # So we integrate over u = cosθ with GL weights, and over φ uniformly.
    theta = np.arccos(x_gl)
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    # Surface element: dΩ = dφ du (since sinθ dθ dφ = du dφ with u=cosθ)
    # Total area should be 4π: sum(w_gl) * (2π/n_phi * n_phi) = 2 * 2π = 4π ✓
    w_theta = w_gl  # weights for u = cosθ
    w_phi = 2.0 * np.pi / n_phi  # weight for each φ point
    # Full weight for each (θ_i, φ_j) point
    W = np.outer(w_theta, w_phi)  # shape (n_theta, n_phi)
    return x_gl, theta, phi, W


def _compute_Y_values(x_gl, theta, phi, k=10):
    """Compute all real orthonormal spherical harmonics Y_m for m=-k..k at grid points.
    Returns array of shape (2k+1, n_theta, n_phi).
    """
    n_theta = len(theta)
    n_phi = len(phi)
    Y = np.zeros((2 * k + 1, n_theta, n_phi))
    cos_theta = np.cos(theta)
    for m in range(-k, k + 1):
        am = abs(m)
        # Normalization N_m
        N_m = np.sqrt((2 * k + 1) / (4 * np.pi) * (np.math.factorial(k - am) / np.math.factorial(k + am)))
        P_am = _assoc_legendre_P(k, am, cos_theta)
        if m == 0:
            Y[m + k, :, :] = N_m * P_am[:, None] * np.ones((1, n_phi))
        elif m > 0:
            Y[m + k, :, :] = np.sqrt(2) * N_m * P_am[:, None] * np.cos(m * phi[None, :])
        else:
            Y[m + k, :, :] = np.sqrt(2) * N_m * P_am[:, None] * np.sin(am * phi[None, :])
    return Y


def _compute_norms(f, W):
    """Compute ||f||_2^2 and ||f||_4^4 from values f on grid with weights W."""
    f2 = f * f
    f4 = f2 * f2
    norm2_sq = np.sum(f2 * W)
    norm4_4 = np.sum(f4 * W)
    return norm2_sq, norm4_4


def _objective_and_grad(coeffs, Y, W):
    """Objective: minimize ||f||_4^4 / ||f||_2^4 (scale invariant).
    Equivalent to minimizing ||f||_4^4 subject to ||f||_2^2 = 1.
    We use the ratio to handle scale invariance naturally.
    """
    f = Y[0] * coeffs[0]
    for i in range(1, 21):
        f = f + Y[i] * coeffs[i]
    f2 = f * f
    f4 = f2 * f2
    norm2_sq = np.sum(f2 * W)
    norm4_4 = np.sum(f4 * W)
    # R^4 = norm4_4 / norm2_sq^2
    R4 = norm4_4 / (norm2_sq * norm2_sq)
    # Gradient of R4 w.r.t. coeffs
    # R4 = A / B^2 where A = sum(f^4 W), B = sum(f^2 W)
    # dR4/dc_i = (4 * sum(f^3 * Y_i * W) * B^2 - A * 2 * B * 2 * sum(f * Y_i * W)) / B^4
    #         = 4 * sum(f^3 * Y_i * W) / B^2 - 4 * A * sum(f * Y_i * W) / B^3
    #         = 4/B^2 * sum(f^3 * Y_i * W) - 4*A/B^3 * sum(f * Y_i * W)
    f3 = f2 * f
    grad = np.zeros(21)
    for i in range(21):
        s1 = np.sum(f3 * Y[i] * W)
        s2 = np.sum(f * Y[i] * W)
        grad[i] = 4.0 * s1 / (norm2_sq * norm2_sq) - 4.0 * norm4_4 * s2 / (norm2_sq ** 3)
    return R4, grad


def _hessian(coeffs, Y, W):
    """Compute Hessian of R4 = ||f||_4^4 / ||f||_2^4."""
    f = Y[0] * coeffs[0]
    for i in range(1, 21):
        f = f + Y[i] * coeffs[i]
    f2 = f * f
    f4 = f2 * f2
    norm2_sq = np.sum(f2 * W)
    norm4_4 = np.sum(f4 * W)
    
    # Hessian entries:
    # R4 = A/B^2, A = sum(f^4 W), B = sum(f^2 W)
    # dR4/dc_i = 4/B^2 * sum(f^3 Y_i W) - 4A/B^3 * sum(f Y_i W)
    # d2R4/dc_i dc_j = 4/B^2 * sum(3f^2 Y_i Y_j W) - 4A/B^3 * sum(Y_i Y_j W)
    #   + 8/B^3 * sum(f^3 Y_i W) * sum(f Y_j W) - 8A/B^4 * (sum(f^2 Y_i W)*sum(f^2 Y_j W) + sum(f Y_i W)*sum(f Y_j W))
    # Wait, let me be more careful.
    # Let g_i = sum(f^3 Y_i W), h_i = sum(f Y_i W), q_ij = sum(f^2 Y_i Y_j W), r_ij = sum(Y_i Y_j W)
    # dR4/dc_i = 4 g_i / B^2 - 4 A h_i / B^3
    # d2R4/dc_i dc_j = 4/B^2 * d(g_i)/dc_j - 4A/B^3 * d(h_i)/dc_j + 8 g_i / B^3 * dB/dc_j - 4A/B^3 * h_i * dB/dc_j + 8A h_i / B^4 * dB/dc_j
    # where dB/dc_j = 2 h_j
    # d(g_i)/dc_j = 3 sum(f^2 Y_i Y_j W) = 3 q_ij
    # d(h_i)/dc_j = r_ij
    # So: d2R4/dc_i dc_j = 12 q_ij / B^2 - 4A r_ij / B^3 + 16 g_i h_j / B^3 - 8A h_i h_j / B^3 + 16 A h_i h_j / B^4
    # Simplify: = 12 q_ij / B^2 - 4A r_ij / B^3 + 16 g_i h_j / B^3 + 8A h_i h_j (2/B^4 - 1/B^3)
    # Hmm, let me just compute numerically or use a simpler approach.
    # Actually for Newton polish, let's just use finite differences or skip Hessian and use more L-BFGS iterations.
    
    # Let's compute it properly:
    # R4 = A * B^{-2}
    # ∂R4/∂c_i = A' B^{-2} + A * (-2) B^{-3} B'
    # where A' = 4 ∑ f^3 Y_i W, B' = 2 ∑ f Y_i W
    # ∂²R4/∂c_i∂c_j = A'' B^{-2} + A' * (-2) B^{-3} B'_j + A * [2 B^{-4} (B'_i)(B'_j) + (-2) B^{-3} B''_ij]
    # where A'' = 12 ∑ f^2 Y_i Y_j W, B''_ij = 2 ∑ Y_i Y_j W
    # So: ∂²R4/∂c_i∂c_j = 12 q_ij / B^2 - 8 A' h_j / B^3 - 8 A h_i h_j / B^3 + 2 A * 2 h_i h_j / B^4 + A * (-2) * 2 r_ij / B^3
    # = 12 q_ij / B^2 - 8 g_i h_j / B^3 - 8 A h_i h_j / B^3 + 4 A h_i h_j / B^4 - 4 A r_ij / B^3
    
    H = np.zeros((21, 21))
    for i in range(21):
        g_i = np.sum(f3 * Y[i] * W)
        h_i = np.sum(f * Y[i] * W)
        for j in range(i, 21):
            q_ij = np.sum(f2 * Y[i] * Y[j] * W)
            r_ij = np.sum(Y[i] * Y[j] * W)
            g_j = np.sum(f3 * Y[j] * W)
            h_j = np.sum(f * Y[j] * W)
            val = (12.0 * q_ij / (norm2_sq ** 2)
                   - 8.0 * g_i * h_j / (norm2_sq ** 3)
                   - 8.0 * norm4_4 * h_i * h_j / (norm2_sq ** 3)
                   + 4.0 * norm4_4 * h_i * h_j / (norm2_sq ** 4)
                   - 4.0 * norm4_4 * r_ij / (norm2_sq ** 3))
            H[i, j] = val
            H[j, i] = val
    return H


def rudin_problem10() -> np.ndarray:
    t_start = time.time()
    deadline = t_start + 600  # 600s deadline with margin
    
    k = 10
    n_theta, n_phi = 70, 130
    
    # Build quadrature
    x_gl, theta, phi, W = _build_quadrature(n_theta, n_phi)
    
    # Precompute Y values
    Y = _compute_Y_values(x_gl, theta, phi, k)
    
    # Baseline: all ones
    best_coeffs = np.ones(21, dtype=np.float64)
    best_R4, _ = _objective_and_grad(best_coeffs, Y, W)
    best_R = best_R4 ** 0.25  # R = (R4)^{1/4}
    
    # Try a few structured initializations
    inits = []
    # All ones
    inits.append(np.ones(21))
    # Only m=0
    c = np.zeros(21); c[10] = 1.0; inits.append(c)
    # Only even m
    c = np.zeros(21); c[0::2] = 1.0; inits.append(c)
    # Only odd m
    c = np.zeros(21); c[1::2] = 1.0; inits.append(c)
    # Random
    rng = np.random.default_rng(42)
    for _ in range(20):
        c = rng.standard_normal(21)
        inits.append(c)
    # Some structured patterns
    c = np.zeros(21)
    for m in range(-10, 11):
        c[m+10] = np.cos(np.pi * m / 11)
    inits.append(c)
    c = np.zeros(21)
    for m in range(-10, 11):
        c[m+10] = np.sin(np.pi * m / 11)
    inits.append(c)
    
    for init in inits:
        if time.time() > deadline - 60:
            break
        try:
            # L-BFGS-B optimization
            result = minimize(
                lambda c: _objective_and_grad(c, Y, W)[0],
                init,
                jac=lambda c: _objective_and_grad(c, Y, W)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12}
            )
            if result.success and result.fun < best_R4:
                best_R4 = result.fun
                best_coeffs = result.x.copy()
                best_R = best_R4 ** 0.25
        except Exception:
            pass
    
    # Newton polish on best solution
    if time.time() < deadline - 30:
        try:
            c = best_coeffs.copy()
            for it in range(50):
                if time.time() > deadline - 30:
                    break
                R4, grad = _objective_and_grad(c, Y, W)
                if np.max(np.abs(grad)) < 1e-12:
                    break
                H = _hessian(c, Y, W)
                # Solve H * delta = -grad
                try:
                    delta = np.linalg.solve(H + 1e-10 * np.eye(21), -grad)
                except np.linalg.LinAlgError:
                    delta = -0.01 * grad
                # Line search
                alpha = 1.0
                for _ in range(30):
                    c_new = c + alpha * delta
                    R4_new, _ = _objective_and_grad(c_new, Y, W)
                    if R4_new < R4:
                        c = c_new
                        break
                    alpha *= 0.5
                if R4_new < best_R4:
                    best_R4 = R4_new
                    best_coeffs = c.copy()
                    best_R = best_R4 ** 0.25
        except Exception:
            pass
    
    # Validate
    if not np.all(np.isfinite(best_coeffs)) or len(best_coeffs) != 21:
        best_coeffs = np.ones(21)
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = best_coeffs / max_abs if max_abs > 0 else np.ones(21)
    if max_abs > 1e5:
        best_coeffs = best_coeffs / max_abs
    
    return best_coeffs


# EVOLVE-BLOCK-END
