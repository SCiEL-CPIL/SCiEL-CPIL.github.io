# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv, gammaln
import time


def _build_quadrature():
    """Build the 70-point Gauss-Legendre (cos theta) x 130-point phi grid."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    weights = np.outer(w_gl, np.full(n_phi, dphi))
    cos_theta = x_gl
    return cos_theta, phi, weights


def _spherical_harmonics_real(cos_theta, phi, k=10):
    """
    Compute real orthonormal spherical harmonics of degree k.
    Returns array of shape (2*k+1, n_theta, n_phi) where index i corresponds to m = i - k.
    """
    n_theta = len(cos_theta)
    n_phi = len(phi)
    Y = np.zeros((2 * k + 1, n_theta, n_phi))
    
    for m_abs in range(k + 1):
        log_N = 0.5 * (np.log(2 * k + 1) - np.log(4 * np.pi) + 
                       gammaln(k - m_abs + 1) - gammaln(k + m_abs + 1))
        N_m = np.exp(log_N)
        P_m = lpmv(m_abs, k, cos_theta)
        
        if m_abs == 0:
            Y[0] = N_m * P_m[:, np.newaxis] * np.ones((1, n_phi))
        else:
            Y[k + m_abs] = np.sqrt(2) * N_m * P_m[:, np.newaxis] * np.cos(m_abs * phi)[np.newaxis, :]
            Y[k - m_abs] = np.sqrt(2) * N_m * P_m[:, np.newaxis] * np.sin(m_abs * phi)[np.newaxis, :]
    
    return Y


def _compute_ratio(coeffs, Y, weights):
    """Compute R = ||f||_4 / ||f||_2 using precomputed harmonics and weights."""
    f = np.tensordot(coeffs, Y, axes=([0], [0]))
    
    f2 = f * f
    f4 = f2 * f2
    
    norm2_sq = np.sum(weights * f2)
    norm4_sq = np.sum(weights * f4)
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_sq))
    
    return norm4 / norm2


def _compute_ratio_and_grad(coeffs, Y, weights):
    """Compute R and its gradient with respect to coeffs (vectorized)."""
    n_modes = Y.shape[0]
    
    f = np.tensordot(coeffs, Y, axes=([0], [0]))
    f2 = f * f
    f4 = f2 * f2
    f3 = f * f2
    
    w_f2 = weights * f2
    w_f4 = weights * f4
    w_f3 = weights * f3
    w_f = weights * f
    
    norm2_sq = np.sum(w_f2)
    norm4_sq = np.sum(w_f4)
    
    if norm2_sq <= 0:
        return np.inf, np.zeros(n_modes)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_sq))
    
    R = norm4 / norm2
    
    if norm4_sq <= 0:
        return R, np.zeros(n_modes)
    
    # Vectorized gradient computation
    d_norm4_sq_dc = 4.0 * np.einsum('ij,mij->m', w_f3, Y)
    d_norm2_sq_dc = 2.0 * np.einsum('ij,mij->m', w_f, Y)
    
    d_norm4_dc = (1.0 / (4.0 * norm4**3)) * d_norm4_sq_dc
    d_norm2_dc = (1.0 / (2.0 * norm2)) * d_norm2_sq_dc
    
    grad = d_norm4_dc / norm2 - norm4 * d_norm2_dc / (norm2 * norm2)
    
    return R, grad


def _random_unit_vector(rng, n=21):
    """Generate a random unit vector."""
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    return v


def _try_symmetric_patterns(Y, weights):
    """Try various symmetric coefficient patterns."""
    best_coeffs = None
    best_ratio = np.inf
    
    patterns = []
    
    # All ones normalized
    c = np.ones(21) / np.sqrt(21)
    patterns.append(c)
    
    # Only m=0
    c = np.zeros(21)
    c[10] = 1.0
    patterns.append(c)
    
    # m=0 and m=±10
    c = np.zeros(21)
    c[10] = 1.0
    c[0] = 1.0
    c[20] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Even m modes: 0, 2, 4, 6, 8, 10
    c = np.zeros(21)
    for m in [0, 2, 4, 6, 8, 10]:
        c[10 + m] = 1.0
        if m > 0:
            c[10 - m] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Odd m modes: 1, 3, 5, 7, 9
    c = np.zeros(21)
    for m in [1, 3, 5, 7, 9]:
        c[10 + m] = 1.0
        c[10 - m] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Cosine-weighted: c_m = cos(m * pi/20) for m=0..10
    c = np.zeros(21)
    for m in range(11):
        c[10 + m] = np.cos(m * np.pi / 20.0)
        if m > 0:
            c[10 - m] = np.cos(m * np.pi / 20.0)
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Sine-weighted
    c = np.zeros(21)
    for m in range(1, 11):
        c[10 + m] = np.sin(m * np.pi / 21.0)
        c[10 - m] = np.sin(m * np.pi / 21.0)
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # High-m concentration
    c = np.zeros(21)
    c[0] = 1.0
    c[20] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Low-m concentration
    c = np.zeros(21)
    c[10] = 1.0
    c[9] = 1.0
    c[11] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Alternating signs
    c = np.zeros(21)
    for m in range(21):
        c[m] = 1.0 if m % 2 == 0 else -1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Gaussian-like distribution
    c = np.zeros(21)
    for m in range(21):
        idx = m - 10
        c[m] = np.exp(-idx**2 / 20.0)
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Weighted by m
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        c = np.zeros(21)
        for m in range(11):
            c[10 + m] = m**alpha if m > 0 else 1.0
            if m > 0:
                c[10 - m] = m**alpha
        c /= np.linalg.norm(c)
        patterns.append(c)
    
    # Inverse weighted
    for alpha in [0.5, 1.0, 1.5]:
        c = np.zeros(21)
        c[10] = 1.0
        for m in range(1, 11):
            c[10 + m] = 1.0 / (m**alpha)
            c[10 - m] = 1.0 / (m**alpha)
        c /= np.linalg.norm(c)
        patterns.append(c)
    
    for c in patterns:
        r = _compute_ratio(c, Y, weights)
        if r < best_ratio:
            best_ratio = r
            best_coeffs = c.copy()
    
    return best_coeffs, best_ratio


def _simulated_annealing(c_init, Y, weights, n_steps=5000, t_start=0.5, t_end=0.001, rng=None):
    """Simulated annealing to find better coefficients."""
    if rng is None:
        rng = np.random.default_rng()
    
    c = c_init.copy()
    c /= np.linalg.norm(c)
    current_ratio = _compute_ratio(c, Y, weights)
    best_c = c.copy()
    best_ratio = current_ratio
    
    for step in range(n_steps):
        t = t_start * (t_end / t_start) ** (step / n_steps)
        
        # Generate random perturbation
        delta = rng.standard_normal(21) * 0.1
        c_new = c + delta
        c_new /= np.linalg.norm(c_new)
        
        new_ratio = _compute_ratio(c_new, Y, weights)
        
        if new_ratio < current_ratio or rng.random() < np.exp((current_ratio - new_ratio) / t):
            c = c_new
            current_ratio = new_ratio
            if new_ratio < best_ratio:
                best_ratio = new_ratio
                best_c = c.copy()
    
    return best_c, best_ratio


def _polish_local(c_init, Y, weights, deadline, safety_margin):
    """High-precision polish using L-BFGS-B with tighter tolerances."""
    def objective_with_grad(c):
        if time.time() > deadline - safety_margin:
            return _compute_ratio(c_init, Y, weights), np.zeros(21)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            return 1e10, np.zeros(21)
        c_normalized = c / c_norm
        R, grad = _compute_ratio_and_grad(c_normalized, Y, weights)
        # Project gradient to constraint surface
        grad_proj = grad - np.dot(grad, c_normalized) * c_normalized
        return R, grad_proj
    
    result = minimize(
        objective_with_grad,
        c_init,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 300}
    )
    
    if result.fun < _compute_ratio(c_init, Y, weights):
        c_opt = result.x / np.linalg.norm(result.x)
        return c_opt, result.fun
    return c_init, _compute_ratio(c_init, Y, weights)


def _build_fast_matrices(Y, weights):
    """
    Build flattened matrices for fast ratio computation.
    Y_flat: shape (n_modes, N) where N = n_theta * n_phi
    w_flat: shape (N,)
    """
    n_modes = Y.shape[0]
    Y_flat = Y.reshape(n_modes, -1)
    w_flat = weights.ravel()
    return Y_flat, w_flat


def _compute_ratio_fast(coeffs, Y_flat, w_flat):
    """Fast ratio computation using precomputed flattened arrays."""
    f = Y_flat @ coeffs
    f2 = f * f
    f4 = f2 * f2
    
    norm2_sq = np.dot(w_flat, f2)
    norm4_sq = np.dot(w_flat, f4)
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_sq))
    
    return norm4 / norm2


def _compute_ratio_and_grad_fast(coeffs, Y_flat, w_flat):
    """Fast ratio and gradient using precomputed flattened arrays."""
    n_modes = Y_flat.shape[0]
    
    f = Y_flat @ coeffs
    f2 = f * f
    f4 = f2 * f2
    f3 = f * f2
    
    w_f2 = w_flat * f2
    w_f4 = w_flat * f4
    w_f3 = w_flat * f3
    w_f = w_flat * f
    
    norm2_sq = np.sum(w_f2)
    norm4_sq = np.sum(w_f4)
    
    if norm2_sq <= 0:
        return np.inf, np.zeros(n_modes)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_sq))
    
    R = norm4 / norm2
    
    if norm4_sq <= 0:
        return R, np.zeros(n_modes)
    
    # Gradient: d/dc [norm4/norm2]
    # d_norm4_sq/dc = 4 * sum(w * f^3 * Y_m) for each m
    # d_norm2_sq/dc = 2 * sum(w * f * Y_m) for each m
    d_norm4_sq_dc = 4.0 * (Y_flat.T @ (w_f3))
    d_norm2_sq_dc = 2.0 * (Y_flat.T @ (w_f))
    
    d_norm4_dc = (1.0 / (4.0 * norm4**3)) * d_norm4_sq_dc
    d_norm2_dc = (1.0 / (2.0 * norm2)) * d_norm2_sq_dc
    
    grad = d_norm4_dc / norm2 - norm4 * d_norm2_dc / (norm2 * norm2)
    
    return R, grad


def _generate_structured_starts(n_structured):
    """Generate a diverse set of structured initial guesses."""
    starts = []
    
    # All ones
    c = np.ones(21) / np.sqrt(21)
    starts.append(c)
    
    # Only m=0
    c = np.zeros(21)
    c[10] = 1.0
    starts.append(c)
    
    # Only m=±10
    c = np.zeros(21)
    c[0] = 1.0
    c[20] = 1.0
    c /= np.linalg.norm(c)
    starts.append(c)
    
    # m=0 and m=±10
    c = np.zeros(21)
    c[10] = 1.0
    c[0] = 1.0
    c[20] = 1.0
    c /= np.linalg.norm(c)
    starts.append(c)
    
    # Even m modes
    c = np.zeros(21)
    for m in [0, 2, 4, 6, 8, 10]:
        c[10 + m] = 1.0
        if m > 0:
            c[10 - m] = 1.0
    c /= np.linalg.norm(c)
    starts.append(c)
    
    # Odd m modes
    c = np.zeros(21)
    for m in [1, 3, 5, 7, 9]:
        c[10 + m] = 1.0
        c[10 - m] = 1.0
    c /= np.linalg.norm(c)
    starts.append(c)
    
    # Cosine-weighted
    c = np.zeros(21)
    for m in range(11):
        c[10 + m] = np.cos(m * np.pi / 20.0)
        if m > 0:
            c[10 - m] = np.cos(m * np.pi / 20.0)
    c /= np.linalg.norm(c)
    starts.append(c)
    
    # Sine-weighted
    c = np.zeros(21)
    for m in range(1, 11):
        c[10 + m] = np.sin(m * np.pi / 21.0)
        c[10 - m] = np.sin(m * np.pi / 21.0)
    c /= np.linalg.norm(c)
    starts.append(c)
    
    # Low-m concentration
    c = np.zeros(21)
    c[9] = 0.5
    c[10] = 1.0
    c[11] = 0.5
    c /= np.linalg.norm(c)
    starts.append(c)
    
    # Bimodal
    c = np.zeros(21)
    c[10] = 1.0
    c[0] = 0.8
    c[20] = 0.8
    c[1] = 0.3
    c[19] = 0.3
    c /= np.linalg.norm(c)
    starts.append(c)
    
    # Gaussian centered at m=0
    c = np.zeros(21)
    for m in range(21):
        idx = m - 10
        c[m] = np.exp(-idx**2 / 15.0)
    c /= np.linalg.norm(c)
    starts.append(c)
    
    # Gaussian centered at m=5
    c = np.zeros(21)
    for m in range(21):
        idx = m - 10
        c[m] = np.exp(-(idx - 5)**2 / 10.0)
    c /= np.linalg.norm(c)
    starts.append(c)
    
    # Gaussian centered at m=-5
    c = np.zeros(21)
    for m in range(21):
        idx = m - 10
        c[m] = np.exp(-(idx + 5)**2 / 10.0)
    c /= np.linalg.norm(c)
    starts.append(c)
    
    # Power law: c_m ∝ |m|^(-alpha)
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        c = np.zeros(21)
        c[10] = 1.0
        for m in range(1, 11):
            c[10 + m] = m**(-alpha)
            c[10 - m] = m**(-alpha)
        c /= np.linalg.norm(c)
        starts.append(c)
    
    # Power law: c_m ∝ |m|^alpha
    for alpha in [0.5, 1.0, 1.5]:
        c = np.zeros(21)
        c[10] = 1.0
        for m in range(1, 11):
            c[10 + m] = m**alpha
            c[10 - m] = m**alpha
        c /= np.linalg.norm(c)
        starts.append(c)
    
    # Alternating signs
    c = np.zeros(21)
    for m in range(21):
        c[m] = 1.0 if m % 2 == 0 else -1.0
    c /= np.linalg.norm(c)
    starts.append(c)
    
    # Linear ramp
    c = np.zeros(21)
    for m in range(21):
        c[m] = (m - 10) / 10.0
    c /= np.linalg.norm(c)
    starts.append(c)
    
    # Quadratic
    c = np.zeros(21)
    for m in range(21):
        idx = m - 10
        c[m] = idx**2
    c /= np.linalg.norm(c)
    starts.append(c)
    
    # Pure high-m (m=±9, ±10)
    c = np.zeros(21)
    c[0] = 1.0
    c[20] = 1.0
    c[1] = 0.8
    c[19] = 0.8
    c /= np.linalg.norm(c)
    starts.append(c)
    
    # Pure low-m (m=0, ±1, ±2)
    c = np.zeros(21)
    c[10] = 1.0
    c[9] = 0.9
    c[11] = 0.9
    c[8] = 0.8
    c[12] = 0.8
    c /= np.linalg.norm(c)
    starts.append(c)
    
    # Symmetric with sign flips
    for sign_pattern in [1, -1]:
        c = np.zeros(21)
        for m in range(11):
            val = np.cos(m * np.pi / 20.0)
            if sign_pattern == -1 and m % 2 == 1:
                val = -val
            c[10 + m] = val
            if m > 0:
                c[10 - m] = val * sign_pattern
        c /= np.linalg.norm(c)
        starts.append(c)
    
    # Truncated: only |m| <= 3
    c = np.zeros(21)
    for m in range(-3, 4):
        c[10 + m] = 1.0
    c /= np.linalg.norm(c)
    starts.append(c)
    
    # Truncated: only |m| >= 7
    c = np.zeros(21)
    for m in list(range(-10, -6)) + list(range(7, 11)):
        c[10 + m] = 1.0
    c /= np.linalg.norm(c)
    starts.append(c)
    
    # Random but reproducible structured patterns
    rng_struct = np.random.default_rng(12345)
    for _ in range(min(10, max(0, n_structured - len(starts)))):
        # Random pattern with some structure
        c = np.zeros(21)
        # Pick a random subset of modes
        active_modes = rng_struct.choice(21, size=rng_struct.integers(5, 15), replace=False)
        for idx in active_modes:
            c[idx] = rng_struct.standard_normal()
        if np.linalg.norm(c) > 1e-10:
            c /= np.linalg.norm(c)
            starts.append(c)
    
    return starts[:n_structured]


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients minimizing R = ||f||_4 / ||f||_2 for degree-10 spherical harmonic.
    Returns 21 real coefficients ordered m = -10..10.
    """
    deadline = time.time() + 600.0
    safety_margin = 90.0
    
    # Build quadrature grid
    cos_theta, phi, weights = _build_quadrature()
    
    # Precompute spherical harmonics
    Y = _spherical_harmonics_real(cos_theta, phi, k=10)
    
    # Build fast computation matrices
    Y_flat, w_flat = _build_fast_matrices(Y, weights)
    
    # Try symmetric patterns first
    best_coeffs, best_ratio = _try_symmetric_patterns(Y, weights)
    
    if best_coeffs is None:
        best_coeffs = np.ones(21, dtype=np.float64) / np.sqrt(21)
        best_ratio = _compute_ratio(best_coeffs, Y, weights)
    
    # Set up optimization
    rng = np.random.default_rng(42)
    
    def objective_with_grad(c):
        if time.time() > deadline - safety_margin:
            return _compute_ratio_fast(best_coeffs, Y_flat, w_flat), np.zeros(21)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            return 1e10, np.zeros(21)
        c_normalized = c / c_norm
        R, grad = _compute_ratio_and_grad_fast(c_normalized, Y_flat, w_flat)
        # Project gradient to constraint surface
        grad_proj = grad - np.dot(grad, c_normalized) * c_normalized
        return R, grad_proj
    
    # Generate diverse structured starts
    structured_starts = _generate_structured_starts(30)
    
    # First pass: try all structured starts
    for c_init in structured_starts:
        if time.time() > deadline - safety_margin:
            break
        
        result = minimize(
            objective_with_grad,
            c_init,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 200}
        )
        
        if result.fun < best_ratio:
            c_opt = result.x / np.linalg.norm(result.x)
            r_opt = _compute_ratio_fast(c_opt, Y_flat, w_flat)
            if r_opt < best_ratio:
                best_ratio = r_opt
                best_coeffs = c_opt
    
    # Second pass: random restarts
    n_random_restarts = 60
    for i in range(n_random_restarts):
        if time.time() > deadline - safety_margin:
            break
        
        c_init = _random_unit_vector(rng)
        
        result = minimize(
            objective_with_grad,
            c_init,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 600, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 200}
        )
        
        if result.fun < best_ratio:
            c_opt = result.x / np.linalg.norm(result.x)
            r_opt = _compute_ratio_fast(c_opt, Y_flat, w_flat)
            if r_opt < best_ratio:
                best_ratio = r_opt
                best_coeffs = c_opt
    
    # Simulated annealing from best found
    if time.time() < deadline - safety_margin:
        c_sa, r_sa = _simulated_annealing(
            best_coeffs, Y, weights, 
            n_steps=4000, t_start=0.3, t_end=0.0005, rng=rng
        )
        if r_sa < best_ratio:
            best_ratio = r_sa
            best_coeffs = c_sa
    
    # Additional refinement: coordinate descent style
    if time.time() < deadline - safety_margin:
        c = best_coeffs.copy()
        for _ in range(200):
            if time.time() > deadline - safety_margin:
                break
            improved = False
            for m in range(21):
                if time.time() > deadline - safety_margin:
                    break
                for delta in [-0.5, -0.25, -0.1, -0.05, 0.05, 0.1, 0.25, 0.5]:
                    c_new = c.copy()
                    c_new[m] += delta * 0.03
                    c_new /= np.linalg.norm(c_new)
                    r_new = _compute_ratio_fast(c_new, Y_flat, w_flat)
                    if r_new < best_ratio - 1e-13:
                        c = c_new
                        best_ratio = r_new
                        improved = True
                        break
            if not improved:
                break
    
    # Third pass: local perturbations of best solution
    if time.time() < deadline - safety_margin:
        for i in range(40):
            if time.time() > deadline - safety_margin:
                break
            
            c_init = best_coeffs + 0.15 * rng.standard_normal(21)
            c_init /= np.linalg.norm(c_init)
            
            result = minimize(
                objective_with_grad,
                c_init,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 200}
            )
            
            if result.fun < best_ratio:
                c_opt = result.x / np.linalg.norm(result.x)
                r_opt = _compute_ratio_fast(c_opt, Y_flat, w_flat)
                if r_opt < best_ratio:
                    best_ratio = r_opt
                    best_coeffs = c_opt
    
    # Final simulated annealing
    if time.time() < deadline - safety_margin:
        c_sa, r_sa = _simulated_annealing(
            best_coeffs, Y, weights,
            n_steps=2500, t_start=0.2, t_end=0.0003, rng=rng
        )
        if r_sa < best_ratio:
            best_ratio = r_sa
            best_coeffs = c_sa
    
    # High-precision polish
    if time.time() < deadline - safety_margin:
        c_polished, r_polished = _polish_local(best_coeffs, Y, weights, deadline, safety_margin)
        if r_polished < best_ratio:
            best_ratio = r_polished
            best_coeffs = c_polished
    
    # Ensure output is valid
    best_coeffs = best_coeffs.astype(np.float64)
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        best_coeffs = np.ones(21, dtype=np.float64)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        target = 1.0
        best_coeffs = best_coeffs * (target / max_abs)
    
    return best_coeffs


# EVOLVE-BLOCK-END
