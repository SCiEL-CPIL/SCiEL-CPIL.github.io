# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv, factorial
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
        N_m = np.sqrt((2 * k + 1) / (4 * np.pi) * 
                      float(factorial(k - m_abs) / factorial(k + m_abs)))
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
    
    # Cosine-weighted
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
    
    # More patterns: weighted by m
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        c = np.zeros(21)
        for m in range(11):
            c[10 + m] = m**alpha
            if m > 0:
                c[10 - m] = m**alpha
        c /= np.linalg.norm(c)
        patterns.append(c)
    
    # Inverse weighted
    for alpha in [0.5, 1.0, 1.5]:
        c = np.zeros(21)
        for m in range(1, 11):
            c[10 + m] = 1.0 / (m**alpha)
            c[10 - m] = 1.0 / (m**alpha)
        c /= np.linalg.norm(c)
        patterns.append(c)
    
    # Additional patterns for broader coverage
    # Concentrate on m=±5
    c = np.zeros(21)
    c[5] = 1.0
    c[15] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Concentrate on m=±7
    c = np.zeros(21)
    c[3] = 1.0
    c[17] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Concentrate on m=±9
    c = np.zeros(21)
    c[1] = 1.0
    c[19] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Mixed: m=0, ±5, ±10
    c = np.zeros(21)
    c[10] = 1.0
    c[5] = 1.0
    c[15] = 1.0
    c[0] = 1.0
    c[20] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Mixed: m=0, ±3, ±6, ±9
    c = np.zeros(21)
    c[10] = 1.0
    for m in [3, 6, 9]:
        c[10 + m] = 1.0
        c[10 - m] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Mixed: m=0, ±4, ±8
    c = np.zeros(21)
    c[10] = 1.0
    for m in [4, 8]:
        c[10 + m] = 1.0
        c[10 - m] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Quadratic weighting
    c = np.zeros(21)
    for m in range(21):
        idx = m - 10
        c[m] = 1.0 - 0.05 * idx**2
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Cubic weighting
    c = np.zeros(21)
    for m in range(21):
        idx = m - 10
        c[m] = 1.0 - 0.02 * abs(idx)**3
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    for c in patterns:
        r = _compute_ratio(c, Y, weights)
        if r < best_ratio:
            best_ratio = r
            best_coeffs = c.copy()
    
    return best_coeffs, best_ratio


def _simulated_annealing(c_init, Y, weights, n_steps=8000, t_start=0.5, t_end=0.001, rng=None):
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
    """High-precision polish using tighter tolerances and more iterations."""
    c = c_init.copy()
    c /= np.linalg.norm(c)
    
    def objective(c):
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            return 1e10, np.zeros(21)
        c_normed = c / c_norm
        R, grad = _compute_ratio_and_grad(c_normed, Y, weights)
        grad_proj = grad - np.dot(grad, c_normed) * c_normed
        return R, grad_proj
    
    result = minimize(
        objective,
        c,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 500}
    )
    
    if result.fun < _compute_ratio(c, Y, weights):
        c_opt = result.x / np.linalg.norm(result.x)
        return c_opt
    
    return c


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
    
    # Try symmetric patterns first
    best_coeffs, best_ratio = _try_symmetric_patterns(Y, weights)
    
    if best_coeffs is None:
        best_coeffs = np.ones(21, dtype=np.float64) / np.sqrt(21)
        best_ratio = _compute_ratio(best_coeffs, Y, weights)
    
    rng = np.random.default_rng(42)
    
    def objective_with_grad(c):
        if time.time() > deadline - safety_margin:
            return _compute_ratio(best_coeffs, Y, weights), np.zeros(21)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            return 1e10, np.zeros(21)
        c_normalized = c / c_norm
        R, grad = _compute_ratio_and_grad(c_normalized, Y, weights)
        grad_proj = grad - np.dot(grad, c_normalized) * c_normalized
        return R, grad_proj
    
    # Phase 1: Broad search with many restarts
    n_restarts = 100
    
    for i in range(n_restarts):
        if time.time() > deadline - safety_margin:
            break
        
        if i == 0:
            c_init = best_coeffs.copy()
        elif i < 25:
            # Structured starts
            if i == 1:
                c_init = np.zeros(21)
                c_init[10] = 1.0
            elif i == 2:
                c_init = np.zeros(21)
                c_init[20] = 1.0
            elif i == 3:
                c_init = np.zeros(21)
                c_init[0] = 1.0
            elif i == 4:
                c_init = np.zeros(21)
                c_init[10:16] = 1.0
                c_init /= np.linalg.norm(c_init)
            elif i == 5:
                c_init = np.zeros(21)
                c_init[10:14] = 1.0
                c_init /= np.linalg.norm(c_init)
            elif i == 6:
                c_init = np.zeros(21)
                c_init[10:12] = 1.0
                c_init /= np.linalg.norm(c_init)
            elif i == 7:
                c_init = np.zeros(21)
                c_init[10:18] = 1.0
                c_init /= np.linalg.norm(c_init)
            elif i == 8:
                c_init = np.zeros(21)
                c_init[10:21] = 1.0
                c_init /= np.linalg.norm(c_init)
            elif i == 9:
                c_init = np.zeros(21)
                for m in range(0, 11, 2):
                    c_init[10 + m] = 1.0
                    if m > 0:
                        c_init[10 - m] = 1.0
                c_init /= np.linalg.norm(c_init)
            elif i == 10:
                c_init = np.zeros(21)
                c_init[0] = 1.0
                c_init[1] = 0.5
                c_init[19] = 0.5
                c_init[20] = 1.0
                c_init /= np.linalg.norm(c_init)
            elif i == 11:
                c_init = np.zeros(21)
                c_init[9] = 1.0
                c_init[10] = 2.0
                c_init[11] = 1.0
                c_init /= np.linalg.norm(c_init)
            elif i == 12:
                c_init = np.zeros(21)
                for m in range(21):
                    idx = m - 10
                    c_init[m] = (-1)**idx * np.exp(-abs(idx) / 5.0)
                c_init /= np.linalg.norm(c_init)
            elif i == 13:
                c_init = np.zeros(21)
                for m in range(21):
                    idx = m - 10
                    c_init[m] = np.sin(idx * np.pi / 20.0)
                c_init /= np.linalg.norm(c_init)
            elif i == 14:
                c_init = np.zeros(21)
                for m in range(21):
                    idx = m - 10
                    c_init[m] = np.cos(idx * np.pi / 20.0)
                c_init /= np.linalg.norm(c_init)
            elif i == 15:
                # Concentrate on m=±5, ±6
                c_init = np.zeros(21)
                c_init[4] = 1.0
                c_init[5] = 1.0
                c_init[16] = 1.0
                c_init[17] = 1.0
                c_init /= np.linalg.norm(c_init)
            elif i == 16:
                # Concentrate on m=±3, ±4
                c_init = np.zeros(21)
                c_init[6] = 1.0
                c_init[7] = 1.0
                c_init[14] = 1.0
                c_init[15] = 1.0
                c_init /= np.linalg.norm(c_init)
            elif i == 17:
                # Peak at m=0, decay
                c_init = np.zeros(21)
                for m in range(21):
                    idx = m - 10
                    c_init[m] = np.exp(-idx**2 / 8.0)
                c_init /= np.linalg.norm(c_init)
            elif i == 18:
                # Peak at m=±8
                c_init = np.zeros(21)
                c_init[2] = 1.0
                c_init[3] = 0.5
                c_init[18] = 1.0
                c_init[19] = 0.5
                c_init /= np.linalg.norm(c_init)
            elif i == 19:
                # Mix of low and high
                c_init = np.zeros(21)
                c_init[10] = 1.0
                c_init[0] = 0.5
                c_init[20] = 0.5
                c_init[9] = 0.3
                c_init[11] = 0.3
                c_init /= np.linalg.norm(c_init)
            elif i == 20:
                # Concentrate on m=±2, ±7
                c_init = np.zeros(21)
                c_init[8] = 1.0
                c_init[12] = 1.0
                c_init[3] = 1.0
                c_init[17] = 1.0
                c_init /= np.linalg.norm(c_init)
            elif i == 21:
                # Concentrate on m=±1, ±8
                c_init = np.zeros(21)
                c_init[9] = 1.0
                c_init[11] = 1.0
                c_init[2] = 1.0
                c_init[18] = 1.0
                c_init /= np.linalg.norm(c_init)
            elif i == 22:
                # Concentrate on m=±4, ±9
                c_init = np.zeros(21)
                c_init[6] = 1.0
                c_init[14] = 1.0
                c_init[1] = 1.0
                c_init[19] = 1.0
                c_init /= np.linalg.norm(c_init)
            elif i == 23:
                # Broad: all modes with linear decay from center
                c_init = np.zeros(21)
                for m in range(21):
                    idx = m - 10
                    c_init[m] = 1.0 - 0.08 * abs(idx)
                c_init /= np.linalg.norm(c_init)
            else:
                # Very broad: all modes equal
                c_init = np.ones(21) / np.sqrt(21)
        else:
            c_init = _random_unit_vector(rng)
        
        result = minimize(
            objective_with_grad,
            c_init,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 200}
        )
        
        if result.fun < best_ratio:
            c_opt = result.x / np.linalg.norm(result.x)
            r_opt = _compute_ratio(c_opt, Y, weights)
            if r_opt < best_ratio:
                best_ratio = r_opt
                best_coeffs = c_opt
    
    # Phase 2: Simulated annealing from best
    if time.time() < deadline - safety_margin:
        c_sa, r_sa = _simulated_annealing(
            best_coeffs, Y, weights,
            n_steps=8000, t_start=0.3, t_end=0.0003, rng=rng
        )
        if r_sa < best_ratio:
            best_ratio = r_sa
            best_coeffs = c_sa
    
    # Phase 3: More targeted restarts near the best solution
    if time.time() < deadline - safety_margin:
        for i in range(60):
            if time.time() > deadline - safety_margin:
                break
            
            # Perturbations of decreasing magnitude
            scale = 0.2 * (0.8 ** (i // 15))
            c_init = best_coeffs + scale * rng.standard_normal(21)
            c_init /= np.linalg.norm(c_init)
            
            result = minimize(
                objective_with_grad,
                c_init,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 150}
            )
            
            if result.fun < best_ratio:
                c_opt = result.x / np.linalg.norm(result.x)
                r_opt = _compute_ratio(c_opt, Y, weights)
                if r_opt < best_ratio:
                    best_ratio = r_opt
                    best_coeffs = c_opt
    
    # Phase 4: High-precision polish
    if time.time() < deadline - safety_margin:
        c_polished = _polish_local(best_coeffs, Y, weights, deadline, safety_margin)
        r_polished = _compute_ratio(c_polished, Y, weights)
        if r_polished < best_ratio:
            best_ratio = r_polished
            best_coeffs = c_polished
    
    # Phase 5: Final coordinate descent refinement
    if time.time() < deadline - safety_margin:
        c = best_coeffs.copy()
        for _ in range(300):
            if time.time() > deadline - safety_margin:
                break
            improved = False
            for m in range(21):
                if time.time() > deadline - safety_margin:
                    break
                for delta in [-0.3, -0.15, -0.05, -0.02, 0.02, 0.05, 0.15, 0.3]:
                    c_new = c.copy()
                    c_new[m] += delta * 0.02
                    c_new /= np.linalg.norm(c_new)
                    r_new = _compute_ratio(c_new, Y, weights)
                    if r_new < best_ratio - 1e-14:
                        c = c_new
                        best_ratio = r_new
                        improved = True
                        break
            if not improved:
                break
        best_coeffs = c
    
    # Phase 6: Final polish
    if time.time() < deadline - safety_margin:
        c_polished = _polish_local(best_coeffs, Y, weights, deadline, safety_margin)
        r_polished = _compute_ratio(c_polished, Y, weights)
        if r_polished < best_ratio:
            best_ratio = r_polished
            best_coeffs = c_polished
    
    # Phase 7: Second round of targeted restarts with different random seed
    if time.time() < deadline - safety_margin:
        rng2 = np.random.default_rng(12345)
        for i in range(30):
            if time.time() > deadline - safety_margin:
                break
            
            scale = 0.15 * (0.9 ** (i // 10))
            c_init = best_coeffs + scale * rng2.standard_normal(21)
            c_init /= np.linalg.norm(c_init)
            
            result = minimize(
                objective_with_grad,
                c_init,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 150}
            )
            
            if result.fun < best_ratio:
                c_opt = result.x / np.linalg.norm(result.x)
                r_opt = _compute_ratio(c_opt, Y, weights)
                if r_opt < best_ratio:
                    best_ratio = r_opt
                    best_coeffs = c_opt
    
    # Phase 8: Another polish
    if time.time() < deadline - safety_margin:
        c_polished = _polish_local(best_coeffs, Y, weights, deadline, safety_margin)
        r_polished = _compute_ratio(c_polished, Y, weights)
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
