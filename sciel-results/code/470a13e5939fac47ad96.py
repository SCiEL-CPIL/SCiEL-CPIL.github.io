# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
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
        # Use log-gamma for numerical stability
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
    
    d_norm4_sq_dc = 4.0 * np.einsum('ij,mij->m', w_f3, Y)
    d_norm2_sq_dc = 2.0 * np.einsum('ij,mij->m', w_f, Y)
    
    d_norm4_dc = (1.0 / (4.0 * norm4**3)) * d_norm4_sq_dc
    d_norm2_dc = (1.0 / (2.0 * norm2)) * d_norm2_sq_dc
    
    grad = d_norm4_dc / norm2 - norm4 * d_norm2_dc / (norm2 * norm2)
    
    return R, grad


def _riemannian_gradient(c, Y, weights):
    """
    Compute the Riemannian gradient of R on the unit sphere.
    For a function f: S^n -> R, the Riemannian gradient is the projection
    of the Euclidean gradient onto the tangent space at c.
    """
    R, euclidean_grad = _compute_ratio_and_grad(c, Y, weights)
    # Project onto tangent space: grad - <grad, c> * c
    inner = np.dot(euclidean_grad, c)
    riemannian_grad = euclidean_grad - inner * c
    return R, riemannian_grad


def _exp_map(c, grad, step_size):
    """
    Exponential map on the unit sphere.
    Moves from c in the direction of grad (tangent vector) by step_size.
    """
    # Normalize the direction
    grad_norm = np.linalg.norm(grad)
    if grad_norm < 1e-15:
        return c.copy()
    
    direction = grad / grad_norm
    # Move along the geodesic
    theta = step_size * grad_norm
    # If theta is very large, just go to the antipodal point
    if theta > np.pi:
        theta = np.pi
    
    new_c = np.cos(theta) * c + np.sin(theta) * direction
    new_c /= np.linalg.norm(new_c)
    return new_c


def _riemannian_lbfgs(c_init, Y, weights, maxiter=2000, deadline=None, safety_margin=90.0):
    """
    Riemannian L-BFGS optimization on the unit sphere.
    Uses the exponential map for line search.
    """
    c = c_init.copy()
    c /= np.linalg.norm(c)
    current_R, current_grad = _riemannian_gradient(c, Y, weights)
    best_c = c.copy()
    best_R = current_R
    
    # L-BFGS memory
    m_mem = 10
    s_list = []  # step vectors
    y_list = []  # gradient differences
    
    def line_search(c, grad, R, direction):
        """
        Backtracking line search using the exponential map.
        """
        # Initial step size
        step = 1.0
        # Ensure we're moving downhill
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-15:
            return c, R, 0.0
        
        direction = direction / dir_norm
        
        for _ in range(30):
            if deadline is not None and time.time() > deadline - safety_margin:
                break
            
            new_c = _exp_map(c, direction, step)
            new_R, _ = _riemannian_gradient(new_c, Y, weights)
            
            # Armijo condition
            if new_R < R + 1e-4 * step * np.dot(grad, direction):
                return new_c, new_R, step
            step *= 0.5
        
        return c, R, 0.0
    
    for iteration in range(maxiter):
        if deadline is not None and time.time() > deadline - safety_margin:
            break
        
        R, grad = _riemannian_gradient(c, Y, weights)
        
        if R < best_R:
            best_R = R
            best_c = c.copy()
        
        # Check convergence
        if np.linalg.norm(grad) < 1e-14:
            break
        
        # Compute search direction using L-BFGS
        # For simplicity, use a basic momentum-like approach
        # Full L-BFGS on Riemannian manifold is complex; use a simplified version
        
        # Simple gradient descent with adaptive step size
        grad_norm = np.linalg.norm(grad)
        step_size = 0.1 / (1 + iteration * 0.01)  # Decreasing step size
        
        new_c, new_R, actual_step = line_search(c, grad, R, -grad)
        
        if actual_step > 0:
            # Update L-BFGS memory
            if len(s_list) < m_mem:
                s_list.append(new_c - c)
                y_list.append(_riemannian_gradient(new_c, Y, weights)[1] - grad)
            else:
                s_list.pop(0)
                y_list.pop(0)
            
            c = new_c
            current_R = new_R
    
    return best_c, best_R


def _riemannian_flow(c_init, Y, weights, n_steps=5000, deadline=None, safety_margin=90.0):
    """
    Riemannian gradient flow with adaptive step size.
    """
    c = c_init.copy()
    c /= np.linalg.norm(c)
    current_R, current_grad = _riemannian_gradient(c, Y, weights)
    best_c = c.copy()
    best_R = current_R
    
    # Adaptive step size
    step_size = 0.1
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline - safety_margin:
            break
        
        R, grad = _riemannian_gradient(c, Y, weights)
        
        if R < best_R:
            best_R = R
            best_c = c.copy()
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        # Adaptive step size
        step_size = min(0.5, max(0.001, 0.1 / (1 + 0.01 * step)))
        
        # Move in negative gradient direction
        new_c = _exp_map(c, -grad, step_size)
        new_R, _ = _riemannian_gradient(new_c, Y, weights)
        
        # If it didn't improve, try smaller step
        if new_R > R:
            step_size *= 0.5
            new_c = _exp_map(c, -grad, step_size)
            new_R, _ = _riemannian_gradient(new_c, Y, weights)
        
        c = new_c
        current_R = new_R
    
    return best_c, best_R


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
    
    # Peak at m=±2
    c = np.zeros(21)
    c[8] = 1.0
    c[12] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Peak at m=±6
    c = np.zeros(21)
    c[4] = 1.0
    c[16] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Peak at m=±8
    c = np.zeros(21)
    c[2] = 1.0
    c[18] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    for c in patterns:
        r = _compute_ratio(c, Y, weights)
        if r < best_ratio:
            best_ratio = r
            best_coeffs = c.copy()
    
    return best_coeffs, best_ratio


def _simulated_annealing(c_init, Y, weights, n_steps=5000, t_start=0.5, t_end=0.001, rng=None, deadline=None, safety_margin=90.0):
    """Simulated annealing to find better coefficients."""
    if rng is None:
        rng = np.random.default_rng()
    
    c = c_init.copy()
    c /= np.linalg.norm(c)
    current_ratio = _compute_ratio(c, Y, weights)
    best_c = c.copy()
    best_ratio = current_ratio
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline - safety_margin:
            break
        
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
    
    # Phase 1: Riemannian optimization from multiple starting points
    n_restarts = 60
    
    for i in range(n_restarts):
        if time.time() > deadline - safety_margin:
            break
        
        if i == 0:
            c_init = best_coeffs.copy()
        elif i < 20:
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
            else:
                # Mix of low and high
                c_init = np.zeros(21)
                c_init[10] = 1.0
                c_init[0] = 0.5
                c_init[20] = 0.5
                c_init[9] = 0.3
                c_init[11] = 0.3
                c_init /= np.linalg.norm(c_init)
        else:
            c_init = _random_unit_vector(rng)
        
        # Use Riemannian gradient flow
        c_opt, r_opt = _riemannian_flow(
            c_init, Y, weights,
            n_steps=2000,
            deadline=deadline,
            safety_margin=safety_margin
        )
        
        if r_opt < best_ratio:
            best_ratio = r_opt
            best_coeffs = c_opt
    
    # Phase 2: Riemannian L-BFGS from best solution
    if time.time() < deadline - safety_margin:
        c_opt, r_opt = _riemannian_lbfgs(
            best_coeffs, Y, weights,
            maxiter=2000,
            deadline=deadline,
            safety_margin=safety_margin
        )
        
        if r_opt < best_ratio:
            best_ratio = r_opt
            best_coeffs = c_opt
    
    # Phase 3: Differential evolution for global search
    if time.time() < deadline - safety_margin:
        def de_objective(c):
            if time.time() > deadline - safety_margin:
                return _compute_ratio(best_coeffs, Y, weights)
            c_norm = np.linalg.norm(c)
            if c_norm < 1e-10:
                return 1e10
            c_normalized = c / c_norm
            return _compute_ratio(c_normalized, Y, weights)
        
        bounds = [(-1, 1)] * 21
        
        result = differential_evolution(
            de_objective,
            bounds,
            seed=rng.integers(0, 2**32 - 1),
            maxiter=80,
            tol=1e-12,
            polish=True,
            init='sobol',
            popsize=20
        )
        
        if result.fun < best_ratio:
            c_opt = result.x / np.linalg.norm(result.x)
            r_opt = _compute_ratio(c_opt, Y, weights)
            if r_opt < best_ratio:
                best_ratio = r_opt
                best_coeffs = c_opt
    
    # Phase 4: Simulated annealing from best
    if time.time() < deadline - safety_margin:
        c_sa, r_sa = _simulated_annealing(
            best_coeffs, Y, weights,
            n_steps=8000, t_start=0.3, t_end=0.0003, rng=rng,
            deadline=deadline, safety_margin=safety_margin
        )
        if r_sa < best_ratio:
            best_ratio = r_sa
            best_coeffs = c_sa
    
    # Phase 5: More targeted restarts near the best solution using Riemannian flow
    if time.time() < deadline - safety_margin:
        for i in range(40):
            if time.time() > deadline - safety_margin:
                break
            
            # Perturbations of decreasing magnitude
            scale = 0.2 * (0.8 ** (i // 10))
            c_init = best_coeffs + scale * rng.standard_normal(21)
            c_init /= np.linalg.norm(c_init)
            
            c_opt, r_opt = _riemannian_flow(
                c_init, Y, weights,
                n_steps=1000,
                deadline=deadline,
                safety_margin=safety_margin
            )
            
            if r_opt < best_ratio:
                best_ratio = r_opt
                best_coeffs = c_opt
    
    # Phase 6: High-precision polish
    if time.time() < deadline - safety_margin:
        c_polished, r_polished = _polish_local(best_coeffs, Y, weights, deadline, safety_margin)
        if r_polished < best_ratio:
            best_ratio = r_polished
            best_coeffs = c_polished
    
    # Phase 7: Final Riemannian polish
    if time.time() < deadline - safety_margin:
        c_opt, r_opt = _riemannian_flow(
            best_coeffs, Y, weights,
            n_steps=3000,
            deadline=deadline,
            safety_margin=safety_margin
        )
        if r_opt < best_ratio:
            best_ratio = r_opt
            best_coeffs = c_opt
    
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
