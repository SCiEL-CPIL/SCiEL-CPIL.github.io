"""Exact analytic objective + Riemannian multi-start optimization for uncertainty bound."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from math import factorial, sqrt, pi


def _compute_bound_analytic(c0, c4, c8):
    """
    Compute the uncertainty product bound A(f)*A(F) analytically using
    Hermite polynomial orthogonality. This is exact (no discretization error).
    
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is fixed so g(0) = 0.
    
    The bound is sqrt(<x^2> - <x>^2) * sqrt(<p^2> - <p>^2) / ||g||^2
    which simplifies to sqrt(delta_x_sq * delta_p_sq).
    """
    # c12 determined by g(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=665280
    c12 = -(c0 * 1.0 + c4 * 12.0 + c8 * 1680.0) / 665280.0
    
    coeffs = np.array([c0, c4, c8, c12])
    indices = [0, 4, 8, 12]
    
    # Norms: <H_n, H_n> = sqrt(pi) * 2^n * n!
    norms = np.array([sqrt(pi) * (2**n) * factorial(n) for n in indices])
    
    # Norm squared
    norm_sq = np.sum(coeffs**2 * norms)
    if norm_sq < 1e-30:
        return np.inf
    
    # <x^2> = sum c_m^2 * (m + 0.5) * <H_m, H_m> / norm_sq
    x2_num = np.sum(coeffs**2 * np.array([(n + 0.5) * norm for n, norm in zip(indices, norms)]))
    delta_x_sq = x2_num / norm_sq
    
    # <p^2> via Q(x) = P'(x) - x*P(x)
    # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # H_n' = n * H_{n-1}
    # x*H_n = 0.5*(H_{n+1} + n*H_{n-1})
    # So H_n' - x*H_n = n*H_{n-1} - 0.5*H_{n+1} - 0.5*n*H_{n-1} = 0.5*n*H_{n-1} - 0.5*H_{n+1}
    
    # For H0: 0.5*0*H_{-1} - 0.5*H1 = -0.5*H1
    # For H4: 0.5*4*H3 - 0.5*H5 = 2*H3 - 0.5*H5
    # For H8: 0.5*8*H7 - 0.5*H9 = 4*H7 - 0.5*H9
    # For H12: 0.5*12*H11 - 0.5*H13 = 6*H11 - 0.5*H13
    
    # Wait, let me recheck. The standard relation is:
    # d/dx H_n = 2n H_{n-1}
    # x H_n = (1/2)(H_{n+1} + 2n H_{n-1})  -- NO, this is wrong
    # Actually: x H_n(x) = (1/2)(H_{n+1}(x) + 2n H_{n-1}(x)) is NOT standard.
    # The correct relations are:
    # H_n'(x) = 2n H_{n-1}(x)
    # x H_n(x) = (1/2) H_{n+1}(x) + n H_{n-1}(x)
    
    # So Q_n = H_n' - x H_n = 2n H_{n-1} - (1/2 H_{n+1} + n H_{n-1}) = n H_{n-1} - 1/2 H_{n+1}
    
    # For H0: 0*H_{-1} - 1/2*H1 = -1/2 H1
    # For H4: 4*H3 - 1/2*H5
    # For H8: 8*H7 - 1/2*H9
    # For H12: 12*H11 - 1/2*H13
    
    d1 = -0.5 * c0
    d3 = 4.0 * c4
    d5 = -0.5 * c4
    d7 = 8.0 * c8
    d9 = -0.5 * c8
    d11 = 12.0 * c12
    d13 = -0.5 * c12
    
    d_coeffs = np.array([d1, d3, d5, d7, d9, d11, d13])
    odd_indices = [1, 3, 5, 7, 9, 11, 13]
    odd_norms = np.array([sqrt(pi) * (2**k) * factorial(k) for k in odd_indices])
    
    p2_num = np.sum(d_coeffs**2 * odd_norms)
    delta_p_sq = p2_num / norm_sq
    
    if delta_x_sq < 0 or delta_p_sq < 0:
        return np.inf
    
    return np.sqrt(delta_x_sq * delta_p_sq)


def _objective_and_grad(c):
    """
    Compute the objective value and analytic gradient for the uncertainty bound.
    c = [c0, c4, c8], c12 is determined by g(0)=0.
    Returns (value, gradient) where gradient is with respect to c.
    """
    c0, c4, c8 = c
    c12 = -(c0 + 12.0 * c4 + 1680.0 * c8) / 665280.0
    
    coeffs = np.array([c0, c4, c8, c12])
    indices = [0, 4, 8, 12]
    norms = np.array([sqrt(pi) * (2**n) * factorial(n) for n in indices])
    
    norm_sq = np.sum(coeffs**2 * norms)
    if norm_sq < 1e-30:
        return np.inf, np.zeros(3)
    
    x2_diag = np.array([(n + 0.5) * norm for n, norm in zip(indices, norms)])
    x2_num = np.sum(coeffs**2 * x2_diag)
    delta_x_sq = x2_num / norm_sq
    
    d1 = -0.5 * c0
    d3 = 4.0 * c4
    d5 = -0.5 * c4
    d7 = 8.0 * c8
    d9 = -0.5 * c8
    d11 = 12.0 * c12
    d13 = -0.5 * c12
    
    d_coeffs = np.array([d1, d3, d5, d7, d9, d11, d13])
    odd_indices = [1, 3, 5, 7, 9, 11, 13]
    odd_norms = np.array([sqrt(pi) * (2**k) * factorial(k) for k in odd_indices])
    
    p2_num = np.sum(d_coeffs**2 * odd_norms)
    delta_p_sq = p2_num / norm_sq
    
    if delta_x_sq <= 0 or delta_p_sq <= 0:
        return np.inf, np.zeros(3)
    
    val = np.sqrt(delta_x_sq * delta_p_sq)
    
    # Analytic gradient via finite differences for robustness (3D is cheap)
    eps = 1e-8
    grad = np.zeros(3)
    for i in range(3):
        cp = c.copy()
        cm = c.copy()
        cp[i] += eps
        cm[i] -= eps
        vp = _compute_bound_analytic(cp[0], cp[1], cp[2])
        vm = _compute_bound_analytic(cm[0], cm[1], cm[2])
        if np.isfinite(vp) and np.isfinite(vm):
            grad[i] = (vp - vm) / (2 * eps)
        else:
            grad[i] = 0.0
    
    return val, grad


def _objective_only(c):
    """Just the objective value."""
    val, _ = _objective_and_grad(c)
    return val


def construct():
    t0 = time.time()
    deadline = t0 + 175.0  # Safety margin vs 210s kill
    
    def elapsed():
        return time.time() - t0
    
    # Precompute constants
    indices = [0, 4, 8, 12]
    norms = np.array([sqrt(pi) * (2**n) * factorial(n) for n in indices])
    x2_diag = np.array([(n + 0.5) * norm for n, norm in zip(indices, norms)])
    odd_indices = [1, 3, 5, 7, 9, 11, 13]
    odd_norms = np.array([sqrt(pi) * (2**k) * factorial(k) for k in odd_indices])
    
    # Seed solution
    c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    seed_val = _compute_bound_analytic(c_seed[0], c_seed[1], c_seed[2])
    
    best_c = c_seed.copy()
    best_val = seed_val
    
    # Multi-start Riemannian optimization
    rng = np.random.RandomState(12345)
    
    def project_to_sphere(c):
        n = np.linalg.norm(c)
        if n < 1e-15:
            return c.copy()
        return c / n
    
    def try_refine(c0):
        nonlocal best_c, best_val
        if elapsed() > deadline:
            return
        c0 = project_to_sphere(c0.copy())
        
        # Riemannian gradient descent with backtracking
        c = c0.copy()
        val, grad = _objective_and_grad(c)
        if not np.isfinite(val):
            return
        
        lr = 0.5
        for it in range(300):
            if elapsed() > deadline:
                break
            # Project gradient to tangent space
            gproj = grad - float(np.dot(grad, c)) * c
            gnorm = np.linalg.norm(gproj)
            if gnorm < 1e-14:
                break
            
            dstep = -gproj / gnorm
            # Backtracking line search
            alpha = lr
            improved = False
            for _ in range(25):
                cand = project_to_sphere(c + alpha * dstep)
                cand_val, _ = _objective_and_grad(cand)
                if np.isfinite(cand_val) and cand_val < val * (1 - 1e-10):
                    c = cand
                    val = cand_val
                    improved = True
                    break
                alpha *= 0.5
            if not improved:
                lr *= 0.7
                if lr < 1e-10:
                    break
        
        # Polish with L-BFGS-B
        if elapsed() < deadline - 3.0:
            try:
                def cart_obj(c):
                    v = _compute_bound_analytic(c[0], c[1], c[2])
                    return v if np.isfinite(v) else 1e10
                
                def cart_grad(c):
                    v, g = _objective_and_grad(c)
                    if not np.isfinite(v):
                        return np.zeros(3)
                    return g
                
                res = minimize(
                    cart_obj, x0=c,
                    jac=cart_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 60}
                )
                if np.isfinite(res.fun) and res.fun < val:
                    c = project_to_sphere(res.x)
                    val, _ = _objective_and_grad(c)
            except Exception:
                pass
        
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_c = c.copy()
    
    # Starting points
    starts = []
    starts.append(c_seed / np.linalg.norm(c_seed))
    
    # Coordinate axes
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1.0
        starts.append(e)
    
    # Random directions
    for _ in range(20):
        d = rng.randn(3)
        starts.append(d / np.linalg.norm(d))
    
    # Also try some structured starts near the seed
    for scale in [0.5, 1.5, 2.0]:
        starts.append(c_seed * scale / np.linalg.norm(c_seed * scale))
    
    for s in starts:
        try_refine(s)
        if elapsed() > deadline - 5.0:
            break
    
    # Final high-precision polish
    if elapsed() < deadline - 2.0:
        try:
            def cart_obj(c):
                v = _compute_bound_analytic(c[0], c[1], c[2])
                return v if np.isfinite(v) else 1e10
            
            def cart_grad(c):
                v, g = _objective_and_grad(c)
                if not np.isfinite(v):
                    return np.zeros(3)
                return g
            
            res = minimize(
                cart_obj, x0=best_c,
                jac=cart_grad,
                method='L-BFGS-B',
                options={'maxiter': 20000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 80}
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_c = project_to_sphere(res.x)
                best_val = _compute_bound_analytic(best_c[0], best_c[1], best_c[2])
        except Exception:
            pass
    
    return np.array([best_c[0], best_c[1], best_c[2]])

# EVOLVE-BLOCK-END
