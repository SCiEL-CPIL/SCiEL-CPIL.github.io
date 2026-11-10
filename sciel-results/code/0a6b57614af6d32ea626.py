# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, gammaln
from scipy.optimize import minimize

_DEADLINE = time.time() + 615

def _build_quadrature():
    """70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    cos_theta = x_gl
    phi = np.linspace(0, 2 * np.pi, 130, endpoint=False)
    dphi = 2.0 * np.pi / 130
    weights_1d = w_gl * dphi
    return cos_theta, phi, weights_1d

def _compute_harmonics_matrix(cos_theta, phi):
    """
    Compute the matrix X where X[i, j] = Y_j(ct_i, phi_j).
    j ranges from 0 to 20, corresponding to m = -10 to 10.
    """
    N_pts_ct = len(cos_theta)
    N_pts_phi = len(phi)
    X = np.zeros((N_pts_ct * N_pts_phi, 21))
    
    ct, ph = np.meshgrid(cos_theta, phi, indexing='ij')
    ct_flat = ct.flatten()
    ph_flat = ph.flatten()
    
    for k, m in enumerate(range(-10, 11)):
        abs_m = abs(m)
        log_N = 0.5 * (np.log(21.0 / (4.0 * np.pi)) + 
                       gammaln(10 - abs_m + 1) - gammaln(10 + abs_m + 1))
        N_m = np.exp(log_N)
        
        P = lpmv(abs_m, 10, ct_flat)
        
        if m == 0:
            Y = N_m * P
        elif m > 0:
            Y = np.sqrt(2.0) * N_m * P * np.cos(m * ph_flat)
        else:
            Y = np.sqrt(2.0) * N_m * P * np.sin(abs_m * ph_flat)
            
        X[:, k] = Y
        
    return X

def _ratio_and_grad(c, X, w_flat):
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient.
    """
    f = X @ c
    f2 = f * f
    f3 = f2 * f
    f4 = f2 * f2
    
    S2 = np.sum(w_flat * f2)
    S4 = np.sum(w_flat * f4)
    
    if S2 < 1e-10 or S4 < 1e-10:
        return 1e10, np.zeros(21)
        
    N2 = np.sqrt(S2)
    N4 = S4 ** 0.25
    R = N4 / N2
    
    term1 = X.T @ (w_flat * f3)
    term2 = X.T @ (w_flat * f)
    
    grad_log_R = term1 / S4 - term2 / S2
    grad_R = R * grad_log_R
    
    return R, grad_R

def _objective_log(c, X, w_flat):
    """
    Minimize log(R) = 0.25*log(S4) - 0.5*log(S2)
    This is scale-invariant and smooth.
    """
    f = X @ c
    f2 = f * f
    f4 = f2 * f2
    
    S2 = np.sum(w_flat * f2)
    S4 = np.sum(w_flat * f4)
    
    if S2 < 1e-10 or S4 < 1e-10:
        return 1e10, np.zeros(21)
    
    obj = 0.25 * np.log(S4) - 0.5 * np.log(S2)
    
    # Gradient: d/dc [0.25 log S4 - 0.5 log S2]
    # = 0.25 * (1/S4) * dS4/dc - 0.5 * (1/S2) * dS2/dc
    # dS4/dc = 4 * sum(w_i f_i^3 X_i)
    # dS2/dc = 2 * sum(w_i f_i X_i)
    f3 = f2 * f
    term1 = X.T @ (w_flat * f3)  # = 1/4 dS4/dc
    term2 = X.T @ (w_flat * f)   # = 1/2 dS2/dc
    
    grad = 0.25 * term1 / S4 - 0.5 * term2 / S2
    
    return obj, grad

def _riemannian_step(c, grad, step_size, X, w_flat):
    """
    Perform a Riemannian gradient step on the unit sphere.
    Project gradient to tangent space, then use exponential map.
    """
    # Project gradient onto tangent space: g_tan = g - (g·c)c
    gc = np.dot(grad, c)
    g_tan = grad - gc * c
    
    # Step in tangent space
    v = -step_size * g_tan
    
    # Exponential map on sphere: exp_c(v) = c*cos(||v||) + (v/||v||)*sin(||v||)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-15:
        return c
    
    cos_v = np.cos(v_norm)
    sin_v = np.sin(v_norm) / v_norm
    
    c_new = c * cos_v + v * sin_v
    
    # Re-normalize for numerical stability
    c_new = c_new / np.linalg.norm(c_new)
    
    return c_new

def _riemannian_optimize(c0, X, w_flat, max_iter=5000, tol=1e-14):
    """
    Riemannian optimization on the unit sphere.
    Uses gradient flow with adaptive step size.
    """
    c = c0.copy()
    c = c / np.linalg.norm(c)
    
    step_size = 0.1
    best_c = c.copy()
    best_val = _ratio_and_grad(c, X, w_flat)[0]
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
        
        val, grad = _ratio_and_grad(c, X, w_flat)
        
        if val < best_val:
            best_val = val
            best_c = c.copy()
        
        # Check convergence
        grad_tan = grad - np.dot(grad, c) * c
        grad_norm = np.linalg.norm(grad_tan)
        
        if grad_norm < tol:
            break
        
        # Adaptive step size: use Armijo-like line search
        accepted = False
        for ls in range(20):
            s = step_size * (0.5 ** ls)
            c_new = _riemannian_step(c, grad, s, X, w_flat)
            val_new = _ratio_and_grad(c_new, X, w_flat)[0]
            
            if val_new < val - 1e-4 * s * grad_norm:
                c = c_new
                accepted = True
                break
        
        if not accepted:
            step_size *= 0.5
            if step_size < 1e-10:
                break
        else:
            step_size = min(step_size * 1.1, 0.5)
    
    return best_c, best_val

def rudin_problem10() -> np.ndarray:
    """
    Minimize ||f||_4 / ||f||_2 for degree-10 real spherical harmonics.
    Returns 21 coefficients c_m for m = -10..10.
    """
    global _DEADLINE
    
    cos_theta, phi, w_1d = _build_quadrature()
    X = _compute_harmonics_matrix(cos_theta, phi)
    w_flat = np.tile(w_1d, 130)
    
    best_coeffs = None
    best_val = np.inf
    
    rng = np.random.default_rng(42)
    
    starts = []
    
    # Pure m starts
    for m in range(-10, 11):
        c = np.zeros(21)
        c[m + 10] = 1.0
        starts.append(c)
    
    # Structured mixed starts: pairs of m values with symmetry
    for m1 in range(-10, 11):
        for m2 in range(m1 + 1, 11):
            if (m1 + m2) % 2 == 0:  # same parity
                for weight in [0.3, 0.5, 0.7]:
                    c = np.zeros(21)
                    c[m1 + 10] = weight
                    c[m2 + 10] = np.sqrt(1.0 - weight * weight)
                    starts.append(c.copy())
                    c[m2 + 10] = -np.sqrt(1.0 - weight * weight)
                    starts.append(c.copy())
    
    # Random starts with various norms
    for _ in range(300):
        c = rng.standard_normal(21)
        norm = np.linalg.norm(c)
        if norm > 1e-10:
            c = c / norm
            starts.append(c)
    
    # More structured random starts with sparsity
    for _ in range(200):
        c = np.zeros(21)
        n_nonzero = rng.integers(2, 8)
        indices = rng.choice(21, size=n_nonzero, replace=False)
        c[indices] = rng.standard_normal(n_nonzero)
        norm = np.linalg.norm(c)
        if norm > 1e-10:
            c = c / norm
            starts.append(c)
    
    # Phase 1: Riemannian optimization from multiple starts
    n_starts = len(starts)
    for idx, start in enumerate(starts):
        if time.time() > _DEADLINE - 60:
            break
        
        # Adaptive max iterations based on remaining time
        remaining = _DEADLINE - time.time()
        max_iter = int(min(3000, max(200, remaining * 20)))
        
        try:
            c_opt, val = _riemannian_optimize(start, X, w_flat, max_iter=max_iter, tol=1e-13)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_coeffs = c_opt.copy()
        except Exception:
            pass
    
    # Phase 2: Refine best with SLSQP for higher precision
    if best_coeffs is not None and time.time() < _DEADLINE - 40:
        constraint = {'type': 'eq', 'fun': lambda c: np.dot(c, c) - 1.0, 'jac': lambda c: 2.0 * c}
        
        try:
            res = minimize(
                lambda c: _ratio_and_grad(c, X, w_flat)[0],
                best_coeffs,
                jac=lambda c: _ratio_and_grad(c, X, w_flat)[1],
                method='SLSQP',
                constraints=[constraint],
                options={'maxiter': 3000, 'ftol': 1e-15}
            )
            val = _ratio_and_grad(res.x, X, w_flat)[0]
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final polish with even tighter tolerance
    if best_coeffs is not None and time.time() < _DEADLINE - 20:
        constraint = {'type': 'eq', 'fun': lambda c: np.dot(c, c) - 1.0, 'jac': lambda c: 2.0 * c}
        
        try:
            res = minimize(
                lambda c: _ratio_and_grad(c, X, w_flat)[0],
                best_coeffs,
                jac=lambda c: _ratio_and_grad(c, X, w_flat)[1],
                method='SLSQP',
                constraints=[constraint],
                options={'maxiter': 5000, 'ftol': 1e-16}
            )
            val = _ratio_and_grad(res.x, X, w_flat)[0]
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Additional local search from best with small perturbations
    if best_coeffs is not None and time.time() < _DEADLINE - 15:
        constraint = {'type': 'eq', 'fun': lambda c: np.dot(c, c) - 1.0, 'jac': lambda c: 2.0 * c}
        
        for _ in range(15):
            if time.time() > _DEADLINE - 15:
                break
            
            perturbation = rng.standard_normal(21) * 0.01
            c_perturbed = best_coeffs + perturbation
            norm = np.linalg.norm(c_perturbed)
            if norm > 1e-10:
                c_perturbed = c_perturbed / norm
                
                try:
                    res = minimize(
                        lambda c: _ratio_and_grad(c, X, w_flat)[0],
                        c_perturbed,
                        jac=lambda c: _ratio_and_grad(c, X, w_flat)[1],
                        method='SLSQP',
                        constraints=[constraint],
                        options={'maxiter': 1000, 'ftol': 1e-14}
                    )
                    val = _ratio_and_grad(res.x, X, w_flat)[0]
                    if np.isfinite(val) and val < best_val:
                        best_val = val
                        best_coeffs = res.x.copy()
                except Exception:
                    pass
    
    if best_coeffs is None:
        best_coeffs = np.zeros(21)
        best_coeffs[10] = 1.0

    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = best_coeffs * (0.01 / max_abs)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs * (1e4 / max_abs)
        
    return best_coeffs

# EVOLVE-BLOCK-END
