import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time

def _build_harmonics():
    """Precompute real orthonormal degree-10 spherical harmonics on the quadrature grid."""
    k = 10
    x, w = np.polynomial.legendre.leggauss(70)
    phi = np.linspace(0, 2*np.pi, 130, endpoint=False)
    
    cos_theta, phi_grid = np.meshgrid(x, phi, indexing='ij')
    dphi = 2*np.pi / 130
    weights_2d = np.outer(w, np.ones(130)) * dphi
    
    P = {}
    for m in range(0, 11):
        P[m] = lpmv(m, 10, cos_theta)
    
    N = {}
    for m in range(0, 11):
        N[m] = np.sqrt(21.0/(4*np.pi) * factorial(10-m)/factorial(10+m))
    
    Y = np.zeros((21, 70, 130))
    for idx, m in enumerate(range(-10, 11)):
        if m == 0:
            Y[idx] = N[0] * P[0]
        elif m > 0:
            Y[idx] = np.sqrt(2) * N[m] * P[m] * np.cos(m * phi_grid)
        else:
            am = abs(m)
            Y[idx] = np.sqrt(2) * N[am] * P[am] * np.sin(am * phi_grid)
    
    return Y, weights_2d

def _compute_f(coeffs, Y):
    """Compute f = sum c_m Y_m on the grid."""
    return np.tensordot(coeffs, Y, axes=(0, 0))

def _objective(coeffs, Y, weights_2d):
    """Compute ||f||_4 / ||f||_2."""
    f = _compute_f(coeffs, Y)
    w = weights_2d
    norm2_sq = np.sum(f**2 * w)
    if norm2_sq < 1e-30:
        return 1e10
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sum(f**4 * w)**0.25
    return norm4 / norm2

def _objective_and_grad(coeffs, Y, weights_2d):
    """Compute objective and correct analytic gradient."""
    f = _compute_f(coeffs, Y)
    w = weights_2d
    norm2_sq = np.sum(f**2 * w)
    if norm2_sq < 1e-30:
        return 1e10, np.zeros_like(coeffs)
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sum(f**4 * w)**0.25
    
    dnorm4_dc = np.einsum('ij,mij,ij->m', f**3, Y, w) / (norm4**3)
    dnorm2_dc = np.einsum('ij,mij,ij->m', f, Y, w) / norm2
    
    grad = dnorm4_dc / norm2 - (norm4 / norm2**2) * dnorm2_dc
    
    return norm4 / norm2, grad

def _objective_normalized_grad(c, Y, weights_2d):
    """Gradient of ||f||_4 / ||f||_2 with respect to c, projected to tangent of unit sphere."""
    f = _compute_f(c, Y)
    w = weights_2d
    norm2_sq = np.sum(f**2 * w)
    if norm2_sq < 1e-30:
        return 1e10, np.zeros_like(c)
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sum(f**4 * w)**0.25
    
    dnorm4_dc = np.einsum('ij,mij,ij->m', f**3, Y, w) / (norm4**3)
    dnorm2_dc = np.einsum('ij,mij,ij->m', f, Y, w) / norm2
    
    grad = dnorm4_dc / norm2 - (norm4 / norm2**2) * dnorm2_dc
    
    dot = np.dot(grad, c)
    grad_t = grad - dot * c
    
    return norm4 / norm2, grad_t

def _sphere_gradient_descent(c, Y, weights_2d, max_iter=200, tol=1e-12):
    """Gradient descent on the unit sphere with adaptive step size."""
    c = c / np.linalg.norm(c)
    
    for it in range(max_iter):
        R, grad_t = _objective_normalized_grad(c, Y, weights_2d)
        
        if np.linalg.norm(grad_t) < tol:
            break
        
        step = 0.1
        improved = False
        for _ in range(200):
            c_new = c - step * grad_t
            norm_new = np.linalg.norm(c_new)
            if norm_new < 1e-10:
                step *= 0.5
                continue
            c_new = c_new / norm_new
            R_new, _ = _objective_normalized_grad(c_new, Y, weights_2d)
            if R_new < R:
                c = c_new
                improved = True
                break
            step *= 0.5
        
        if not improved or step < 1e-14:
            break
    
    return c, _objective_normalized(c, Y, weights_2d) if False else None

def _objective_normalized(c, Y, weights_2d):
    """Objective on unit sphere."""
    f = _compute_f(c, Y)
    w = weights_2d
    norm2_sq = np.sum(f**2 * w)
    if norm2_sq < 1e-30:
        return 1e10
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sum(f**4 * w)**0.25
    return norm4 / norm2

def _newton_step_on_sphere(c, Y, weights_2d, max_iter=100, tol=1e-12):
    """Newton's method on the unit sphere to find stationary points of R(c)."""
    c = c / np.linalg.norm(c)
    n = len(c)
    
    for it in range(max_iter):
        R, grad_t = _objective_normalized_grad(c, Y, weights_2d)
        
        if np.linalg.norm(grad_t) < tol:
            break
        
        step = 0.1
        improved = False
        for _ in range(200):
            c_new = c - step * grad_t
            norm_new = np.linalg.norm(c_new)
            if norm_new < 1e-10:
                step *= 0.5
                continue
            c_new = c_new / norm_new
            R_new, _ = _objective_normalized_grad(c_new, Y, weights_2d)
            if R_new < R:
                c = c_new
                improved = True
                break
            step *= 0.5
        
        if not improved or step < 1e-14:
            break
    
    return c, _objective_normalized(c, Y, weights_2d)

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    """
    start_time = time.time()
    deadline = 580.0  # safety margin before 630s limit
    
    Y, weights_2d = _build_harmonics()
    
    best_coeffs = np.ones(21) / np.sqrt(21)
    best_score = _objective(best_coeffs, Y, weights_2d)
    
    # Structured initial guesses
    structured_guesses = []
    
    # m=0 only
    c = np.zeros(21)
    c[10] = 1.0
    structured_guesses.append(c)
    
    # m=0 dominant
    c = np.zeros(21)
    c[10] = 1.0
    for m in [1, 2, 3, 4, 5]:
        c[10+m] = 0.1
        c[10-m] = 0.1
    structured_guesses.append(c)
    
    # Even m cos only
    c = np.zeros(21)
    for m in [0, 2, 4, 6, 8, 10]:
        c[10+m] = 1.0
    structured_guesses.append(c)
    
    # All cos terms
    c = np.zeros(21)
    for m in range(0, 11):
        c[10+m] = 1.0
    structured_guesses.append(c)
    
    # Symmetric: c[m] = c[-m]
    c = np.zeros(21)
    for m in range(0, 11):
        c[10+m] = 1.0
        if m > 0:
            c[10-m] = 1.0
    structured_guesses.append(c)
    
    # Alternating signs
    c = np.zeros(21)
    for m in range(0, 11):
        c[10+m] = (-1)**m
    structured_guesses.append(c)
    
    # Random with different seeds
    for seed in [1, 2, 3, 4, 5]:
        np.random.seed(seed)
        c = np.random.randn(21)
        structured_guesses.append(c)
    
    for c in structured_guesses:
        s = _objective(c, Y, weights_2d)
        if s < best_score and np.isfinite(s):
            best_score = s
            best_coeffs = c.copy()
    
    # Multi-restart optimization with correct gradient
    np.random.seed(42)
    n_restarts = 50
    
    for i in range(n_restarts):
        if time.time() - start_time > deadline - 60:
            break
        
        # Generate diverse starting points
        if i < 10:
            # Random unit vectors
            x0 = np.random.randn(21)
        elif i < 20:
            # Sparse random
            x0 = np.random.randn(21)
            mask = np.random.random(21) < 0.3
            x0 *= mask
            if np.linalg.norm(x0) < 1e-10:
                x0 = np.random.randn(21)
        elif i < 30:
            # Perturb best solution
            x0 = best_coeffs + 0.5 * np.random.randn(21)
        else:
            # Mix of random and structured
            x0 = np.random.randn(21)
            x0[10] += 2.0  # emphasize m=0
        
        x0 = x0 / np.linalg.norm(x0) * 5.0
        
        try:
            res = minimize(
                lambda c: _objective(c, Y, weights_2d),
                x0,
                jac=lambda c: _objective_and_grad(c, Y, weights_2d)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Polish with tighter tolerance
    if time.time() - start_time < deadline - 30:
        try:
            res = minimize(
                lambda c: _objective(c, Y, weights_2d),
                best_coeffs,
                jac=lambda c: _objective_and_grad(c, Y, weights_2d)[1],
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100}
            )
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Final gradient flow polish on the sphere
    if time.time() - start_time < deadline - 20:
        try:
            c_norm = best_coeffs / np.linalg.norm(best_coeffs)
            c_polished, R_polished = _newton_step_on_sphere(c_norm, Y, weights_2d, max_iter=200)
            # Scale back
            c_polished_scaled = c_polished * np.linalg.norm(best_coeffs)
            R_scaled = _objective(c_polished_scaled, Y, weights_2d)
            if R_scaled < best_score and np.isfinite(R_scaled):
                best_score = R_scaled
                best_coeffs = c_polished_scaled
        except Exception:
            pass
    
    # Validate
    if len(best_coeffs) != 21 or not np.all(np.isfinite(best_coeffs)):
        return np.ones(21)
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs < 1e-3:
            best_coeffs *= 1.0 / max_abs if max_abs > 0 else 1.0
        elif max_abs > 1e5:
            best_coeffs *= 1e5 / max_abs
        if not np.all(np.isfinite(best_coeffs)):
            return np.ones(21)
    
    return best_coeffs.astype(np.float64)
