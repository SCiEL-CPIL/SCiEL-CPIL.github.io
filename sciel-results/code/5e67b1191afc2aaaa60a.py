import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize, differential_evolution
import time

def _build_harmonics():
    """Precompute real orthonormal degree-10 spherical harmonics on the quadrature grid."""
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
    
    # Precompute weighted harmonics for fast objective/gradient
    Y_w = Y * weights_2d  # (21, 70, 130)
    
    return Y, Y_w, weights_2d

def _objective(coeffs, Y, Y_w, weights_2d):
    """Compute ||f||_4 / ||f||_2."""
    f = np.tensordot(coeffs, Y, axes=(0, 0))
    w = weights_2d
    norm2_sq = np.sum(f**2 * w)
    if norm2_sq < 1e-30:
        return 1e10
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sum(f**4 * w)**0.25
    return norm4 / norm2

def _objective_and_grad(coeffs, Y, Y_w, weights_2d):
    """Compute objective and correct analytic gradient using precomputed weighted harmonics."""
    f = np.tensordot(coeffs, Y, axes=(0, 0))
    w = weights_2d
    norm2_sq = np.sum(f**2 * w)
    if norm2_sq < 1e-30:
        return 1e10, np.zeros_like(coeffs)
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sum(f**4 * w)**0.25
    
    # d||f||4/dc_m = (1/norm4^3) * sum_ij(f^3 * Y_m * w_ij) = (1/norm4^3) * sum_ij(f^3 * (Y_m * w_ij))
    # d||f||2/dc_m = (1/norm2) * sum_ij(f * Y_m * w_ij) = (1/norm2) * sum_ij(f * (Y_m * w_ij))
    # Using Y_w = Y * w: dnorm4_dc = Y_w @ (f^3) flattened, dnorm2_dc = Y_w @ f flattened
    
    f3_flat = (f**3).ravel()
    f_flat = f.ravel()
    
    dnorm4_dc = Y_w.reshape(21, -1) @ f3_flat / (norm4**3)
    dnorm2_dc = Y_w.reshape(21, -1) @ f_flat / norm2
    
    grad = dnorm4_dc / norm2 - (norm4 / norm2**2) * dnorm2_dc
    
    return norm4 / norm2, grad

def _objective_sphere(c, Y, Y_w, weights_2d):
    """Objective on unit sphere: since R is scale-invariant, minimize ||f||_4 when ||c||_2=1."""
    f = np.tensordot(c, Y, axes=(0, 0))
    w = weights_2d
    norm4 = np.sum(f**4 * w)**0.25
    return norm4

def _objective_sphere_grad(c, Y, Y_w, weights_2d):
    """Gradient of ||f||_4 with respect to c, projected to tangent of unit sphere."""
    f = np.tensordot(c, Y, axes=(0, 0))
    w = weights_2d
    norm4 = np.sum(f**4 * w)**0.25
    if norm4 < 1e-30:
        return 1e10, np.zeros_like(c)
    
    # d||f||4/dc_m = (1/norm4^3) * sum(f^3 * Y_m * w)
    f3_flat = (f**3).ravel()
    dnorm4_dc = Y_w.reshape(21, -1) @ f3_flat / (norm4**3)
    
    # Project to tangent space: grad_t = grad - (grad . c) * c
    dot = np.dot(dnorm4_dc, c)
    grad_t = dnorm4_dc - dot * c
    
    return norm4, grad_t

def _sphere_polish(c, Y, Y_w, weights_2d, max_iter=5000, tol=1e-14):
    """Sphere-constrained gradient descent with backtracking for polishing."""
    c = c.copy()
    nrm = np.linalg.norm(c)
    if nrm < 1e-15:
        return c
    c = c / nrm  # normalize to unit sphere
    
    prev_val = _objective_sphere(c, Y, Y_w, weights_2d)
    lr = 0.01
    
    for it in range(max_iter):
        val, grad_t = _objective_sphere_grad(c, Y, Y_w, weights_2d)
        if abs(prev_val - val) < tol:
            break
        prev_val = val
        
        # Backtracking line search
        step = lr
        improved = False
        for _ in range(30):
            c_new = c - step * grad_t
            c_new = c_new / np.linalg.norm(c_new)
            val_new = _objective_sphere(c_new, Y, Y_w, weights_2d)
            if val_new < val:
                c = c_new
                improved = True
                break
            step *= 0.5
        if not improved:
            break
        # Adaptive learning rate
        if val < prev_val:
            lr = min(lr * 1.2, 0.1)
        else:
            lr *= 0.5
    
    return c

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    """
    start_time = time.time()
    deadline = 580.0  # safety margin before 630s limit
    
    Y, Y_w, weights_2d = _build_harmonics()
    
    best_coeffs = np.ones(21) / np.sqrt(21)
    best_score = _objective(best_coeffs, Y, Y_w, weights_2d)
    
    # Structured initial guesses
    structured_guesses = []
    
    # m=0 only
    c = np.zeros(21)
    c[10] = 1.0
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
    
    # Symmetric: c[m] = c[-m] for all m
    c = np.zeros(21)
    for m in range(0, 11):
        c[10+m] = 1.0
        if m > 0:
            c[10-m] = 1.0
    structured_guesses.append(c)
    
    # Alternating signs on cos terms
    c = np.zeros(21)
    for m in range(0, 11):
        c[10+m] = (-1)**m
    structured_guesses.append(c)
    
    # m=0 dominant with small others
    c = np.zeros(21)
    c[10] = 1.0
    for m in [1, 2, 3, 4, 5]:
        c[10+m] = 0.1
        c[10-m] = 0.1
    structured_guesses.append(c)
    
    # Low-m biased: emphasize m=0,1,2
    c = np.zeros(21)
    c[10] = 1.0
    c[11] = 0.8
    c[9] = 0.8
    c[12] = 0.6
    c[8] = 0.6
    structured_guesses.append(c)
    
    # High-m biased
    c = np.zeros(21)
    for m in [8, 9, 10]:
        c[10+m] = 1.0
        if m > 0:
            c[10-m] = 1.0
    structured_guesses.append(c)
    
    # Triplet patterns
    c = np.zeros(21)
    for m in [0, 3, 6, 9]:
        c[10+m] = 1.0
        if m > 0:
            c[10-m] = 1.0
    structured_guesses.append(c)
    
    c = np.zeros(21)
    for m in [1, 4, 7, 10]:
        c[10+m] = 1.0
        if m > 0:
            c[10-m] = 1.0
    structured_guesses.append(c)
    
    for c in structured_guesses:
        s = _objective(c, Y, Y_w, weights_2d)
        if s < best_score and np.isfinite(s):
            best_score = s
            best_coeffs = c.copy()
    
    # Multi-restart optimization with correct gradient
    np.random.seed(42)
    n_restarts = 60
    
    for i in range(n_restarts):
        if time.time() - start_time > deadline - 120:
            break
        
        # Generate diverse starting points
        if i < 12:
            # Random unit vectors
            x0 = np.random.randn(21)
        elif i < 24:
            # Sparse random (various sparsity levels)
            sparsity = 0.2 + 0.6 * (i - 12) / 12
            x0 = np.random.randn(21)
            mask = np.random.random(21) < sparsity
            x0 *= mask
            if np.linalg.norm(x0) < 1e-10:
                x0 = np.random.randn(21)
        elif i < 36:
            # Perturb best solution with varying noise
            noise_scale = 0.3 + 0.7 * (i - 24) / 12
            x0 = best_coeffs + noise_scale * np.random.randn(21)
        elif i < 48:
            # Structured random with m-symmetry
            x0 = np.zeros(21)
            for m in range(11):
                if np.random.random() < 0.5:
                    val = np.random.randn()
                    x0[10+m] = val
                    if m > 0:
                        x0[10-m] = val * np.random.choice([-1, 1])
        else:
            # Low-m biased random
            x0 = np.zeros(21)
            for m in range(11):
                weight = 1.0 / (1.0 + 0.5*m)
                if np.random.random() < 0.7:
                    x0[10+m] = np.random.randn() * weight
                    if m > 0:
                        x0[10-m] = np.random.randn() * weight
        
        nrm = np.linalg.norm(x0)
        if nrm < 1e-10:
            x0 = np.random.randn(21)
            nrm = np.linalg.norm(x0)
        x0 = x0 / nrm * 5.0
        
        try:
            res = minimize(
                lambda c: _objective(c, Y, Y_w, weights_2d),
                x0,
                jac=lambda c: _objective_and_grad(c, Y, Y_w, weights_2d)[1],
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Differential evolution for global search
    if time.time() - start_time < deadline - 200:
        try:
            bounds = [(-10, 10)] * 21
            de_res = differential_evolution(
                lambda c: _objective(c, Y, Y_w, weights_2d),
                bounds,
                seed=42,
                maxiter=150,
                popsize=15,
                tol=1e-10,
                polish=True,
                init='sobol'
            )
            if de_res.fun < best_score and np.isfinite(de_res.fun):
                best_score = de_res.fun
                best_coeffs = de_res.x.copy()
        except Exception:
            pass
    
    # Sphere-constrained polish
    if time.time() - start_time < deadline - 40:
        try:
            polished = _sphere_polish(best_coeffs, Y, Y_w, weights_2d, max_iter=5000, tol=1e-14)
            s_polished = _objective(polished, Y, Y_w, weights_2d)
            if s_polished < best_score and np.isfinite(s_polished):
                best_score = s_polished
                best_coeffs = polished.copy()
        except Exception:
            pass
    
    # Final polish with high-precision L-BFGS-B using analytic gradient
    if time.time() - start_time < deadline - 20:
        try:
            res = minimize(
                lambda c: _objective(c, Y, Y_w, weights_2d),
                best_coeffs,
                jac=lambda c: _objective_and_grad(c, Y, Y_w, weights_2d)[1],
                method='L-BFGS-B',
                options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Second polish pass with even tighter tolerance
    if time.time() - start_time < deadline - 5:
        try:
            res = minimize(
                lambda c: _objective(c, Y, Y_w, weights_2d),
                best_coeffs,
                jac=lambda c: _objective_and_grad(c, Y, Y_w, weights_2d)[1],
                method='L-BFGS-B',
                options={'maxiter': 50000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Validate
    if len(best_coeffs) != 21 or not np.all(np.isfinite(best_coeffs)):
        return np.ones(21)
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        # Scale to valid range
        if max_abs < 1e-3:
            best_coeffs *= 1.0 / max_abs if max_abs > 0 else 1.0
        elif max_abs > 1e5:
            best_coeffs *= 1e5 / max_abs
        if not np.all(np.isfinite(best_coeffs)):
            return np.ones(21)
    
    return best_coeffs.astype(np.float64)
