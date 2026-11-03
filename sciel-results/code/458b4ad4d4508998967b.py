"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize, LinearConstraint

# Precompute index pairs for energy/gradient
_i = np.triu_indices(N, k=1)
_pair_i = _i[0]
_pair_j = _i[1]
_n_pairs = len(_pair_i)


def _energy_and_grad(X):
    """Compute logarithmic energy and its gradient (in R^3 space, before projection)."""
    Xi = X[_pair_i]
    Xj = X[_pair_j]
    diff = Xi - Xj
    d2 = np.einsum('ij,ij->i', diff, diff)
    # Clip to avoid log(0)
    d2 = np.maximum(d2, 1e-30)
    d = np.sqrt(d2)
    # Energy: sum log(1/d) = -sum log(d)
    E = -np.sum(np.log(d))
    # Gradient: dE/dx_i = sum_{j!=i} (1/d) * (x_i - x_j) / d = sum (x_i - x_j)/d^2
    # Actually: dE/dx_i = sum_j d/dx_i [log(1/|x_i-x_j|)] = sum_j d/dx_i [-log|xi-xj|]
    # = sum_j [-(xi-xj)/|xi-xj|^2] = -sum_j (xi-xj)/d^2
    # Wait: E = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log|xi-xj|
    # dE/dxi = -sum_{j != i} (xi-xj)/|xi-xj|^2
    # So grad_xi = -sum_{j!=i} (xi-xj)/d_ij^2
    
    grad = np.zeros((N, 3))
    # For each pair (i,j), contribution to grad[i] is -(Xi-Xj)/d2 = (Xj-Xi)/d2
    # contribution to grad[j] is -(Xj-Xi)/d2 = (Xi-Xj)/d2
    np.add.at(grad, _pair_i, -diff / d2[:, None])
    np.add.at(grad, _pair_j, diff / d2[:, None])
    return E, grad


def _project_to_sphere(X):
    """Normalize rows to unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def _tangent_grad(X, grad):
    """Project gradient to tangent space of sphere at each point."""
    # Tangent projection: g - (g·x)x
    dot = np.einsum('ij,ij->i', X, grad)
    return grad - dot[:, None] * X


def _fibonacci_start():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_start(rng):
    X = rng.standard_normal((N, 3))
    return _project_to_sphere(X)


def _anisotropic_start(rng, scale):
    """Generate points with anisotropic noise for diversity."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    base = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    noise = rng.standard_normal((N, 3)) * scale
    X = base + noise
    return _project_to_sphere(X)


def _lbfgs_optimize(X0, max_iter=2000, tol=1e-12):
    """Run L-BFGS on the sphere using tangent-space gradient."""
    X0 = _project_to_sphere(X0.copy())
    
    def fun(X):
        Xn = _project_to_sphere(X)
        E, _ = _energy_and_grad(Xn)
        return E
    
    def jac(X):
        Xn = _project_to_sphere(X)
        _, grad = _energy_and_grad(Xn)
        return _tangent_grad(Xn, grad)
    
    res = minimize(fun, X0, jac=jac, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10, 'maxls': 50})
    return _project_to_sphere(res.x)


def _gradient_descent_polish(X, max_iter=5000, tol=1e-14):
    """Polish with adaptive-step gradient descent on the sphere."""
    X = _project_to_sphere(X.copy())
    E, _ = _energy_and_grad(X)
    best_E = E
    best_X = X.copy()
    
    lr = 0.01
    min_lr = 1e-8
    max_lr = 0.5
    no_improve = 0
    
    for it in range(max_iter):
        _, grad = _energy_and_grad(X)
        tg = _tangent_grad(X, grad)
        
        # Adaptive step: try to reduce energy
        X_new = X - lr * tg
        X_new = _project_to_sphere(X_new)
        E_new, _ = _energy_and_grad(X_new)
        
        if E_new < E:
            X = X_new
            E = E_new
            no_improve = 0
            # Increase lr slightly if improving
            if no_improve == 0 and lr < max_lr:
                lr = min(lr * 1.05, max_lr)
            if E < best_E:
                best_E = E
                best_X = X.copy()
        else:
            no_improve += 1
            lr = max(lr * 0.7, min_lr)
            if no_improve > 50:
                break
    
    return best_X


def _check_valid(X):
    """Check that points are valid: finite, on sphere, no close pairs."""
    if not np.all(np.isfinite(X)):
        return False
    norms = np.linalg.norm(X, axis=1)
    if not np.all(np.abs(norms - 1.0) < 1e-6):
        return False
    # Check minimum distance
    Xi = X[_pair_i]
    Xj = X[_pair_j]
    diff = Xi - Xj
    d2 = np.einsum('ij,ij->i', diff, diff)
    if np.min(d2) < (1e-9) ** 2:
        return False
    return True


def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = []
    starts.append(_fibonacci_start())
    # Random starts
    for _ in range(5):
        starts.append(_random_start(rng))
    # Anisotropic starts with different scales
    for scale in [0.1, 0.2, 0.3, 0.5]:
        starts.append(_anisotropic_start(rng, scale))
    # More random
    for _ in range(3):
        starts.append(_random_start(rng))
    
    best_X = None
    best_E = np.inf
    
    for idx, X0 in enumerate(starts):
        if time.time() > deadline - 10:
            break
        
        # L-BFGS optimization
        X_opt = _lbfgs_optimize(X0, max_iter=1500, tol=1e-12)
        
        # Gradient descent polish
        X_polished = _gradient_descent_polish(X_opt, max_iter=3000, tol=1e-14)
        
        E, _ = _energy_and_grad(_project_to_sphere(X_polished))
        
        if E < best_E:
            best_E = E
            best_X = _project_to_sphere(X_polished)
        
        # If time is running out, stop
        if time.time() > deadline - 20:
            break
    
    # Final high-precision polish on the best result
    if best_X is not None and time.time() < deadline - 5:
        # One more round of L-BFGS with tighter tolerance
        X_final = _lbfgs_optimize(best_X, max_iter=500, tol=1e-14)
        X_final = _gradient_descent_polish(X_final, max_iter=2000, tol=1e-15)
        E_final, _ = _energy_and_grad(_project_to_sphere(X_final))
        if E_final < best_E:
            best_E = E_final
            best_X = _project_to_sphere(X_final)
    
    if best_X is None:
        best_X = _fibonacci_start()
    
    return best_X
# EVOLVE-BLOCK-END
