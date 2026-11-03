import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100
_DEADLINE = 145.0  # Safety margin before 150s hard kill
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _project(X):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return X / norms


def _energy_and_grad(X):
    """
    Compute logarithmic energy and gradient for points on the sphere.
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient is the gradient w.r.t. the ambient R^3 coordinates.
    """
    n = X.shape[0]
    # Compute squared distances efficiently
    sq_norms = np.einsum('ij,ij->i', X, X)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (X @ X.T)
    
    # Clip to avoid log(0) or negative values due to floating point
    np.fill_diagonal(dist_sq, 0.0)
    dist_sq = np.maximum(dist_sq, 1e-30)
    
    # Energy: -0.5 * sum_{i!=j} log(dist_sq)
    # Since log(sqrt(d2)) = 0.5 log(d2), and we sum over i<j which is half of i!=j
    mask = ~np.eye(n, dtype=bool)
    log_dist_sq = np.zeros_like(dist_sq)
    np.log(dist_sq, out=log_dist_sq, where=mask)
    energy = -0.5 * np.sum(log_dist_sq)
    
    # Gradient:
    # E = -0.5 * sum_{i!=j} log(|x_i - x_j|^2)
    # dE/dx_i = -0.5 * sum_{j!=i} (1/|x_i - x_j|^2) * 2*(x_i - x_j)
    #          = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    #          = sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2
    #
    # Vectorized:
    # diff[i,j] = x_i - x_j
    # We want sum_j (x_j - x_i) / dist_sq[i,j] = - sum_j diff[i,j] / dist_sq[i,j]
    
    inv_dist_sq = np.zeros_like(dist_sq)
    np.divide(1.0, dist_sq, out=inv_dist_sq, where=mask)
    
    # diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    # grad[i] = - sum_j diff[i,j] * inv_dist_sq[i,j]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _tangent_project(grad, X):
    """
    Project the ambient gradient onto the tangent space of the sphere at each point.
    tangent_grad_i = grad_i - (grad_i . x_i) x_i
    """
    dot = np.einsum('ij,ij->i', grad, X)
    return grad - dot[:, np.newaxis] * X


def _optimize_lbfgs(X0, max_iter=1000):
    """
    Optimize using L-BFGS-B on the sphere using a penalty method or
    by working in ambient space and projecting.
    Here we use a simple penalty approach to keep points on the sphere,
    or better, we can use the fact that L-BFGS-B can handle bounds,
    but sphere constraint is non-linear.
    
    A robust way is to use the ambient gradient projected onto the tangent space
    and use a custom optimizer or a library that supports manifold optimization.
    Since we only have standard scipy, we will use a penalty method or
    a change of variables.
    
    Actually, a common trick is to optimize in R^3 and project back to the sphere
    at each step, but standard minimizers don't do this.
    
    Alternative: Use scipy.optimize.minimize with method='BFGS' or 'L-BFGS-B'
    and a penalty term for the sphere constraint.
    """
    def fun_and_grad(x_flat):
        X = x_flat.reshape(N, 3)
        E, G = _energy_and_grad(X)
        # Penalty for sphere constraint
        mu = 100.0
        norm_sq = np.sum(X * X, axis=1)
        P = mu * np.sum((norm_sq - 1.0) ** 2)
        G_P = 4.0 * mu * (norm_sq - 1.0)[:, np.newaxis] * X
        return E + P, (G + G_P).flatten()

    x0 = X0.flatten()
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
        X_opt = res.x.reshape(N, 3)
        X_opt = _project(X_opt)
        E_opt = _energy_and_grad(X_opt)[0]
        return X_opt, E_opt
    except Exception:
        return X0, _energy_and_grad(X0)[0]


def _optimize_steepest_descent(X0, steps=5000):
    """
    Steepest descent on the sphere using tangent projection.
    """
    X = X0.copy()
    best_E = _energy_and_grad(X)[0]
    best_X = X.copy()
    lr = 0.01
    
    for k in range(steps):
        if _remaining() < 5.0:
            break
        E, G = _energy_and_grad(X)
        G_tan = _tangent_project(G, X)
        
        # Adaptive step size
        # Simple backtracking line search
        alpha = lr
        for _ in range(20):
            X_new = X - alpha * G_tan
            X_new = _project(X_new)
            E_new = _energy_and_grad(X_new)[0]
            if E_new < E:
                break
            alpha *= 0.5
        else:
            break
            
        X = X_new
        if E_new < best_E:
            best_E = E_new
            best_X = X.copy()
            
        # Decay learning rate slowly
        if k % 100 == 0:
            lr = max(1e-4, lr * 0.99)
            
    return best_X, best_E


def _fibonacci_spiral():
    """Generate points using a Fibonacci spiral (golden ratio) distribution."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere():
    """Generate random points on the unit sphere."""
    v = np.random.randn(N, 3)
    return _project(v)


def _sobol_sphere():
    """Generate quasi-random points using Sobol sequence if available, else fallback."""
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=3, scramble=True)
        samples = sampler.random(N)
        # Map [0,1]^3 to sphere? This is tricky.
        # Better to use random and normalize, or use a specific mapping.
        # For now, fallback to random if mapping is not straightforward.
        return _random_sphere()
    except Exception:
        return _random_sphere()


def construct():
    start_time = time.time()
    
    # Initial candidate: Fibonacci spiral
    X0 = _fibonacci_spiral()
    best_X = X0.copy()
    best_E = _energy_and_grad(best_X)[0]
    
    # 1. Optimize from Fibonacci start using L-BFGS-B (with penalty)
    if _remaining() > 10.0:
        X_opt, E_opt = _optimize_lbfgs(X0, max_iter=200)
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt.copy()
    
    # 2. Refine with Steepest Descent (tangent space)
    if _remaining() > 5.0:
        X_ref, E_ref = _optimize_steepest_descent(best_X, steps=2000)
        if E_ref < best_E:
            best_E = E_ref
            best_X = X_ref.copy()
    
    # 3. Try random restarts if time permits
    restarts = 0
    while _remaining() > 20.0:
        X_rand = _random_sphere()
        # Quick optimization from random start
        X_opt, E_opt = _optimize_lbfgs(X_rand, max_iter=100)
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt.copy()
            # Refine the new best
            if _remaining() > 10.0:
                X_ref, E_ref = _optimize_steepest_descent(best_X, steps=1000)
                if E_ref < best_E:
                    best_E = E_ref
                    best_X = X_ref.copy()
        restarts += 1
        if restarts >= 5:
            break
            
    # Final polish with very small steps
    if _remaining() > 2.0:
        X_pol, E_pol = _optimize_steepest_descent(best_X, steps=500)
        if E_pol < best_E:
            best_E = E_pol
            best_X = X_pol.copy()
            
    # Ensure final points are on the sphere
    best_X = _project(best_X)
    
    return best_X
