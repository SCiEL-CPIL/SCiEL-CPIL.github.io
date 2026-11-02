"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize, LinearConstraint

N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _log_energy_and_grad(X):
    """
    Compute the logarithmic energy and its gradient for points on the sphere.
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i: sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    # X shape (N, 3)
    # Compute pairwise distances
    # d_ij = |x_i - x_j|
    # diff[i,j] = x_i - x_j
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero
    dist = np.maximum(dist, 1e-15)
    
    # Energy: -sum_{i<j} log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each i, grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j
    # grad_i = sum_j (x_i - x_j) / dist_ij^2
    # Note: we need to exclude j=i where dist=0
    inv_dist_sq = np.zeros_like(dist)
    np.fill_diagonal(inv_dist_sq, 0.0)
    inv_dist_sq = 1.0 / (dist * dist)
    inv_dist_sq = np.where(dist > 1e-12, inv_dist_sq, 0.0)
    
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(X):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def _golden_spiral():
    """Generate golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(seed):
    """Generate random points on the unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def construct():
    """
    Minimize the logarithmic energy of 150 points on the unit sphere.
    Returns an array of shape (N, 3) with points on the unit sphere.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Start with golden spiral
    X = _golden_spiral()
    best_X = X.copy()
    best_energy = _log_energy_and_grad(best_X)[0]
    
    # Use L-BFGS-B with projection, but we need to handle the sphere constraint.
    # A common approach: optimize in R^3 and project after each step, 
    # or use a parameterization. Here we'll use the approach of optimizing
    # with a penalty or using the fact that the gradient on the sphere is
    # the projection of the Euclidean gradient.
    
    # Actually, let's use a different approach: use scipy's minimize with 
    # bounds or a custom method. Since we're on the sphere, we can use
    # the gradient projected onto the tangent space.
    
    # Let's implement a simple gradient descent with projection, using
    # adaptive step sizes and momentum.
    
    # Alternative: use scipy.optimize.minimize with a custom objective
    # that includes a penalty for leaving the sphere, or use the 
    # fact that we can reparameterize.
    
    # Let's try L-BFGS-B with a large penalty for norm deviation.
    # Or better: use the projected gradient method.
    
    # Actually, the cleanest approach is to use scipy's minimize with 
    # the 'L-BFGS-B' method on the unconstrained problem but with a 
    # very strong penalty, or to use a manifold optimization approach.
    
    # Let's use a simple but effective approach: 
    # 1. Start with golden spiral
    # 2. Use projected gradient descent with adaptive step size
    # 3. Polish with L-BFGS using a penalty method
    
    def energy_with_penalty(X, penalty_weight):
        """Energy with a penalty for deviating from the unit sphere."""
        E, grad = _log_energy_and_grad(X)
        norms = np.linalg.norm(X, axis=1)
        penalty = penalty_weight * np.sum((norms - 1.0)**2)
        grad_penalty = -2.0 * penalty_weight * (norms - 1.0)[:, np.newaxis] * X / norms[:, np.newaxis]
        return E + penalty, grad + grad_penalty
    
    # Use L-BFGS-B with a large penalty
    # We'll do multiple rounds with decreasing penalty
    for penalty in [1e4, 1e5, 1e6, 1e7]:
        if time.time() > deadline:
            break
        
        def obj(x):
            X = x.reshape(N, 3)
            val, _ = energy_with_penalty(X, penalty)
            return val
        
        def grad_obj(x):
            X = x.reshape(N, 3)
            _, g = energy_with_penalty(X, penalty)
            return g.flatten()
        
        x0 = X.flatten()
        res = minimize(obj, x0, jac=grad_obj, method='L-BFGS-B',
                       options={'maxiter': 1000, 'gtol': 1e-10, 'ftol': 1e-12})
        
        X = res.x.reshape(N, 3)
        X = _project_to_sphere(X)
        
        E, _ = _log_energy_and_grad(X)
        if E < best_energy:
            best_energy = E
            best_X = X.copy()
    
    # Final polish: use projected gradient descent with very small steps
    if time.time() < deadline - 5.0:
        X = best_X.copy()
        E, _ = _log_energy_and_grad(X)
        best_E = E
        
        # Adaptive step size projected gradient descent
        lr = 0.01
        momentum = 0.9
        v = np.zeros_like(X)
        
        for it in range(5000):
            if time.time() > deadline:
                break
            
            E, grad = _log_energy_and_grad(X)
            
            # Project gradient onto tangent space
            grad_tan = grad - np.sum(grad * X, axis=1, keepdims=True) * X
            
            v = momentum * v + (1 - momentum) * grad_tan
            X_new = X - lr * v
            X_new = _project_to_sphere(X_new)
            
            E_new, _ = _log_energy_and_grad(X_new)
            
            if E_new < E:
                X = X_new
                if E_new < best_E:
                    best_E = E_new
                    best_X = X.copy()
                # Decrease learning rate slowly
                lr *= 0.999
            else:
                # Reduce learning rate
                lr *= 0.5
                v = 0.0
        
        # If we improved, update
        if best_E < best_energy:
            best_energy = best_E
            best_X = X.copy()
    
    # Final high-precision polish with very tight tolerance
    if time.time() < deadline - 2.0:
        X = best_X.copy()
        
        def obj_final(x):
            Xf = x.reshape(N, 3)
            val, _ = _log_energy_and_grad(Xf)
            return val
        
        def grad_final(x):
            Xf = x.reshape(N, 3)
            _, g = _log_energy_and_grad(Xf)
            return g.flatten()
        
        res = minimize(obj_final, X.flatten(), jac=grad_final, method='L-BFGS-B',
                       options={'maxiter': 2000, 'gtol': 1e-12, 'ftol': 1e-14})
        
        X = res.x.reshape(N, 3)
        X = _project_to_sphere(X)
        
        E, _ = _log_energy_and_grad(X)
        if E < best_energy:
            best_energy = E
            best_X = X.copy()
    
    # Ensure all points are on the unit sphere
    best_X = _project_to_sphere(best_X)
    
    # Validate
    norms = np.linalg.norm(best_X, axis=1)
    if not np.all(np.isfinite(best_X)) or np.any(norms < 1e-15):
        # Fallback to golden spiral
        best_X = _golden_spiral()
    
    return best_X
# EVOLVE-BLOCK-END
