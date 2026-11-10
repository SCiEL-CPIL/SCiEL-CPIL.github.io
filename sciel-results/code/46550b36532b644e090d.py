"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _energy_and_grad(x):
    """
    Compute logarithmic energy and Euclidean gradient for N points.
    Uses dot-product squared distances for speed.
    x: (N, 3) array of points (assumed unit norm).
    Returns (energy, grad) where grad is (N, 3).
    """
    # Compute squared distances via dot products: |x_i - x_j|^2 = 2 - 2*x_i.x_j
    X = x  # (N, 3)
    D = X @ X.T  # (N, N) dot products
    dist_sq = 2.0 - 2.0 * D  # (N, N)
    np.fill_diagonal(dist_sq, 1.0)  # avoid zero on diagonal
    
    # Clip to avoid negative values from floating point
    dist_sq = np.maximum(dist_sq, 1e-30)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of -log(dist[i,j])
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: grad[i] = -sum_{j != i} (x_i - x_j) / dist[i,j]^2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    S = np.sum(inv_dist_sq, axis=1)  # (N,)
    W = inv_dist_sq @ x  # (N, 3)
    grad = -x * S[:, np.newaxis] + W
    
    return energy, grad

def _project_and_normalize(x):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return x / norms

def _tangent_project(grad, x):
    """Project gradient to tangent space of sphere at each point."""
    return grad - (grad * x).sum(axis=1, keepdims=True) * x

def _fmin_on_sphere(x0, maxiter=5000, tol=1e-12):
    """
    Minimize energy on the sphere using L-BFGS-B with sphere projection.
    The key insight: we optimize in R^{3N} but project back to sphere after each step.
    We compute the Riemannian gradient (tangent projection) for the optimizer.
    """
    x = _project_and_normalize(x0.copy())
    
    def objective_and_grad(x_flat):
        x_2d = x_flat.reshape(N, 3)
        x_2d = _project_and_normalize(x_2d)
        e, g = _energy_and_grad(x_2d)
        # Riemannian gradient: project to tangent space
        g_tangent = _tangent_project(g, x_2d)
        return e, g_tangent.flatten()
    
    res = minimize(
        objective_and_grad,
        x.flatten(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': tol, 'gtol': 1e-10, 'maxls': 30}
    )
    
    x_opt = _project_and_normalize(res.x.reshape(N, 3))
    return x_opt

def _riemannian_gradient_descent(x0, maxiter=50000, lr=0.01, tol=1e-14):
    """
    Riemannian gradient descent with backtracking line search on the sphere.
    More robust for final polishing.
    """
    x = _project_and_normalize(x0.copy())
    e, g = _energy_and_grad(x)
    g_tan = _tangent_project(g, x)
    
    for it in range(maxiter):
        if time.time() > _deadline:
            break
        
        # Check convergence
        grad_norm = np.linalg.norm(g_tan)
        if grad_norm < tol:
            break
        
        # Gradient step
        step = lr * g_tan
        x_new = _project_and_normalize(x - step)
        e_new, g_new = _energy_and_grad(x_new)
        
        # Backtracking if energy increased
        if e_new > e:
            # Try smaller step
            for trial in range(10):
                step = (lr / (2.0 ** (trial + 1))) * g_tan
                x_new = _project_and_normalize(x - step)
                e_new, g_new = _energy_and_grad(x_new)
                if e_new < e:
                    break
            else:
                # If still not decreasing, reduce learning rate
                lr *= 0.5
                continue
        
        x = x_new
        e = e_new
        g = g_new
        g_tan = _tangent_project(g, x)
        
        # Adaptive learning rate
        if it % 100 == 0:
            # Simple momentum-like adaptation
            pass
    
    return _project_and_normalize(x)

def _energy(x):
    """Compute energy only."""
    e, _ = _energy_and_grad(x)
    return e

def _fibonacci_spiral(n=N, offset=0.5):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _perturbed_fibonacci(n=N, sigma=0.1, seed=0):
    """Fibonacci spiral with Gaussian perturbation."""
    rng = np.random.RandomState(seed)
    x = _fibonacci_spiral(n)
    x += rng.randn(n, 3) * sigma
    return _project_and_normalize(x)

def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 140.0  # Safety margin before 150s
    
    # Generate diverse initializations
    np.random.seed(42)
    
    # Fibonacci spiral
    x0 = _fibonacci_spiral(N)
    
    # Perturbed Fibonacci variants
    perturbations = []
    for sigma in [0.05, 0.1, 0.2, 0.3, 0.5]:
        for seed in [0, 1, 2]:
            perturbations.append(_perturbed_fibonacci(N, sigma, seed))
    
    # Random initializations
    for seed in range(10):
        rng = np.random.RandomState(100 + seed)
        x_rand = rng.randn(N, 3)
        x_rand /= np.linalg.norm(x_rand, axis=1, keepdims=True)
        perturbations.append(x_rand)
    
    # Also include the original x0
    all_inits = [x0] + perturbations
    
    best_x = x0.copy()
    best_e = _energy(best_x)
    
    # Optimize each initialization
    for idx, x_init in enumerate(all_inits):
        if time.time() > _deadline:
            break
        
        remaining_time = _deadline - time.time()
        # Allocate iterations based on remaining time
        maxiter = min(5000, max(100, int(remaining_time * 50)))
        
        try:
            x_opt = _fmin_on_sphere(x_init, maxiter=maxiter, tol=1e-10)
            e_opt = _energy(x_opt)
            
            if e_opt < best_e:
                best_e = e_opt
                best_x = x_opt.copy()
        except Exception:
            continue
    
    # Final high-precision polish with Riemannian GD
    if time.time() < _deadline - 5.0:
        try:
            x_polished = _riemannian_gradient_descent(best_x, maxiter=100000, lr=0.005, tol=1e-15)
            e_polished = _energy(x_polished)
            if e_polished < best_e:
                best_x = x_polished
                best_e = e_polished
        except Exception:
            pass
    
    # Another L-BFGS polish with very tight tolerance
    if time.time() < _deadline - 2.0:
        try:
            x_final = _fmin_on_sphere(best_x, maxiter=10000, tol=1e-15)
            e_final = _energy(x_final)
            if e_final < best_e:
                best_x = x_final
        except Exception:
            pass
    
    # Ensure points are on unit sphere and valid
    best_x = _project_and_normalize(best_x)
    
    # Check for very close points
    D = best_x @ best_x.T
    dist_sq = 2.0 - 2.0 * D
    np.fill_diagonal(dist_sq, 1.0)
    min_dist_sq = np.min(dist_sq)
    
    if min_dist_sq < 1e-18:  # dist < 1e-9
        # Fall back to Fibonacci spiral
        best_x = _fibonacci_spiral(N)
    
    return best_x
# EVOLVE-BLOCK-END
