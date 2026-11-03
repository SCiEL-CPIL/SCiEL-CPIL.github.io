"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    
    Args:
        pts: (N, 3) array of points on unit sphere
    
    Returns:
        energy: scalar
        grad: (N, 3) array
    """
    n = pts.shape[0]
    # Compute pairwise distances
    # diff[i,j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)  # avoid division by zero on diagonal
    
    dist = np.sqrt(dist2)
    
    # Energy: sum_{i<j} log(1/dist_ij) = -sum_{i<j} log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dist[iu])
    energy = -math.fsum(log_dists.tolist())
    
    # Gradient: grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j, so grad_i = sum_j diff[i,j] / dist2[i,j]
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    grad = np.sum(diff * inv_dist2[:, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _gradient_descent_with_momentum(pts, max_iter=50000, deadline=None):
    """Gradient descent with momentum and Armijo line search on sphere."""
    pts = _project_to_sphere(pts.copy())
    energy, grad = _energy_and_grad(pts)
    best_energy = energy
    best_pts = pts.copy()
    
    momentum = np.zeros_like(pts)
    lr = 0.01
    mu = 0.9  # momentum coefficient
    armijo_c = 1e-4
    armijo_rho = 0.5
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update momentum
        momentum = mu * momentum - lr * grad
        new_pts = _project_to_sphere(pts + momentum)
        
        # Armijo line search
        new_energy, new_grad = _energy_and_grad(new_pts)
        step = 1.0
        while new_energy > energy - armijo_c * step * np.dot(grad.flatten(), momentum.flatten()):
            step *= armijo_rho
            if step < 1e-12:
                break
            new_pts = _project_to_sphere(pts + step * momentum)
            new_energy, new_grad = _energy_and_grad(new_pts)
        
        if step < 1e-12:
            # Reduce learning rate
            lr *= 0.5
            if lr < 1e-12:
                break
            continue
        
        pts = new_pts
        energy = new_energy
        grad = new_grad
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Check convergence
        if np.linalg.norm(grad, ord=np.inf) < 1e-14:
            break
    
    return best_pts, best_energy


def _scipy_optimize(pts, deadline=None):
    """Use scipy.optimize.minimize with L-BFGS-B on sphere-constrained problem.
    We use the fact that for points on sphere, we can optimize in R^3 and project."""
    try:
        from scipy.optimize import minimize as sp_minimize
    except ImportError:
        return pts, _energy_and_grad(pts)[0]
    
    def objective_and_grad(x_flat):
        pts = x_flat.reshape(N, 3)
        pts = _project_to_sphere(pts)
        e, g = _energy_and_grad(pts)
        return e, g.flatten()
    
    def objective(x_flat):
        return objective_and_grad(x_flat)[0]
    
    def grad(x_flat):
        return objective_and_grad(x_flat)[1]
    
    x0 = pts.flatten()
    
    # Use L-BFGS-B with tight tolerances
    result = sp_minimize(
        objective,
        x0,
        jac=grad,
        method='L-BFGS-B',
        options={
            'maxiter': 10000,
            'ftol': 1e-18,
            'gtol': 1e-14,
            'maxls': 50,
            'maxfun': 50000,
            'disp': False,
        }
    )
    
    if result.success or result.fun < _energy_and_grad(pts)[0]:
        return _project_to_sphere(result.x.reshape(N, 3)), result.fun
    return pts, _energy_and_grad(pts)[0]


def _polish(pts, deadline=None):
    """Final high-precision polish pass with very small steps."""
    pts = _project_to_sphere(pts.copy())
    energy, grad = _energy_and_grad(pts)
    best_energy = energy
    best_pts = pts.copy()
    
    lr = 1e-6
    mu = 0.9
    momentum = np.zeros_like(pts)
    armijo_c = 1e-4
    armijo_rho = 0.5
    
    for it in range(10000):
        if deadline is not None and time.time() > deadline:
            break
        
        momentum = mu * momentum - lr * grad
        new_pts = _project_to_sphere(pts + momentum)
        new_energy, new_grad = _energy_and_grad(new_pts)
        
        step = 1.0
        while new_energy > energy - armijo_c * step * np.dot(grad.flatten(), momentum.flatten()):
            step *= armijo_rho
            if step < 1e-14:
                break
            new_pts = _project_to_sphere(pts + step * momentum)
            new_energy, new_grad = _energy_and_grad(new_pts)
        
        if step < 1e-14:
            lr *= 0.5
            if lr < 1e-14:
                break
            continue
        
        pts = new_pts
        energy = new_energy
        grad = new_grad
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        if np.linalg.norm(grad, ord=np.inf) < 1e-16:
            break
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    # Safety margin: stop 10s before the 120s limit
    deadline = start_time + 110.0
    
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral()
    
    # Phase 1: Gradient descent with momentum
    pts, e1 = _gradient_descent_with_momentum(pts, max_iter=30000, deadline=deadline)
    
    if time.time() > deadline - 5.0:
        return _project_to_sphere(pts)
    
    # Phase 2: SciPy L-BFGS-B
    pts, e2 = _scipy_optimize(pts, deadline=deadline)
    
    if time.time() > deadline - 3.0:
        return _project_to_sphere(pts)
    
    # Phase 3: Final polish
    pts, e3 = _polish(pts, deadline=deadline)
    
    return _project_to_sphere(pts)

# EVOLVE-BLOCK-END
