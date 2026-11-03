"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _energy_and_grad(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) and gradient.
    
    Energy = -sum_{i<j} log(|x_i - x_j|)
    Gradient for point k: sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of log(1/dist) = -sum over i<j of log(dist)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dist[iu])
    energy = -np.sum(log_dists)
    
    # Gradient: for each point k, grad_k = sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    # diff[k,j] = x_k - x_j, so gradient_k = sum_j diff[k,j] / dist_sq[k,j]
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = np.einsum('ij,ijk->ik', inv_dist_sq, diff)
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_start():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_start(seed):
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    return _project_to_sphere(pts)

def _optimize_from_start(pts0, deadline, max_iter=5000, tol=1e-14):
    """Run L-BFGS-B style optimization with sphere constraint via re-projection."""
    best_pts = pts0.copy()
    best_energy, _ = _energy_and_grad(best_pts)
    
    x0 = pts0.copy()
    
    def obj_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        x = _project_to_sphere(x)
        e, g = _energy_and_grad(x)
        # Project gradient back to tangent space
        # grad_tangent = g - (g . x) * x  (for unit sphere constraint)
        g_tan = g - np.sum(g * x, axis=1, keepdims=True) * x
        return e, g_tan.ravel()
    
    # Use L-BFGS-B with finite-difference-free gradient
    # We'll do manual iterations with a simple gradient descent + line search
    
    lr = 0.1
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        x = _project_to_sphere(x0)
        e, g = _energy_and_grad(x)
        g_tan = g - np.sum(g * x, axis=1, keepdims=True) * x
        
        # Check convergence
        g_norm = np.linalg.norm(g_tan)
        if g_norm < tol:
            break
        
        # Adaptive learning rate
        step = lr / (1.0 + 0.01 * it)
        x_new = x - step * g_tan
        x_new = _project_to_sphere(x_new)
        
        e_new, _ = _energy_and_grad(x_new)
        
        if e_new < e:
            x0 = x_new
            if e_new < best_energy:
                best_energy = e_new
                best_pts = x_new.copy()
            # Increase learning rate slightly if improving
            lr = min(lr * 1.05, 0.5)
        else:
            # Decrease learning rate
            lr *= 0.7
    
    return best_pts, best_energy

def _polish(pts, deadline, max_iter=10000, tol=1e-15):
    """High-precision polish using very small steps."""
    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(best_pts)
    x0 = pts.copy()
    lr = 0.01
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        x = _project_to_sphere(x0)
        e, g = _energy_and_grad(x)
        g_tan = g - np.sum(g * x, axis=1, keepdims=True) * x
        
        g_norm = np.linalg.norm(g_tan)
        if g_norm < tol:
            break
        
        step = lr / (1.0 + 0.005 * it)
        x_new = x - step * g_tan
        x_new = _project_to_sphere(x_new)
        
        e_new, _ = _energy_and_grad(x_new)
        
        if e_new < e - 1e-16:
            x0 = x_new
            if e_new < best_energy:
                best_energy = e_new
                best_pts = x_new.copy()
            lr = min(lr * 1.02, 0.1)
        else:
            lr *= 0.8
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Multi-start with different initializations
    starts = []
    
    # Fibonacci spiral
    starts.append(_fibonacci_start())
    
    # Random starts with different seeds
    for seed in [42, 123, 456, 789, 1011, 2022, 3033, 4044]:
        if time.time() > deadline - 30:
            break
        starts.append(_random_start(seed))
    
    # Also try a slightly perturbed Fibonacci
    if time.time() > deadline - 30:
        break
    rng = np.random.RandomState(999)
    fib = _fibonacci_start()
    perturbed = _project_to_sphere(fib + 0.1 * rng.randn(N, 3))
    starts.append(perturbed)
    
    best_pts = None
    best_energy = np.inf
    
    for i, s in enumerate(starts):
        if time.time() > deadline - 10:
            break
        
        pts, energy = _optimize_from_start(s, deadline - 15)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Final polish on the best result
    if best_pts is not None and time.time() < deadline - 5:
        best_pts, best_energy = _polish(best_pts, deadline - 2)
    
    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
