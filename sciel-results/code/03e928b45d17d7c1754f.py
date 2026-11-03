"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _energy_and_tangent_grad(pts):
    """Compute logarithmic energy and its tangent-space gradient.
    
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient on sphere: for each point i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    Then project to tangent space: grad_i_tan = grad_i - (grad_i . x_i) * x_i
    """
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(dist, 1.0)
    
    # Energy: -sum_{i<j} log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each i, sum over j != i of (x_i - x_j) / dist_ij^2
    # dist^2 matrix
    dist2 = dist * dist
    np.fill_diagonal(dist2, 1.0)  # avoid div by zero
    
    # grad_i = sum_j (x_i - x_j) / dist_ij^2
    # diff[i,j] = x_i - x_j
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad[i] = sum_j diff[i,j] * inv_dist2[i,j]
    grad = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    
    # Project to tangent space: grad_tan = grad - (grad . x_i) x_i
    dot = np.sum(grad * pts, axis=1)
    grad_tan = grad - dot[:, np.newaxis] * pts
    
    return energy, grad_tan

def _exp_map(pts, tangent_vecs, step):
    """Exponential map on the unit sphere.
    
    For a point x on the sphere and tangent vector v, 
    exp_x(step * v) = x * cos(step*|v|) + (v/|v|) * sin(step*|v|)
    """
    v = step * tangent_vecs
    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    
    # Handle zero vectors
    mask = v_norm > 1e-15
    v_norm_safe = np.where(mask, v_norm, 1.0)
    
    cos_t = np.cos(v_norm_safe)
    sin_t = np.sin(v_norm_safe) / v_norm_safe
    sin_t = np.where(mask, sin_t, 0.0)
    
    new_pts = pts * cos_t + v * sin_t
    
    # Renormalize to ensure unit length
    new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
    return new_pts

def _fibonacci_spiral(n, offset=0.5, twist_factor=1.0):
    """Generate points on sphere using golden angle spiral with optional offset."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * twist_factor * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.normal(size=(n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _perturb(pts, scale, rng):
    """Add Gaussian perturbation in tangent space and re-extract."""
    noise = rng.normal(scale=scale, size=pts.shape)
    # Project noise to tangent space
    dot = np.sum(pts * noise, axis=1, keepdims=True)
    noise_tan = noise - dot * pts
    new_pts = pts + noise_tan
    new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
    return new_pts

def _riemannian_gd(pts, max_iter=500, lr=0.01, momentum=0.9, tol=1e-10):
    """Riemannian gradient descent with momentum on the sphere."""
    n = len(pts)
    v = np.zeros_like(pts)  # momentum velocity
    best_energy = np.inf
    best_pts = pts.copy()
    
    for it in range(max_iter):
        energy, grad_tan = _energy_and_tangent_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Momentum update in tangent space
        v = momentum * v - lr * grad_tan
        
        # Project velocity to tangent space
        dot_v = np.sum(v * pts, axis=1, keepdims=True)
        v_tan = v - dot_v * pts
        
        # Adaptive step size: normalize velocity
        v_norm = np.linalg.norm(v_tan, axis=1, keepdims=True)
        max_v = np.max(v_norm)
        if max_v > 0:
            v_tan = v_tan / max_v * 0.1  # cap step size
        
        pts = _exp_map(pts, v_tan, 1.0)
        
        # Check convergence
        if np.linalg.norm(grad_tan) < tol:
            break
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_energy = np.inf
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Standard Fibonacci spiral
    initializations.append(_fibonacci_spiral(N))
    
    # 2. Fibonacci with different offsets
    for offset in [0.0, 0.25, 0.75]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # 3. Fibonacci with different twist factors
    for twist in [0.5, 1.5, 2.0]:
        initializations.append(_fibonacci_spiral(N, twist_factor=twist))
    
    # 4. Random initializations
    for _ in range(5):
        initializations.append(_random_sphere(N, rng))
    
    # 5. Perturbed Fibonacci
    for scale in [0.01, 0.05, 0.1]:
        base = _fibonacci_spiral(N)
        initializations.append(_perturb(base, scale, rng))
    
    # Optimize each initialization
    for init_idx, init_pts in enumerate(initializations):
        # Check deadline
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # First pass: coarser optimization
        pts_coarse, energy_coarse = _riemannian_gd(
            init_pts, max_iter=300, lr=0.05, momentum=0.9, tol=1e-8
        )
        
        if energy_coarse < best_energy:
            best_energy = energy_coarse
            best_pts = pts_coarse.copy()
        
        # Check deadline
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Second pass: fine optimization from the best result
        pts_fine, energy_fine = _riemannian_gd(
            pts_coarse, max_iter=500, lr=0.01, momentum=0.9, tol=1e-10
        )
        
        if energy_fine < best_energy:
            best_energy = energy_fine
            best_pts = pts_fine.copy()
    
    # If we haven't found anything, use the initial Fibonacci
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    # Final polish: one more fine-tuning pass if time allows
    if time.time() - start_time < DEADLINE - 5:
        pts_polish, energy_polish = _riemannian_gd(
            best_pts, max_iter=300, lr=0.005, momentum=0.9, tol=1e-12
        )
        if energy_polish < best_energy:
            best_pts = pts_polish
    
    # Ensure all points are on unit sphere and finite
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
    
    return best_pts
# EVOLVE-BLOCK-END
