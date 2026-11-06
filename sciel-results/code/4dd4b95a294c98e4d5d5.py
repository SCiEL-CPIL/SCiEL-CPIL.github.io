"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/d_ij) using math.fsum for precision."""
    n = len(pts)
    terms = []
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            d = np.linalg.norm(pi - pts[j])
            if d < 1e-15:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _fast_energy(pts):
    """Fast vectorized energy for optimization loops. Returns (energy, min_dist_sq)."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist_sq = float(np.min(dist_sq))
    if min_dist_sq < 1e-12:
        return float('inf'), min_dist_sq
    energy = -0.25 * float(np.sum(np.log(dist_sq)))
    return energy, min_dist_sq


def _fast_grad_tangent(pts):
    """Compute gradient of log energy projected to tangent space. Vectorized.
    
    E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    dE/dx_i = -sum_{k != i} (x_i - x_k) / d_ik^2
    Project to tangent: grad_t = grad - (grad . x_i) * x_i
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    mask = ~np.eye(n, dtype=bool)
    if np.any(dist_sq[mask] < 1e-12):
        return None
    
    inv_d2 = np.zeros_like(dist_sq)
    valid = mask & (dist_sq > 1e-24)
    inv_d2[valid] = 1.0 / dist_sq[valid]
    
    weighted_diff = diff * inv_d2[:, :, np.newaxis]
    grad_raw = -np.sum(weighted_diff, axis=1)
    
    dot_products = np.sum(grad_raw * pts, axis=1, keepdims=True)
    grad_tangent = grad_raw - dot_products * pts
    
    return grad_tangent


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=0):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)


def _gradient_descent(pts, deadline, max_iters=5000, lr_init=0.01):
    """Run projected gradient descent with adaptive step size."""
    pts = pts.copy()
    energy, min_ds = _fast_energy(pts)
    if min_ds < 1e-12:
        pts = pts + 0.01 * np.random.randn(*pts.shape)
        pts = _project_to_sphere(pts)
        energy, _ = _fast_energy(pts)
    
    lr = lr_init
    prev_energy = energy
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        grad = _fast_grad_tangent(pts)
        if grad is None:
            pts = pts + 0.01 * np.random.randn(*pts.shape)
            pts = _project_to_sphere(pts)
            continue
        
        accepted = False
        for attempt in range(8):
            new_pts = pts - lr * grad
            new_pts = _project_to_sphere(new_pts)
            new_energy, min_d2 = _fast_energy(new_pts)
            
            if new_energy < prev_energy and min_d2 > 1e-10:
                pts = new_pts
                prev_energy = new_energy
                lr *= 0.999
                accepted = True
                break
            else:
                lr *= 0.5
        
        if not accepted and lr < 1e-12:
            break
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
    
    return pts, prev_energy


def _polish(pts, deadline):
    """Final polish with very small steps and high precision."""
    pts = pts.copy()
    energy, _ = _fast_energy(pts)
    lr = 0.001
    
    for it in range(2000):
        if time.time() > deadline:
            break
        
        grad = _fast_grad_tangent(pts)
        if grad is None:
            break
        
        new_pts = pts - lr * grad
        new_pts = _project_to_sphere(new_pts)
        new_energy, min_d2 = _fast_energy(new_pts)
        
        if new_energy < energy and min_d2 > 1e-10:
            pts = new_pts
            energy = new_energy
            lr *= 0.995
        else:
            lr *= 0.5
            if lr < 1e-15:
                break
        
        if np.linalg.norm(grad) < 1e-16:
            break
    
    return pts, energy


def construct():
    """Place N points on the unit sphere to minimize logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Start with Fibonacci sphere
    best_pts = _fibonacci_sphere(N)
    best_energy = _log_energy(best_pts)
    
    # Phase 1: Gradient descent from Fibonacci
    pts, energy = _gradient_descent(best_pts, deadline - 30.0, max_iters=5000, lr_init=0.01)
    if energy < best_energy:
        best_energy = energy
        best_pts = pts.copy()
    
    # Phase 2: Multiple random restarts with varied seeds
    np.random.seed(42)
    for attempt in range(15):
        if time.time() > deadline - 15.0:
            break
        
        # Vary the random seed for diversity
        rand_pts = _random_sphere(N, seed=attempt * 7919 + 13)
        pts, energy = _gradient_descent(rand_pts, deadline - 10.0, max_iters=3000, lr_init=0.005)
        
        # Use precise energy for comparison
        precise_energy = _log_energy(pts)
        if precise_energy < best_energy:
            best_energy = precise_energy
            best_pts = pts.copy()
    
    # Phase 3: Polish the best solution
    if time.time() < deadline - 5.0:
        best_pts, best_energy = _polish(best_pts, deadline - 2.0)
    
    # Final projection and validation
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close and all values are finite
    _, min_d2 = _fast_energy(best_pts)
    if min_d2 < 1e-18 or not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
