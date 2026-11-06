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
    min_dist_sq = np.min(dist_sq)
    if min_dist_sq < 1e-12:
        return float('inf'), min_dist_sq
    energy = -0.25 * np.sum(np.log(dist_sq))
    return energy, min_dist_sq


def _fast_grad_tangent(pts):
    """Compute gradient of log energy projected to tangent space. Vectorized."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    mask = ~np.eye(n, dtype=bool)
    if np.any(dist_sq[mask] < 1e-12):
        return None
    
    inv_d2 = 1.0 / dist_sq
    inv_d2[~mask] = 0.0
    
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


def _fibonacci_sphere(n, seed=0):
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
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _optimize_from_start(pts, deadline, max_iters=3000):
    """Run vectorized gradient descent from initial points. Returns (pts, energy)."""
    pts = pts.copy()
    _, min_ds = _fast_energy(pts)
    if min_ds < 1e-12:
        pts += 0.01 * np.random.randn(*pts.shape)
        pts = _project_to_sphere(pts)
    
    lr = 0.01
    prev_energy = _fast_energy(pts)[0]
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        grad = _fast_grad_tangent(pts)
        if grad is None:
            pts += 0.01 * np.random.randn(*pts.shape)
            pts = _project_to_sphere(pts)
            continue
        
        # Try a few step sizes if current one fails
        accepted = False
        for attempt in range(5):
            pts_new = pts - lr * grad
            pts_new = _project_to_sphere(pts_new)
            
            new_energy, _ = _fast_energy(pts_new)
            
            if new_energy < prev_energy:
                pts = pts_new
                prev_energy = new_energy
                lr *= 0.999
                accepted = True
                break
            else:
                lr *= 0.5
        
        if not accepted and lr < 1e-10:
            break
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
    
    return pts, prev_energy


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin of 5 seconds
    
    # Start with Fibonacci sphere
    best_pts = _fibonacci_sphere(N)
    best_energy = _log_energy(best_pts)
    
    # Try multiple initializations and optimize
    n_restarts = 12
    max_iters = 3000
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Different initializations
        if restart == 0:
            pts = _fibonacci_sphere(N, seed=0)
        elif restart == 1:
            pts = _fibonacci_sphere(N, seed=1)
            # Perturb slightly
            pts += 0.001 * np.random.randn(*pts.shape)
            pts = _project_to_sphere(pts)
        else:
            pts = _random_sphere(N, seed=restart * 42 + 7)
        
        # Optimize using vectorized gradient descent
        optimized_pts, opt_energy = _optimize_from_start(pts, deadline, max_iters)
        
        # Check if this is better (use precise energy for comparison)
        final_energy = _log_energy(optimized_pts)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pts = optimized_pts.copy()
    
    # Final cleanup: ensure points are on unit sphere and valid
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    min_dist = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci sphere
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
