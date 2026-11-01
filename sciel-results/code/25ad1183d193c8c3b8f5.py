"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time


def _energy(pts):
    # Calculate energy sum_{i<j} log(1/d_ij)
    # Vectorized
    dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(N, 1)
    d = dists[iu]
    return -np.sum(np.log(d))


def _grad(pts):
    # Gradient of energy
    # dE/dx_k = sum_{j != k} (x_k - x_j) / ||x_k - x_j||^2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    inv_dists_sq = 1.0 / dists_sq
    np.fill_diagonal(inv_dists_sq, 0.0)
    grad = np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
    return grad


def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _optimize(pts, max_time=5.0):
    pts = pts.copy()
    pts = _project_to_sphere(pts)
    
    current_energy = _energy(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    start_time = time.time()
    
    step = 0.01
    velocity = np.zeros_like(pts)
    mu = 0.8
    
    while time.time() - start_time < max_time:
        g = _grad(pts)
        
        # Project gradient to tangent plane
        dot = np.sum(g * pts, axis=1, keepdims=True)
        g_tangent = g - dot * pts
        
        # Update with momentum
        velocity = mu * velocity + step * g_tangent
        pts_new = pts - velocity
        
        # Project to sphere
        pts_new = _project_to_sphere(pts_new)
        
        # Check energy
        new_energy = _energy(pts_new)
        
        if new_energy < current_energy:
            pts = pts_new
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
            step *= 1.1
            if step > 0.05:
                step = 0.05
        else:
            step *= 0.5
            velocity = 0.0
            
    return best_pts, best_energy


def construct():
    deadline = 130.0
    start_time = time.time()
    
    best_pts = None
    best_energy = float('inf')
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    candidates = [pts_fib]
    
    # 2. Random starts
    rng = np.random.default_rng(12345)
    for _ in range(15):
        pts_rand = rng.random((N, 3))
        pts_rand = _project_to_sphere(pts_rand)
        candidates.append(pts_rand)
        
    # 3. Slightly perturbed Fibonacci
    for _ in range(5):
        pts_pert = pts_fib + 0.1 * rng.random((N, 3))
        pts_pert = _project_to_sphere(pts_pert)
        candidates.append(pts_pert)

    # Optimize each candidate
    # We have ~21 candidates.
    # Budget: 130s.
    # Let's give each 4 seconds. 21 * 4 = 84s. Safe.
    
    for idx, init_pts in enumerate(candidates):
        if time.time() - start_time > deadline:
            break
            
        remaining = deadline - (time.time() - start_time)
        alloc_time = min(4.0, remaining)
        
        if alloc_time < 0.5:
            continue
            
        opt_pts, opt_energy = _optimize(init_pts, max_time=alloc_time)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_pts = opt_pts
            
    if best_pts is None:
        best_pts = pts_fib
    else:
        best_pts = _project_to_sphere(best_pts)
        
    return best_pts
# EVOLVE-BLOCK-END
