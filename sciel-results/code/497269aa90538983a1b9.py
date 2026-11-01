import numpy as np
import time
import math

N = 100

def _energy_and_grad(points):
    # points: (N, 3)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    # Avoid division by zero
    dist2[dist2 < 1e-12] = 1e-12
    dist = np.sqrt(dist2)
    
    # Energy
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient
    mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(mask, False)
    
    inv_dist2 = np.zeros_like(dist2)
    np.divide(1.0, dist2, out=inv_dist2, where=mask)
    
    # weighted_diff[i, j, k] = diff[i, j, k] * inv_dist2[i, j]
    weighted_diff = diff * inv_dist2[:, :, np.newaxis]
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad

def _riemannian_step(points, grad, alpha):
    # Riemannian gradient
    dots = np.sum(grad * points, axis=1, keepdims=True)
    rgrad = grad - dots * points
    
    # Step
    new_points = points - alpha * rgrad
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    # Handle zero norms (should not happen)
    norms[norms < 1e-12] = 1e-12
    new_points = new_points / norms
    
    return new_points

def _optimize_single(points, time_limit=10.0):
    start = time.time()
    alpha = 0.01
    best_energy, _ = _energy_and_grad(points)
    best_points = points.copy()
    stagnant_count = 0
    
    while time.time() - start < time_limit:
        energy, grad = _energy_and_grad(points)
        
        # Backtracking line search
        alpha = 0.01
        improved = False
        while True:
            new_points = _riemannian_step(points, grad, alpha)
            new_energy, _ = _energy_and_grad(new_points)
            
            if new_energy < energy - 1e-10: # Sufficient decrease
                points = new_points
                improved = True
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_points = points.copy()
                    stagnant_count = 0
                break
            else:
                alpha *= 0.5
                if alpha < 1e-8:
                    break
        
        if not improved:
            stagnant_count += 1
            if stagnant_count > 50:
                # Perturb
                noise = np.random.normal(0, 0.01, points.shape)
                points = points + noise
                norms = np.linalg.norm(points, axis=1, keepdims=True)
                points = points / norms
                stagnant_count = 0
        else:
            stagnant_count = 0
            
    return best_points, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    configs = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    configs.append(fib_points)
    
    # 2. Random points
    rng = np.random.default_rng(42)
    for _ in range(20): # More restarts
        pts = rng.normal(size=(N, 3))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        configs.append(pts / norms)
        
    best_energy = np.inf
    best_points = fib_points
    
    for idx, cfg in enumerate(configs):
        if time.time() > deadline - 5.0: # Leave 5s for final check
            break
            
        # Allocate remaining time to this restart
        time_left = deadline - time.time()
        # Give a fixed chunk, e.g., 5 seconds per restart
        alloc_time = min(5.0, time_left)
        
        opt_points, opt_energy = _optimize_single(cfg, time_limit=alloc_time)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_points = opt_points
            
    return best_points
