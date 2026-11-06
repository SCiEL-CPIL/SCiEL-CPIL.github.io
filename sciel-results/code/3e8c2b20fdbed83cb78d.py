import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_score = 0.0
    best_centers = None
    
    # Initial grid
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial_centers = grid[:N].copy()
    
    def evaluate(centers):
        # Compute min distance to other centers
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Compute min distance to walls
        dists_wall = 0.5 - np.abs(centers)
        min_wall_dist = np.min(dists_wall)
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        return r, min_pair_dist, min_wall_dist

    # Initial score
    score, _, _ = evaluate(initial_centers)
    best_score = score
    best_centers = initial_centers.copy()
    
    def compute_energy_and_grad(centers):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists_sq = np.maximum(dists_sq, 1e-12)
        
        inv_dists_sq = 1.0 / dists_sq
        np.fill_diagonal(inv_dists_sq, 0.0)
        
        inv_dists_4 = inv_dists_sq**2
        
        weighted_c = np.sum(centers[np.newaxis, :, :] * inv_dists_4[:, :, np.newaxis], axis=1)
        weighted_sum = np.sum(inv_dists_4, axis=1)
        
        grad_pair = -2.0 * (centers * weighted_sum[:, np.newaxis] - weighted_c)
        
        d_wall = 0.5 - np.abs(centers)
        d_wall = np.maximum(d_wall, 1e-6)
        inv_d_wall_sq = 1.0 / (d_wall**2)
        
        inv_d_wall_3 = inv_d_wall_sq / d_wall
        grad_wall = 2.0 * np.sign(centers) * inv_d_wall_3
        
        total_grad = grad_pair + grad_wall
        energy = np.sum(inv_dists_sq) + np.sum(inv_d_wall_sq)
        
        return energy, total_grad

    # Main optimization loop
    step_size = 1e-4
    centers = initial_centers.copy()
    
    for step in range(10000):
        if time.time() > deadline:
            break
            
        energy, grad = compute_energy_and_grad(centers)
        
        # Simple adaptive step: if gradient norm is large, reduce step
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            effective_step = step_size / (1.0 + grad_norm * 0.1)
        else:
            effective_step = step_size
            
        centers = centers - effective_step * grad
        centers = np.clip(centers, -0.5, 0.5)
        
        score, _, _ = evaluate(centers)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            
        step_size *= 0.9995

    # Restarts
    while time.time() < deadline - 10:
        noise = np.random.normal(0, 0.005, size=(N, 3))
        centers = best_centers + noise
        centers = np.clip(centers, -0.5, 0.5)
        
        step_size = 1e-5
        for step in range(1000):
            if time.time() > deadline - 5:
                break
            energy, grad = compute_energy_and_grad(centers)
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                effective_step = step_size / (1.0 + grad_norm * 0.1)
            else:
                effective_step = step_size
                
            centers = centers - effective_step * grad
            centers = np.clip(centers, -0.5, 0.5)
            
            score, _, _ = evaluate(centers)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                
            step_size *= 0.998

    return best_centers
