"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 200

# EVOLVE-BLOCK-START
def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)

def _hexagonal_init():
    """Generate hexagonal lattice points and select N closest to center."""
    i_max = 20
    j_max = 20
    points = []
    for j in range(-j_max, j_max + 1):
        for i in range(-i_max, i_max + 1):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2.0
            points.append((x, y))
    points = np.array(points)
    dists_to_center = np.sum(points**2, axis=1)
    idx = np.argsort(dists_to_center)[:N]
    selected = points[idx]
    max_extent = max(np.max(np.abs(selected[:, 0])), np.max(np.abs(selected[:, 1])))
    s = 0.5 / (max_extent + 0.5)
    return selected * s

def _force_relaxation(centers, start_time, timeout):
    """Force-based relaxation to maximize minimum distance."""
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    current_r = best_r
    
    # Adaptive step size
    step = 0.01
    stall = 0
    
    while time.time() - start_time < timeout:
        # Compute all pairwise distances and forces
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        np.fill_diagonal(dists, np.inf)
        np.fill_diagonal(dists_sq, np.inf)
        
        min_pair = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        
        wall_dists = np.min(0.5 - np.abs(centers), axis=1)
        min_wall = np.min(wall_dists)
        min_wall_idx = np.argmin(wall_dists)
        
        current_r = min(min_pair / 2.0, min_wall)
        
        if current_r > best_r:
            best_r = current_r
            best_centers = centers.copy()
            stall = 0
        else:
            stall += 1
        
        if stall > 200:
            break
        
        # Compute forces: repulsion from close pairs + attraction to center
        # Force from each pair: push apart if dist < threshold
        threshold = min_pair * 1.5
        forces = np.zeros_like(centers)
        
        # Only process pairs within threshold for efficiency
        mask = (dists < threshold) & (dists > 1e-10)
        for i in range(N):
            for j in range(i + 1, N):
                if mask[i, j]:
                    d = dists[i, j]
                    f = 1.0 / (d * d)  # Inverse square repulsion
                    v = diff[i, j] / d
                    forces[i] += f * v
                    forces[j] -= f * v
        
        # Wall forces: push away from walls
        for i in range(N):
            x, y = centers[i]
            # Distance to each wall
            dx_l = x + 0.5  # left
            dx_r = 0.5 - x  # right
            dy_b = y + 0.5  # bottom
            dy_t = 0.5 - y  # top
            
            # Force from walls (inverse square)
            if dx_l < 0.1:
                forces[i, 0] += 1.0 / (dx_l * dx_l + 1e-10)
            if dx_r < 0.1:
                forces[i, 0] -= 1.0 / (dx_r * dx_r + 1e-10)
            if dy_b < 0.1:
                forces[i, 1] += 1.0 / (dy_b * dy_b + 1e-10)
            if dy_t < 0.1:
                forces[i, 1] -= 1.0 / (dy_t * dy_t + 1e-10)
        
        # Apply forces with adaptive step
        # Normalize forces to prevent explosion
        force_norms = np.linalg.norm(forces, axis=1)
        max_fn = np.max(force_norms)
        if max_fn > 1e-10:
            forces = forces / max_fn
        
        centers += step * forces
        centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)
        
        # Adapt step size
        new_r = _compute_radius(centers)
        if new_r > current_r:
            step = min(step * 1.1, 0.05)
        else:
            step *= 0.9
            if step < 1e-5:
                break
    
    return best_centers, best_r

def _polish(centers, start_time, timeout):
    """High-precision polish: iteratively fix the most violated constraint."""
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    step = 1e-5
    stall = 0
    
    while time.time() - start_time < timeout:
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        
        wall_dists = np.min(0.5 - np.abs(centers), axis=1)
        min_wall = np.min(wall_dists)
        min_wall_idx = np.argmin(wall_dists)
        
        current_r = min(min_pair / 2.0, min_wall)
        
        if current_r > best_r:
            best_r = current_r
            best_centers = centers.copy()
            stall = 0
        else:
            stall += 1
        
        if stall > 500:
            break
        
        if min_pair / 2.0 < min_wall:
            # Fix the closest pair
            i, j = min_pair_idx
            v = centers[j] - centers[i]
            norm = np.linalg.norm(v)
            if norm < 1e-12:
                v = np.array([1.0, 0.0])
            else:
                v = v / norm
            
            old_i = centers[i].copy()
            old_j = centers[j].copy()
            
            centers[i] -= step * v
            centers[j] += step * v
            centers[i] = np.clip(centers[i], -0.5, 0.5)
            centers[j] = np.clip(centers[j], -0.5, 0.5)
            
            new_r = _compute_radius(centers)
            if new_r <= current_r:
                centers[i] = old_i
                centers[j] = old_j
        else:
            # Fix the closest wall
            i = min_wall_idx
            x, y = centers[i]
            if abs(x) >= abs(y):
                v = np.array([-np.sign(x), 0.0]) if abs(x) > 1e-12 else np.array([1.0, 0.0])
            else:
                v = np.array([0.0, -np.sign(y)]) if abs(y) > 1e-12 else np.array([0.0, 1.0])
            
            old_i = centers[i].copy()
            centers[i] += step * v
            centers[i] = np.clip(centers[i], -0.5, 0.5)
            
            new_r = _compute_radius(centers)
            if new_r <= current_r:
                centers[i] = old_i
    
    return best_centers, best_r

def construct():
    start_time = time.time()
    timeout = 110.0  # 40s margin for safety
    
    # Phase 1: Hexagonal initialization
    centers = _hexagonal_init()
    
    # Phase 2: Force-based relaxation (bulk of the time)
    remaining = timeout - (time.time() - start_time)
    if remaining > 5.0:
        centers, r1 = _force_relaxation(centers, start_time, time.time() + remaining - 5.0)
    
    # Phase 3: High-precision polish
    remaining = timeout - (time.time() - start_time)
    if remaining > 0.5:
        centers, r2 = _polish(centers, start_time, time.time() + remaining)
    
    # Final validation
    final_r = _compute_radius(centers)
    if final_r <= 0 or not np.isfinite(final_r):
        # Fallback to hexagonal init
        centers = _hexagonal_init()
    
    return centers
# EVOLVE-BLOCK-END
