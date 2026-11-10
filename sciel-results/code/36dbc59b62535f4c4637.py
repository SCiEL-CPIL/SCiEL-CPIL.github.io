import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START

def _generate_hex_lattice(N):
    s = 2.0 / np.sqrt(N) * 1.1
    max_k = int(1.0 / s) + 2
    points = []
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    if len(points) >= N:
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)[:N]
        return points[indices]
    else:
        return np.random.uniform(-0.8, 0.8, (N, 2))

def _compute_radius(points):
    n = len(points)
    if n < 2:
        return 0.0
        
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    radii = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.max(radii)
    
    return min(min_dist / 2.0, min_boundary)

def _relax(points, r_target, steps, lr):
    for _ in range(steps):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        dists_safe = np.where(dists < 1e-10, 1e-10, dists)
        
        overlap = 2.0 * r_target - dists
        overlap = np.where(overlap > 0, overlap, 0.0)
        
        force_dirs = diff / dists_safe[:, :, np.newaxis]
        
        forces = np.sum(overlap[:, :, np.newaxis] * force_dirs, axis=1)
        
        radii = np.linalg.norm(points, axis=1)
        boundary_overlap = radii - (1.0 - r_target)
        boundary_overlap = np.where(boundary_overlap > 0, boundary_overlap, 0.0)
        
        radii_safe = np.where(radii < 1e-10, 1e-10, radii)
        boundary_forces = -boundary_overlap[:, np.newaxis] * points / radii_safe[:, np.newaxis]
        
        forces += boundary_forces
        
        points = points + forces * lr
        
        radii = np.linalg.norm(points, axis=1)
        out_mask = radii > 1.0
        if np.any(out_mask):
            points[out_mask] = points[out_mask] / radii[out_mask, np.newaxis] * 0.999
            
    return points

def construct():
    points = _generate_hex_lattice(N)
    best_points = points.copy()
    best_r = _compute_radius(points)
    
    deadline = time.time() + 110.0
    
    lr = 0.05
    steps_per_target = 50
    target_increase = 0.002
    
    while time.time() < deadline:
        r_target = best_r * (1.0 + target_increase)
        
        trial_points = _relax(best_points.copy(), r_target, steps_per_target, lr)
        
        trial_r = _compute_radius(trial_points)
        
        if trial_r >= r_target - 1e-9:
            best_points = trial_points
            best_r = trial_r
            lr *= 0.995
            # Try to increase the step size for future targets slightly
            target_increase *= 1.01
            if target_increase > 0.01:
                target_increase = 0.01
        else:
            lr *= 0.9
            target_increase *= 0.9
            if lr < 1e-6:
                break
                
    return best_points

# EVOLVE-BLOCK-END
