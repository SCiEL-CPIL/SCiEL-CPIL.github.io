"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def _min_distance(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    mask = ~np.eye(N, dtype=bool)
    min_pair_dist = np.min(dist[mask])
    
    d_left = points[:, 0] + 0.5
    d_right = 0.5 - points[:, 0]
    d_bottom = points[:, 1] + 0.5
    d_top = 0.5 - points[:, 1]
    
    min_boundary_dist = np.min([np.min(d_left), np.min(d_right), np.min(d_bottom), np.min(d_top)])
    
    return min(min_pair_dist, min_boundary_dist)

def _relax_step(points, r, lr):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    mask = (dist < 2 * r) & (dist > 1e-9)
    
    factor = np.zeros_like(dist)
    factor[mask] = (2 * r - dist[mask]) / dist[mask]
    
    forces = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    left_limit = -0.5 + r
    right_limit = 0.5 - r
    viol_left = np.clip(left_limit - points[:, 0], 0, None)
    viol_right = np.clip(points[:, 0] - right_limit, 0, None)
    forces[:, 0] += viol_left - viol_right
    
    bottom_limit = -0.5 + r
    top_limit = 0.5 - r
    viol_bottom = np.clip(bottom_limit - points[:, 1], 0, None)
    viol_top = np.clip(points[:, 1] - top_limit, 0, None)
    forces[:, 1] += viol_bottom - viol_top
    
    new_points = points + lr * forces
    new_points[:, 0] = np.clip(new_points[:, 0], -0.5, 0.5)
    new_points[:, 1] = np.clip(new_points[:, 1], -0.5, 0.5)
    
    return new_points

def _init_hex(N):
    s = np.sqrt(2 / (N * np.sqrt(3)))
    
    while True:
        points = []
        i = 0
        while True:
            y = -0.5 + s * np.sqrt(3) / 2 + i * s * np.sqrt(3) / 2
            if y > 0.5:
                break
            offset = (i % 2) * s / 2
            j = 0
            while True:
                x = -0.5 + s / 2 + offset + j * s
                if x > 0.5:
                    break
                points.append([x, y])
                j += 1
            i += 1
        
        points = np.array(points)
        if len(points) >= N:
            return points[:N]
        else:
            s *= 0.9

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    points = _init_hex(N)
    
    d_min = _min_distance(points)
    r = d_min / 2.0 * 0.9
    
    best_r = r
    best_points = points.copy()
    
    lr = 0.1
    
    while time.time() < deadline:
        for _ in range(100):
            points = _relax_step(points, r, lr)
            
        d_min = _min_distance(points)
        
        if d_min > 2 * r:
            r = d_min / 2.0
            best_r = d_min / 2.0
            best_points = points.copy()
        else:
            points += np.random.normal(0, 0.001, points.shape)
            points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
            points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
            
    return best_points
# EVOLVE-BLOCK-END
