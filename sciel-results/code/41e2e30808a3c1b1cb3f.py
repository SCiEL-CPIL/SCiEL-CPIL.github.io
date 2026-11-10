import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # 1. Generate initial hexagonal lattice
    s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    
    k = int(1.0 / s) + 5
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0 - s * 0.5:
                pts.append((x, y))
    
    pts = np.array(pts)
    radii = np.hypot(pts[:, 0], pts[:, 1])
    idx = np.argsort(radii, kind="stable")[:N]
    points = pts[idx].copy()

    def calc_score(pts):
        n = len(pts)
        if n < 2: return 0.0
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        mask = ~np.eye(n, dtype=bool)
        min_pair = np.min(dist[mask])
        radii = np.linalg.norm(pts, axis=1)
        min_bound = np.min(1.0 - radii)
        return min(min_pair / 2.0, min_bound)
        
    best_score = calc_score(points)
    best_points = points.copy()
    
    step_size = 0.05
    cooling = 0.9995
    
    # We will run for as many iterations as possible
    # But we check time every 100 iterations
    
    for it in range(20000):
        if it % 100 == 0 and time.time() > deadline:
            break
            
        n = len(points)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        dist[dist < 1e-10] = 1e-10
        mask = ~np.eye(n, dtype=bool)
        
        inv_dist_sq = np.zeros_like(dist)
        inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
        
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        radii = np.linalg.norm(points, axis=1)
        boundary_force = np.zeros_like(points)
        over = radii > 0.8
        if np.any(over):
            mag = (radii[over] - 0.8) ** 2
            boundary_force[over] = -mag[:, np.newaxis] * points[over] / radii[over, np.newaxis]
            
        forces += boundary_force
        
        force_norms = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_norms)
        if max_force > 1e-10:
            forces /= max_force
        
        points += step_size * forces
        
        radii = np.linalg.norm(points, axis=1)
        mask_out = radii > 1.0
        if np.any(mask_out):
            points[mask_out] /= radii[mask_out, np.newaxis]
            
        if it % 10 == 0:
            current_score = calc_score(points)
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
                
        step_size *= cooling
        
        # Random perturbation every 500 steps
        if it % 500 == 0:
            points += np.random.normal(0, 0.005, points.shape)
            radii = np.linalg.norm(points, axis=1)
            mask_out = radii > 1.0
            if np.any(mask_out):
                points[mask_out] /= radii[mask_out, np.newaxis]

    return best_points
