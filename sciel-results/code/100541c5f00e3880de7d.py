import numpy as np
import time

def construct():
    N = 300
    deadline = time.time() + 110.0
    
    points = _hexagonal_init(N)
    r = _min_radius(points)
    
    points, r = _relax(points, r, deadline)
    
    # Final safety check
    if not np.all(np.isfinite(points)):
        return _hexagonal_init(N)
        
    radii = np.linalg.norm(points, axis=1)
    if np.any(radii > 1.0):
        points = points / radii[:, np.newaxis] * 0.999999
        r = _min_radius(points)
        
    return points

def _min_radius(points):
    n = len(points)
    if n == 0: return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    radii = np.linalg.norm(points, axis=1)
    min_boundary_dist = np.min(1.0 - radii)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def _hexagonal_init(N):
    s = np.sqrt(4 * np.pi / (N * np.sqrt(3)))
    k = int(2.0 / s) + 2
    xs = []
    ys = []
    for j in range(-k, k+1):
        for i in range(-k, k+1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2.0
            if x*x + y*y <= 1.0:
                xs.append(x)
                ys.append(y)
    points = np.column_stack((xs, ys))
    
    if len(points) > N:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        points = points[idx]
    return points

def _relax(points, r, deadline):
    N = len(points)
    alpha = 0.005
    best_points = points.copy()
    best_r = r
    patience = 0
    stall_count = 0
    
    while time.time() < deadline:
        F = np.zeros_like(points)
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        dists_safe = np.sqrt(dists**2 + 1e-10)
        
        inv_d2 = 1.0 / (dists_safe ** 2)
        inv_d2[np.diag_indices(N)] = 0
        
        forces = diff * inv_d2[:, :, np.newaxis]
        mags = np.linalg.norm(forces, axis=2)
        max_force = 50.0
        scale = np.where(mags > max_force, max_force / mags, 1.0)
        forces *= scale[:, :, np.newaxis]
        
        F += np.sum(forces, axis=1)
        
        radii = np.linalg.norm(points, axis=1)
        limit = 1.0 - r
        mask = radii > limit
        if np.any(mask):
            c_norm = points[mask] / radii[mask][:, np.newaxis]
            mag = (radii[mask] - limit)
            F[mask] -= 50.0 * mag[:, np.newaxis] * c_norm
        
        new_points = points + alpha * F
        
        new_r = _min_radius(new_points)
        
        if new_r > best_r:
            best_r = new_r
            best_points = new_points.copy()
            r = new_r
            patience = 0
            stall_count = 0
        else:
            patience += 1
            if patience > 20:
                stall_count += 1
                if stall_count > 10:
                    break
                points = best_points.copy()
                alpha *= 0.5
                patience = 0
            else:
                alpha *= 0.95
        
        points = new_points

    return best_points, best_r
