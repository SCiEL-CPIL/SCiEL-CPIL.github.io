import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 1. Initialization
    # Hexagonal lattice with slight jitter
    def get_initial_points():
        points = []
        # Target radius ~ 0.053 -> diameter 0.106
        # Spacing ~ 0.106
        spacing = 0.105
        k = int(1.0 / spacing) + 2
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 0.95:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
            # Add random points
            while len(points) < N:
                p_new = np.random.uniform(-0.9, 0.9, 2)
                points = np.vstack([points, p_new])
        
        # Randomly select N points
        idx = np.random.choice(len(points), N, replace=False)
        p = points[idx]
        
        # Add small jitter
        p += np.random.normal(0, 0.001, p.shape)
        
        # Ensure inside
        radii = np.linalg.norm(p, axis=1)
        mask = radii > 0.99
        if np.any(mask):
            p[mask] *= 0.99 / radii[mask, np.newaxis]
            
        return p

    p = get_initial_points()
    
    def calc_score(points):
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        radii = np.linalg.norm(points, axis=1)
        min_boundary_dist = np.min(1.0 - radii)
        
        return min(min_pair_dist / 2.0, min_boundary_dist)
    
    best_p = p.copy()
    best_score = calc_score(p)
    
    beta = 5.0
    eta = 0.0005
    
    while time.time() < deadline:
        # Compute Gradient
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, 1.0)
        
        # Handle close points
        close_mask = dists < 1e-6
        if np.any(close_mask):
            # Perturb
            idx = np.where(close_mask)
            for i, j in zip(idx[0], idx[1]):
                if i < j:
                    p[i] += np.random.normal(0, 1e-4, 2)
                    p[j] -= np.random.normal(0, 1e-4, 2)
            dists[close_mask] = 1e-6
            diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            np.fill_diagonal(dists, 1.0)
            
        mask = ~np.eye(N, dtype=bool)
        weights = np.exp(-beta * dists)
        weights[~mask] = 0.0
        
        u = diffs / dists[:, :, np.newaxis]
        weighted_u = weights[:, :, np.newaxis] * u
        grad_pairs = np.sum(weighted_u, axis=1) * beta
        
        radii = np.linalg.norm(p, axis=1)
        radii_safe = np.where(radii < 1e-6, 1e-6, radii)
        u_boundary = p / radii_safe[:, np.newaxis]
        
        boundary_dist = 1.0 - radii
        weights_boundary = np.exp(-beta * boundary_dist)
        grad_boundary = beta * weights_boundary[:, np.newaxis] * u_boundary
        
        total_grad = grad_pairs + grad_boundary
        
        p -= eta * total_grad
        
        beta *= 1.02
        eta *= 0.995
        
        if int(beta) % 5 == 0:
            current_score = calc_score(p)
            if current_score > best_score:
                best_score = current_score
                best_p = p.copy()
                
    return best_p
