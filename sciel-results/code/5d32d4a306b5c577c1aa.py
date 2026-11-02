"""Improved packing: grid start + iterative repulsion + wall constraint + local polish."""
import numpy as np
import time

N = 50

def _score(pts):
    """Compute the radius for a given set of points."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check bounds
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    # Wall distance
    wall = 0.5 - np.max(np.abs(pts), axis=1)
    min_wall = np.min(wall)
    return min(min_pair / 2.0, min_wall)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    # Initial grid
    m = 7  # 7x7 grid gives 49, need 50
    # Use a 7x7 grid but take 50 points from an 8x8 grid or adjust
    # Let's use a proper grid
    m = int(np.ceil(np.sqrt(N)))  # 8
    # Actually, let's start with a better initial configuration
    # Use a hexagonal-like packing initial guess
    
    # Start with grid
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts = grid[:N].astype(np.float64).copy()
    
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Iterative optimization
    # Use a repulsion-based approach with constraint handling
    
    # Parameters
    repulsion_strength = 1.0
    learning_rate = 0.01
    min_lr = 1e-6
    lr_decay = 0.999
    max_iters = 100000
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1.0)  # avoid division by zero
        
        # Repulsion force: F = diff / dist^3 (softened)
        # Only consider pairs that are too close
        for i in range(N):
            for j in range(i + 1, N):
                d = dist[i, j]
                if d < 0.3:  # interaction range
                    # Force proportional to 1/d^2 in direction away
                    f = repulsion_strength / (d * d + 1e-6)
                    forces[i] -= f * diff[i, j] / d
                    forces[j] += f * diff[i, j] / d
        
        # Wall repulsion: push points away from walls
        for i in range(N):
            for k in range(2):
                if pts[i, k] > 0:
                    # Near right/top wall
                    dist_wall = 0.5 - pts[i, k]
                    if dist_wall < 0.1:
                        forces[i, k] -= repulsion_strength * 0.5 / (dist_wall + 1e-6)
                else:
                    # Near left/bottom wall
                    dist_wall = 0.5 + pts[i, k]
                    if dist_wall < 0.1:
                        forces[i, k] += repulsion_strength * 0.5 / (dist_wall + 1e-6)
        
        # Update
        pts += learning_rate * forces
        
        # Project back to square
        pts = np.clip(pts, -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Evaluate
        current_score = _score(pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
        
        # Decay learning rate
        learning_rate *= lr_decay
        if learning_rate < min_lr:
            break
    
    # Local polish: try small perturbations
    polish_time = deadline - time.time()
    if polish_time > 5.0:
        # Try local search with random perturbations
        for attempt in range(200):
            if time.time() > deadline - 2.0:
                break
            
            # Try moving a random point slightly
            idx = np.random.randint(N)
            orig = best_pts[idx].copy()
            best_pts[idx] += np.random.randn(2) * 0.005
            best_pts[idx] = np.clip(best_pts[idx], -0.5 + 1e-9, 0.5 - 1e-9)
            
            s = _score(best_pts)
            if s > best_score:
                best_score = s
            else:
                best_pts[idx] = orig
    
    return best_pts
