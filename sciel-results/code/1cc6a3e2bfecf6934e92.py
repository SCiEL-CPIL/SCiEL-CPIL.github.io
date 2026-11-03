import numpy as np
import time

N = 100

def construct():
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure they are on the sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # Gradient Descent
    # Learning rate schedule
    lr = 1e-4
    min_lr = 1e-9
    decay = 0.999
    
    best_pts = pts.copy()
    best_energy = np.inf
    
    # Precompute indices for pairs? 
    # Vectorized gradient:
    # diff[i, j] = pts[i] - pts[j]
    # dist[i, j] = norm(diff[i, j])
    # grad[i] = sum_j (diff[i, j] / dist[i, j]^2)
    
    while time.time() < deadline:
        # Compute pairwise differences
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        dist[dist < 1e-12] = 1e-12
        
        # Compute gradient
        # grad[i] = sum_j (pts[i] - pts[j]) / dist[i, j]^2
        # Note: diagonal is 0/0, handled by dist clipping, but term is 0 anyway?
        # Actually, for i=j, diff=0, so numerator is 0. 0/dist^2 = 0.
        
        # We need sum over j for each i.
        # term[i, j] = diff[i, j] / dist[i, j]^2
        # We can do this by broadcasting.
        
        # dist_sq = dist**2
        # inv_dist_sq = 1.0 / dist_sq
        # inv_dist_sq[np.diag_indices(N)] = 0.0
        
        # grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        dist_sq = dist * dist
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Update
        pts_new = pts + lr * grad
        
        # Project to sphere
        norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
        pts_new = pts_new / norms
        
        # Check energy?
        # To save time, we might not check every step, but let's do it periodically or if time allows.
        # Actually, computing energy is O(N^2) just like gradient.
        
        # Let's compute energy to track best
        # E = -sum_{i<j} log(dist_ij)
        # We already have dist.
        # Use upper triangle
        tri_u = np.triu_indices(N, 1)
        d_pairs = dist[tri_u]
        energy = -np.sum(np.log(d_pairs))
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_new.copy()
        
        pts = pts_new
        
        # Decay learning rate
        lr *= decay
        if lr < min_lr:
            break
            
    return best_pts
