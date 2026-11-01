import numpy as np
import time

N = 150

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin for 120s limit

    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Calculate initial energy
    def calc_energy(pts):
        # pts is N x 3
        # Compute pairwise distances
        # dist[i, j] = |pts[i] - pts[j]|
        # We only need i < j
        # Using broadcasting
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        # Avoid log(0)
        dist = np.maximum(dist, 1e-12)
        # Mask upper triangle
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        energy = np.sum(np.log(1.0 / dist[mask]))
        return energy

    best_energy = calc_energy(pts)
    best_pts = pts.copy()
    
    # Gradient descent parameters
    step_size = 0.01
    # We can use a schedule for step size or adaptive
    # Let's try a simple adaptive step size: if energy increases, reduce step size and revert
    
    max_iters = 50000 # High limit, will break by time
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # Compute forces
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        # Avoid division by zero
        dist_sq = np.maximum(dist_sq, 1e-12)
        
        # Force on i from j: (pts[i] - pts[j]) / dist_sq[i,j]
        # F_i = sum_j (pts[i] - pts[j]) / dist_sq[i,j]
        # Note: diagonal is 0/0, but we set dist_sq to 1e-12, so diff is 0, so term is 0.
        forces = np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
        
        # Project forces to tangent space
        # P_i = I - pts[i] pts[i]^T
        # proj_force_i = forces[i] - dot(forces[i], pts[i]) * pts[i]
        dot_products = np.sum(forces * pts, axis=1)
        proj_forces = forces - dot_products[:, np.newaxis] * pts
        
        # Update points
        new_pts = pts + step_size * proj_forces
        
        # Normalize
        norms = np.sqrt(np.sum(new_pts**2, axis=1))
        new_pts = new_pts / norms[:, np.newaxis]
        
        # Calculate new energy
        new_energy = calc_energy(new_pts)
        
        # Accept if better
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
            pts = new_pts
            # Maybe increase step size slightly if it went well? Or keep constant.
            # Let's keep constant for simplicity, or decay slowly.
            # step_size *= 1.001
        else:
            # Revert and reduce step size
            step_size *= 0.9
            
    return best_pts
