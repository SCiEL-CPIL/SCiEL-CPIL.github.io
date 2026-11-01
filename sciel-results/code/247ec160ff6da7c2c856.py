"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    # Set random seed for reproducibility in debugging, but we want diversity.
    # random.seed(42)
    
    def get_radius(pts):
        # Check boundary
        norms = np.linalg.norm(pts, axis=1)
        if np.any(norms > 1.0):
            return -1.0 # Invalid
        
        r_boundary = 1.0 - np.max(norms)
        
        # Pairwise distances
        # N=50, 50x50 matrix is fine
        dists = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist < 1e-10:
            return -1.0 # Coincident centers
            
        r_pair = min_dist / 2.0
        
        return min(r_boundary, r_pair)

    def init_spiral():
        # Generate points in a spiral to ensure good initial spread
        pts = np.zeros((N, 2))
        for i in range(N):
            # r from 0 to 1
            r = (i + 0.5) / N
            theta = i * 2.39996323 # Golden angle
            pts[i] = [r * np.cos(theta), r * np.sin(theta)]
        
        # Ensure inside circle (should be, but just in case)
        norms = np.linalg.norm(pts, axis=1)
        for i in range(N):
            if norms[i] > 1.0:
                pts[i] = pts[i] / norms[i] * 0.99
        return pts

    best_points = None
    best_r = -1.0
    
    # Run multiple restarts
    num_restarts = 10
    iters_per_restart = 20000
    
    start_time = time.time()
    max_time = 100.0 # Leave some buffer for 120s limit
    
    for restart in range(num_restarts):
        if time.time() - start_time > max_time:
            break
            
        pts = init_spiral()
        # Add some random noise to break symmetry
        pts += np.random.normal(0, 0.05, pts.shape)
        # Project back to valid region if necessary
        norms = np.linalg.norm(pts, axis=1)
        for i in range(N):
            if norms[i] > 1.0:
                pts[i] = pts[i] / norms[i] * 0.99
                
        current_r = get_radius(pts)
        if current_r < 0:
            # If initial is invalid, try to fix or skip
            # For now, just proceed, SA will try to fix
            pass
            
        local_best_r = current_r
        local_best_points = pts.copy()
        
        T = 0.1 # Initial temperature
        T_min = 1e-6
        alpha = 0.9995 # Cooling rate
        
        for i in range(iters_per_restart):
            if time.time() - start_time > max_time:
                break
                
            # Pick a random point
            idx = np.random.randint(N)
            
            # Generate a random move
            # Step size can be adaptive, but let's keep it simple
            step = 0.05
            delta = np.random.normal(0, step, 2)
            
            new_pts = pts.copy()
            new_pts[idx] += delta
            
            # Check boundary
            if np.linalg.norm(new_pts[idx]) > 1.0:
                continue # Reject move that goes outside
            
            new_r = get_radius(new_pts)
            
            # Acceptance criterion
            if new_r >= current_r:
                accept = True
            else:
                # Probability of accepting worse solution
                if T > 0:
                    prob = np.exp((new_r - current_r) / T)
                    accept = (random.random() < prob)
                else:
                    accept = False
            
            if accept:
                pts = new_pts
                current_r = new_r
                if current_r > local_best_r:
                    local_best_r = current_r
                    local_best_points = pts.copy()
                    
            T *= alpha
            
        if local_best_r > best_r:
            best_r = local_best_r
            best_points = local_best_points

    # If no valid point found (should not happen), fallback to center
    if best_points is None:
        best_points = np.zeros((N, 2))
        # Try to make it valid? 
        # Just return a valid spiral
        best_points = init_spiral()
        
    return best_points
# EVOLVE-BLOCK-END
