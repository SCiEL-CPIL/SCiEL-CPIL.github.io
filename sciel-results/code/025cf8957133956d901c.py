import numpy as np
import math
import time

N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def get_fibonacci_points(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts

    def log_energy(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        e = -0.25 * np.sum(np.log(dist_sq))
        return e

    def gradient(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        
        inv_dist_sq = 1.0 / dist_sq
        mask = ~np.eye(N, dtype=bool)
        inv_dist_sq = inv_dist_sq * mask
        
        grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        dot = np.sum(grad * pts, axis=1, keepdims=True)
        proj_grad = grad - dot * pts
        
        return proj_grad

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

    def optimize(pts, max_steps, time_limit, initial_step=0.01, momentum=0.9, step_decay=0.999, min_step=1e-5):
        best_pts = pts.copy()
        best_e = log_energy(pts)
        
        step = initial_step
        v = np.zeros_like(pts)
        
        for step_idx in range(max_steps):
            if time.time() > time_limit:
                break
                
            grad = gradient(pts)
            
            v = momentum * v - step * grad
            pts_new = normalize(pts + v)
            
            e_new = log_energy(pts_new)
            
            if e_new < best_e:
                best_e = e_new
                best_pts = pts_new.copy()
            
            pts = pts_new
            step *= step_decay
            if step < min_step:
                step = min_step
                
        return best_pts, best_e

    # Phase 1: Coarse optimization from Fibonacci start
    pts = get_fibonacci_points(N)
    pts, e1 = optimize(pts, 3000, start_time + 25.0, initial_step=0.02, momentum=0.9, step_decay=0.998, min_step=1e-4)

    # Phase 2: Fine optimization
    pts, e2 = optimize(pts, 10000, start_time + 70.0, initial_step=1e-3, momentum=0.85, step_decay=0.9995, min_step=1e-6)

    # Phase 3: Very fine polish
    pts, e3 = optimize(pts, 20000, start_time + 105.0, initial_step=1e-5, momentum=0.8, step_decay=0.9998, min_step=1e-7)

    # Final cleanup
    best_pts = normalize(pts)
    
    # Check for invalid states
    if not np.all(np.isfinite(best_pts)):
        # Fallback to Fibonacci
        return get_fibonacci_points(N)

    # Verify minimum distance
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    
    if min_dist < 1e-9:
        # Perturb points that are too close
        # This is a safety measure, unlikely to trigger
        rng = np.random.RandomState(42)
        for i in range(N):
            for j in range(i+1, N):
                if dist_sq[i,j] < 1e-18:
                    # Move point i slightly
                    direction = rng.randn(3)
                    direction -= np.dot(direction, best_pts[i]) * best_pts[i]
                    best_pts[i] += 1e-8 * direction
                    best_pts[i] /= np.linalg.norm(best_pts[i])

    return best_pts

# EVOLVE-BLOCK-END
