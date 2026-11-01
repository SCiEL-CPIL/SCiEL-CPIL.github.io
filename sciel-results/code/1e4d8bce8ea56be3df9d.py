import numpy as np
import scipy.spatial as spatial
import time

N = 130

def construct():
    start_time = time.time()
    timeout = 145.0  # Leave 5s margin for 150s limit
    
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = np.inf
    
    def compute_covering_radius(pts):
        try:
            hull = spatial.ConvexHull(pts)
            max_rad = 0.0
            for simplex in hull.simplices:
                v1, v2, v3 = pts[simplex]
                
                u = v1 - v2
                w = v2 - v3
                c_dir = np.cross(u, w)
                c_norm = np.linalg.norm(c_dir)
                if c_norm < 1e-10:
                    continue
                c = c_dir / c_norm
                
                cos_alpha = c.dot(v1)
                cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
                alpha = np.arccos(cos_alpha)
                
                if alpha > max_rad:
                    max_rad = alpha
            return max_rad
        except Exception:
            return np.inf

    best_radius = compute_covering_radius(points)
    
    eta = 0.05 # Learning rate
    
    while time.time() - start_time < timeout:
        try:
            hull = spatial.ConvexHull(points)
        except Exception:
            break
            
        grads = np.zeros_like(points)
        
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex]
            
            u = v1 - v2
            w = v2 - v3
            c_dir = np.cross(u, w)
            c_norm = np.linalg.norm(c_dir)
            if c_norm < 1e-10:
                continue
            c = c_dir / c_norm
            
            cos_alpha = c.dot(v1)
            cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
            alpha = np.arccos(cos_alpha)
            
            weight = alpha
            
            for idx in simplex:
                v = points[idx]
                grad = c - (v.dot(c)) * v
                grads[idx] += weight * grad
        
        points = points + eta * grads
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        current_radius = compute_covering_radius(points)
        
        if np.isfinite(current_radius) and current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        else:
            # Revert if it got worse or invalid
            points = best_points.copy()
            eta *= 0.9 # Reduce learning rate on failure
            
        eta *= 0.999 # Slow decay
        
    return best_points
