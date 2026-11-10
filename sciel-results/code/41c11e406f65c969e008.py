"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay


def _normalize(v):
    v = np.asarray(v)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return v
        return v / norm
    else:
        norms = np.linalg.norm(v, axis=1)
        norms[norms < 1e-12] = 1.0
        return v / norms[:, np.newaxis]


def _covering_radius(pts):
    """Compute covering radius using a dense sample grid on the sphere."""
    # Use a fixed set of sample points for consistency and speed
    # 10,000 points is a good balance for N=50
    # We can generate them once or use a simple grid. 
    # For simplicity and reproducibility in this function, we'll generate a 
    # deterministic set of points if not cached, but since we call this often,
    # let's just use a pre-computed global or generate quickly.
    
    # To avoid global state, we generate a standard set of 10000 points.
    # This is fast enough.
    n_samples = 10000
    # Use Fibonacci spiral for even distribution
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute dot products: (n_samples, N)
    dots = samples @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    
    # For each sample, the nearest point maximizes the dot product
    max_dots = np.max(dots, axis=1)
    
    # The angle is arccos(max_dot)
    # We want the max angle
    angles = np.arccos(max_dots)
    max_angle = np.max(angles)
    
    return max_angle * 180.0 / np.pi


def _local_optimize(pts, max_time=30.0):
    """Local optimization to reduce covering radius."""
    start_time = time.time()
    pts = _normalize(pts.copy())
    
    best_cr = _covering_radius(pts)
    best_pts = pts.copy()
    
    step_size = 0.05
    n_iter = 0
    
    while time.time() - start_time < max_time:
        n_iter += 1
        improved = False
        
        for i in range(N):
            if time.time() - start_time > max_time - 5.0:
                break
            
            # Try moving point i in random direction
            for trial in range(10):
                direction = np.random.randn(3)
                norm_dir = np.linalg.norm(direction)
                if norm_dir < 1e-10:
                    continue
                direction /= norm_dir
                
                new_pts = pts.copy()
                # Ensure we handle the normalization correctly for a single point update
                new_pt = new_pts[i] + step_size * direction
                new_pts[i] = _normalize(new_pt)
                
                cr = _covering_radius(new_pts)
                if cr < best_cr - 1e-10:
                    best_cr = cr
                    best_pts = new_pts.copy()
                    pts = new_pts.copy()
                    improved = True
                    break
        
        if not improved:
            # Reduce step size
            step_size *= 0.8
            if step_size < 1e-4:
                break
    
    return best_pts, best_cr


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Local optimization
    if time.time() < deadline - 10.0:
        optimized_pts, cr = _local_optimize(pts, max_time=50.0)
        if cr < best_cr:
            best_pts = optimized_pts
            best_cr = cr
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 20.0:
            break
        
        # Random start
        rand_pts = np.random.randn(N, 3)
        rand_pts = _normalize(rand_pts)
        
        opt_pts, cr = _local_optimize(rand_pts, max_time=15.0)
        if cr < best_cr:
            best_pts = opt_pts
            best_cr = cr
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
