"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import math
import time

def _covering_radius(points):
    """Compute the covering radius (in radians) as the max circumradius of convex hull facets."""
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            v = p2 - p1
            w = p3 - p1
            vv = np.dot(v, v)
            ww = np.dot(w, w)
            vw = np.dot(v, w)
            det = vv * ww - vw * vw
            if abs(det) < 1e-12:
                continue
            t1 = (0.5 * vv * ww - 0.5 * ww * vw) / det
            t2 = (0.5 * ww * vv - 0.5 * vv * vw) / det
            C_plane = p1 + t1 * v + t2 * w
            norm_C = np.linalg.norm(C_plane)
            if norm_C < 1e-12:
                continue
            C = C_plane / norm_C
            cos_R = np.clip(np.dot(C, p1), -1.0, 1.0)
            R = np.arccos(cos_R)
            if R > max_r:
                max_r = R
        return max_r
    except Exception:
        return np.pi

def _lloyd_iteration(points):
    """
    Perform one Lloyd's algorithm iteration on the sphere.
    Moves each point to the centroid of its Voronoi cell (approximated by circumcenters of incident facets).
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return points, False
    
    # Map vertex index to list of incident simplices (facets)
    incident = [[] for _ in range(len(points))]
    for i, simplex in enumerate(hull.simplices):
        for v in simplex:
            incident[v].append(i)
            
    new_points = points.copy()
    for i in range(len(points)):
        if not incident[i]:
            continue
            
        # Sum of circumcenters of incident facets
        sum_C = np.zeros(3)
        count = 0
        for facet_idx in incident[i]:
            simplex = hull.simplices[facet_idx]
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            v = p2 - p1
            w = p3 - p1
            vv = np.dot(v, v)
            ww = np.dot(w, w)
            vw = np.dot(v, w)
            det = vv * ww - vw * vw
            if abs(det) < 1e-12:
                continue
            t1 = (0.5 * vv * ww - 0.5 * ww * vw) / det
            t2 = (0.5 * ww * vv - 0.5 * vv * vw) / det
            C_plane = p1 + t1 * v + t2 * w
            norm_C = np.linalg.norm(C_plane)
            if norm_C < 1e-12:
                continue
            C = C_plane / norm_C
            sum_C += C
            count += 1
            
        if count > 0:
            new_C = sum_C / count
            norm_new_C = np.linalg.norm(new_C)
            if norm_new_C > 1e-12:
                new_points[i] = new_C / norm_new_C
            else:
                # Fallback: keep original or random small jitter
                pass
                
    return new_points, True

def _lloyd_optimize(points, max_iters=200, tol=1e-6):
    """
    Run Lloyd's algorithm until convergence.
    """
    current = points.copy()
    prev_energy = _covering_radius(current)
    
    for _ in range(max_iters):
        new_points, success = _lloyd_iteration(current)
        if not success:
            break
            
        # Check convergence based on point movement
        diff = np.linalg.norm(new_points - current)
        if diff < tol:
            current = new_points
            break
            
        current = new_points
        # Optional: track energy, but Lloyd's usually monotonically improves the "Voronoi energy"
        # which correlates with covering radius. We don't strictly need to check energy for acceptance
        # as the move is deterministic, but we can break if energy increases significantly (shouldn't happen often).
        
    return current

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(rng, n):
    pts = rng.normal(size=(n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _clustered_init(rng, n, k_clusters=10):
    """Initialize with points clustered around k random centers."""
    centers = rng.normal(size=(k_clusters, 3))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    pts = np.zeros((n, 3))
    for i in range(n):
        c = centers[i % k_clusters]
        # Add small noise
        noise = rng.normal(size=3) * 0.1
        pts[i] = c + noise
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def construct():
    start_time = time.time()
    timeout = 110.0  # Leave 10s buffer
    
    best_points = None
    best_energy = np.pi
    
    # Generate diverse initializations
    inits = []
    
    # 1. Fibonacci Spiral (Symmetric)
    inits.append(_fibonacci_spiral(N))
    
    # 2. Random
    rng = np.random.default_rng(42)
    inits.append(_random_init(rng, N))
    
    # 3. Clustered (Symmetry breaking)
    inits.append(_clustered_init(rng, N, k_clusters=13))
    inits.append(_clustered_init(rng, N, k_clusters=26))
    
    # 4. Another random with different seed
    rng2 = np.random.default_rng(123)
    inits.append(_random_init(rng2, N))
    
    # 5. "Antipodal" pairs with jitter
    half = N // 2
    pts_half = rng2.normal(size=(half, 3))
    pts_half /= np.linalg.norm(pts_half, axis=1, keepdims=True)
    pts_anti = np.vstack([pts_half, -pts_half])
    if N % 2 != 0:
        extra = rng2.normal(size=(1, 3))
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        pts_anti = np.vstack([pts_anti, extra])
    inits.append(pts_anti)
    
    # Run Lloyd's optimization on each initialization
    for init in inits:
        if time.time() - start_time > timeout:
            break
            
        # Lloyd's algorithm
        optimized = _lloyd_optimize(init, max_iters=300)
        
        # Evaluate
        energy = _covering_radius(optimized)
        
        if energy < best_energy:
            best_energy = energy
            best_points = optimized.copy()
            
    # If we have time left, do a final polish or a few more random restarts
    if best_points is not None:
        remaining_time = timeout - (time.time() - start_time)
        if remaining_time > 20.0:
            # Try 2 more random restarts
            for seed in [999, 12345]:
                rng3 = np.random.default_rng(seed)
                init = _random_init(rng3, N)
                if time.time() - start_time > timeout:
                    break
                optimized = _lloyd_optimize(init, max_iters=300)
                energy = _covering_radius(optimized)
                if energy < best_energy:
                    best_energy = energy
                    best_points = optimized.copy()
                    
    # Final normalization
    if best_points is not None:
        best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    else:
        # Fallback
        best_points = _fibonacci_spiral(N)
        
    return best_points
# EVOLVE-BLOCK-END
