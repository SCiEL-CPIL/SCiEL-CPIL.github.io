"""Rewritten program for sphere covering with 50 caps using Geometric Directed Search."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _covering_radius(points):
    """Compute the covering radius (in radians) for a set of points on the unit sphere.
    
    The covering radius equals the maximum circumradius of the spherical facets
    of the convex hull.
    """
    points = np.asarray(points, dtype=np.float64)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        rng = np.random.RandomState(42)
        n_samples = 200000
        gauss = rng.randn(n_samples, 3)
        gauss /= np.linalg.norm(gauss, axis=1, keepdims=True)
        d, idx = tree.query(gauss)
        max_theta = 2 * np.arcsin(np.clip(d.max() / 2.0, 0, 1))
        return max_theta, None, None

    simplices = hull.simplices
    p0 = points[simplices[:, 0]]
    p1 = points[simplices[:, 1]]
    p2 = points[simplices[:, 2]]
    
    e1 = p1 - p0
    e2 = p2 - p0
    
    n = np.cross(e1, e2)
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    
    valid = n_norm.squeeze() > 1e-12
    if not np.all(valid):
        n_norm[~valid] = 1.0
    
    C = n / n_norm
    
    centroid = (p0 + p1 + p2) / 3.0
    centroid_norm = np.linalg.norm(centroid, axis=1, keepdims=True)
    centroid_norm[centroid_norm < 1e-12] = 1.0
    centroid = centroid / centroid_norm
    
    dot_cc = np.sum(C * centroid, axis=1)
    C = np.where(dot_cc[:, None] < 0, -C, C)
    
    dots = np.sum(C * p0, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    thetas = np.arccos(dots)
    
    if not np.all(valid):
        thetas[~valid] = 0.0
        
    max_idx = np.argmax(thetas)
    max_radius = thetas[max_idx]
    
    return max_radius, thetas, simplices


def _fibonacci_sphere(n, offset=0.0):
    """Generate n points on the unit sphere using a Fibonacci spiral."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _geometric_search(pts, max_iter=10000, step_size=0.05, seed=42):
    """Geometric directed search: move points towards circumcenters of worst facets."""
    rng = np.random.RandomState(seed)
    current_pts = pts.copy()
    current_radius, thetas, simplices = _covering_radius(current_pts)
    best_pts = current_pts.copy()
    best_radius = current_radius
    
    T = 0.01
    T_min = 1e-6
    cooling = 0.999
    
    for iteration in range(max_iter):
        if T < T_min:
            break
            
        # Identify the worst facet
        max_idx = np.argmax(thetas)
        worst_simplex = simplices[max_idx]
        worst_radius = thetas[max_idx]
        
        # Compute circumcenter of the worst facet
        p0 = current_pts[worst_simplex[0]]
        p1 = current_pts[worst_simplex[1]]
        p2 = current_pts[worst_simplex[2]]
        
        e1 = p1 - p0
        e2 = p2 - p0
        n = np.cross(e1, e2)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-12:
            # Degenerate, fall back to random
            idx = rng.randint(N)
            delta = rng.randn(3) * step_size
            new_pt = current_pts[idx] + delta
            new_pt /= np.linalg.norm(new_pt)
            current_pts[idx] = new_pt
            new_radius, thetas, simplices = _covering_radius(current_pts)
            if new_radius < current_radius:
                current_radius = new_radius
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = current_pts.copy()
            else:
                prob = np.exp(-(new_radius - current_radius) / T)
                if rng.rand() < prob:
                    current_radius = new_radius
                else:
                    current_pts[idx] = current_pts[idx] # revert
            T *= cooling
            continue

        C = n / n_norm
        centroid = (p0 + p1 + p2) / 3.0
        centroid /= np.linalg.norm(centroid)
        
        if np.dot(C, centroid) < 0:
            C = -C
            
        # Try moving each of the 3 points towards C
        best_move = None
        best_new_radius = current_radius
        
        for k in range(3):
            idx = worst_simplex[k]
            original = current_pts[idx].copy()
            
            # Move towards circumcenter
            # Blend with current position to keep on sphere and control step size
            # We want to move a fraction of the way to C
            # Vector from original to C on sphere
            # Simple approach: original + step_size * (C - original)
            # But (C - original) is not tangent. Project to tangent plane.
            v = C - original
            # Project v onto tangent plane at original: v - (v . original) * original
            proj = v - np.dot(v, original) * original
            proj_norm = np.linalg.norm(proj)
            
            if proj_norm < 1e-12:
                continue
                
            # Normalize and scale
            move_dir = proj / proj_norm
            new_pt = original + step_size * move_dir
            new_pt /= np.linalg.norm(new_pt)
            
            current_pts[idx] = new_pt
            new_radius, new_thetas, new_simplices = _covering_radius(current_pts)
            
            if new_radius < best_new_radius:
                best_new_radius = new_radius
                best_move = (idx, new_pt)
                
            # Revert to try next candidate
            current_pts[idx] = original
            
        if best_move is not None:
            idx, new_pt = best_move
            current_pts[idx] = new_pt
            current_radius = best_new_radius
            thetas, simplices = _covering_radius(current_pts)[1], _covering_radius(current_pts)[2]
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = current_pts.copy()
        else:
            # No improvement from geometric move, try random perturbation to escape
            idx = rng.randint(N)
            original = current_pts[idx].copy()
            delta = rng.randn(3) * step_size * 0.5
            new_pt = original + delta
            new_pt /= np.linalg.norm(new_pt)
            current_pts[idx] = new_pt
            new_radius, thetas, simplices = _covering_radius(current_pts)
            
            if new_radius < current_radius:
                current_radius = new_radius
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = current_pts.copy()
            else:
                prob = np.exp(-(new_radius - current_radius) / T)
                if rng.rand() < prob:
                    current_radius = new_radius
                else:
                    current_pts[idx] = original
        
        T *= cooling
        
        # Occasionally shuffle points to break symmetry? 
        # Not strictly necessary if we have random fallbacks.
        
    return best_pts, best_radius


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    
    best_pts = None
    best_radius = np.inf
    
    # Multi-start with different initial configurations
    initializations = [
        ("fib0", _fibonacci_sphere(N, 0.0)),
        ("fib025", _fibonacci_sphere(N, 0.25)),
        ("fib05", _fibonacci_sphere(N, 0.5)),
        ("fib075", _fibonacci_sphere(N, 0.75)),
    ]
    
    # First pass: Geometric search on each initialization
    for name, init_pts in initializations:
        if time.time() - start_time > 80:
            break
        
        pts, radius = _geometric_search(init_pts, max_iter=2000, step_size=0.05, seed=hash(name) % 100000)
        
        if radius < best_radius:
            best_radius = radius
            best_pts = pts.copy()
            
    # Second pass: Finer geometric search on the best solution
    if best_pts is not None and time.time() - start_time < 100:
        pts, radius = _geometric_search(best_pts, max_iter=3000, step_size=0.02, seed=77777)
        if radius < best_radius:
            best_radius = radius
            best_pts = pts
            
    # Third pass: Even finer polish
    if best_pts is not None and time.time() - start_time < 112:
        pts, radius = _geometric_search(best_pts, max_iter=2000, step_size=0.005, seed=12345)
        if radius < best_radius:
            best_radius = radius
            best_pts = pts
            
    # Final ultra-fine polish
    if best_pts is not None and time.time() - start_time < 117:
        pts, radius = _geometric_search(best_pts, max_iter=1000, step_size=0.001, seed=54321)
        if radius < best_radius:
            best_radius = radius
            best_pts = pts
            
    # Ensure all points are on the unit sphere
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
