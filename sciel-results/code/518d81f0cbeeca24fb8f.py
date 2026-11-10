"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _covering_radius(pts):
    """Compute covering radius in radians. Returns (radius, worst_facet_index, worst_point)."""
    try:
        hull = ConvexHull(pts)
        n_sites = len(pts)
        max_ang = 0.0
        worst_facet = -1
        worst_pt = None
        
        # Vectorized computation of circumcenters for all facets
        tris = hull.simplices
        a = pts[tris[:, 0]]
        b = pts[tris[:, 1]]
        c = pts[tris[:, 2]]
        
        # Normal vectors to each triangle
        normals = np.cross(b - a, c - a)
        norms = np.linalg.norm(normals, axis=1)
        valid = norms > 1e-12
        if not np.any(valid):
            return 0.0, -1, None
        
        normals = normals[valid]
        norms = norms[valid]
        normals = normals / norms[:, np.newaxis]
        
        # Ensure normals point outward (away from origin)
        a_valid = a[valid]
        dot_products = np.sum(normals * a_valid, axis=1)
        flip = dot_products < 0
        normals[flip] = -normals[flip]
        
        # Compute angular distance from each circumcenter to nearest site
        dots = np.clip(pts @ normals.T, -1.0, 1.0)
        dists = np.arccos(dots)  # shape: (n_sites, n_facets)
        min_dists = np.min(dists, axis=0)  # shape: (n_facets,)
        
        max_idx = np.argmax(min_dists)
        max_ang = min_dists[max_idx]
        worst_facet = np.where(valid)[0][max_idx]
        worst_pt = normals[max_idx].copy()
        
        return max_ang, worst_facet, worst_pt
    except Exception:
        return np.pi, -1, None


def _fibonacci_start():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_start(rng):
    """Generate random initial points on the sphere."""
    pts = rng.standard_normal((N, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _lloyd_step(pts, n_iter=30):
    """Perform Lloyd's algorithm iterations to minimize covering radius."""
    pts = pts.copy()
    for _ in range(n_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # Vectorized circumcenter computation
        tris = hull.simplices
        a = pts[tris[:, 0]]
        b = pts[tris[:, 1]]
        c = pts[tris[:, 2]]
        
        normals = np.cross(b - a, c - a)
        norms = np.linalg.norm(normals, axis=1)
        valid = norms > 1e-12
        if not np.any(valid):
            break
        
        normals = normals[valid]
        norms = norms[valid]
        normals = normals / norms[:, np.newaxis]
        
        a_valid = a[valid]
        dot_products = np.sum(normals * a_valid, axis=1)
        flip = dot_products < 0
        normals[flip] = -normals[flip]
        
        circumcenters = normals
        
        new_pts = pts.copy()
        for i in range(len(pts)):
            dists = np.linalg.norm(circumcenters - pts[i], axis=1)
            k = min(5, len(circumcenters))
            idx = np.argsort(dists)[:k]
            centroid = circumcenters[idx].mean(axis=0)
            nc = np.linalg.norm(centroid)
            if nc > 1e-10:
                centroid = centroid / nc
                new_pts[i] = 0.5 * pts[i] + 0.5 * centroid
                new_pts[i] /= np.linalg.norm(new_pts[i])
        
        pts = new_pts
    
    return pts


def _local_optimize(pts, n_iter=100, base_alpha=0.05):
    """Local optimization targeting the worst facet with adaptive step size."""
    pts = pts.copy()
    for it in range(n_iter):
        radius, wf, wpt = _covering_radius(pts)
        if wf < 0:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        tri = hull.simplices[wf]
        a_idx, b_idx, c_idx = tri[0], tri[1], tri[2]
        
        # Adaptive step size: reduce if improvement is small
        alpha = base_alpha * (0.5 ** (it // 20))
        
        for idx in [a_idx, b_idx, c_idx]:
            new_pt = pts[idx] + alpha * (wpt - pts[idx])
            new_pt /= np.linalg.norm(new_pt)
            pts[idx] = new_pt
    
    return pts


def _polish(pts, n_iter=50):
    """High-precision polish phase with adaptive step-size reduction."""
    pts = pts.copy()
    for it in range(n_iter):
        radius, wf, wpt = _covering_radius(pts)
        if wf < 0:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        tri = hull.simplices[wf]
        
        # Try multiple step sizes
        for alpha_mult in [1.0, 0.5, 0.25, 0.1, 0.05]:
            alpha = 0.01 * alpha_mult
            trial = pts.copy()
            for idx in tri:
                new_pt = trial[idx] + alpha * (wpt - trial[idx])
                new_pt /= np.linalg.norm(new_pt)
                trial[idx] = new_pt
            
            r_trial, _, _ = _covering_radius(trial)
            if r_trial < radius:
                pts = trial
                break
    
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Generate multiple initial configurations
    best_pts = None
    best_radius = np.pi
    
    # Fibonacci start
    pts_fib = _fibonacci_start()
    r_fib, _, _ = _covering_radius(pts_fib)
    if r_fib < best_radius:
        best_radius = r_fib
        best_pts = pts_fib.copy()
    
    # Several random starts
    for _ in range(8):
        if time.time() > deadline - 60:
            break
        pts_rand = _random_start(rng)
        r_rand, _, _ = _covering_radius(pts_rand)
        if r_rand < best_radius:
            best_radius = r_rand
            best_pts = pts_rand.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_start()
        best_radius, _, _ = _covering_radius(best_pts)
    
    pts = best_pts.copy()
    
    # Phase 2: Lloyd's algorithm for global improvement
    while time.time() < deadline - 40:
        pts_lloyd = _lloyd_step(pts, n_iter=20)
        r_lloyd, _, _ = _covering_radius(pts_lloyd)
        if r_lloyd < best_radius:
            best_radius = r_lloyd
            best_pts = pts_lloyd.copy()
            pts = pts_lloyd.copy()
        else:
            pts_lloyd = _lloyd_step(pts, n_iter=5)
            r_lloyd, _, _ = _covering_radius(pts_lloyd)
            if r_lloyd < best_radius:
                best_radius = r_lloyd
                best_pts = pts_lloyd.copy()
                pts = pts_lloyd.copy()
            else:
                break
    
    # Phase 3: Local optimization targeting worst facets
    while time.time() < deadline - 15:
        pts_local = _local_optimize(pts, n_iter=50)
        r_local, _, _ = _covering_radius(pts_local)
        if r_local < best_radius:
            best_radius = r_local
            best_pts = pts_local.copy()
            pts = pts_local.copy()
        else:
            pts_local = _local_optimize(pts, n_iter=20)
            r_local, _, _ = _covering_radius(pts_local)
            if r_local < best_radius:
                best_radius = r_local
                best_pts = pts_local.copy()
                pts = pts_local.copy()
            else:
                break
    
    # Phase 4: High-precision polish
    while time.time() < deadline - 5:
        pts_polished = _polish(pts, n_iter=30)
        r_polished, _, _ = _covering_radius(pts_polished)
        if r_polished < best_radius:
            best_radius = r_polished
            best_pts = pts_polished.copy()
            pts = pts_polished.copy()
        else:
            break
    
    # Ensure output is valid
    pts_out = best_pts.copy()
    norms = np.linalg.norm(pts_out, axis=1)
    pts_out = pts_out / norms[:, np.newaxis]
    
    # Check for coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(pts_out[i] - pts_out[j]) < 1e-8:
                pts_out[j] += np.random.randn(3) * 0.01
                pts_out[j] /= np.linalg.norm(pts_out[j])
    
    return pts_out
# EVOLVE-BLOCK-END
