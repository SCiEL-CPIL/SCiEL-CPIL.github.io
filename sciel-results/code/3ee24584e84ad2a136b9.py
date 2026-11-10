"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _compute_facet_data(pts):
    """
    Computes normals (circumcenters) and angular distances (radii) for all facets.
    Returns:
        normals: array of shape (n_facets, 3)
        radii: array of shape (n_facets,)
        simplices: array of shape (n_facets, 3)
        valid_mask: boolean array of shape (n_facets,)
    """
    try:
        hull = ConvexHull(pts)
        tris = hull.simplices
        a = pts[tris[:, 0]]
        b = pts[tris[:, 1]]
        c = pts[tris[:, 2]]
        
        # Normal vectors to each triangle
        normals = np.cross(b - a, c - a)
        norms = np.linalg.norm(normals, axis=1)
        valid = norms > 1e-12
        
        if not np.any(valid):
            return None, None, None, None

        normals = normals[valid]
        norms = norms[valid]
        normals = normals / norms[:, np.newaxis]
        
        # Ensure normals point outward (away from origin)
        a_valid = a[valid]
        dot_products = np.sum(normals * a_valid, axis=1)
        flip = dot_products < 0
        normals[flip] = -normals[flip]
        
        # Compute angular distance from each circumcenter to nearest site
        # dots[i, j] is dot product of point i and normal j
        dots = np.clip(pts @ normals.T, -1.0, 1.0)
        dists = np.arccos(dots)  # shape: (n_sites, n_facets)
        min_dists = np.min(dists, axis=0)  # shape: (n_facets,)
        
        # Map back to original indices if needed, but we only care about valid ones for now
        # The simplices returned should correspond to the valid normals
        valid_simplices = tris[valid]
        
        return normals, min_dists, valid_simplices, valid

    except Exception:
        return None, None, None, None


def _covering_radius(pts):
    """Compute covering radius in radians. Returns (radius, worst_facet_index, worst_point)."""
    normals, radii, simplices, valid = _compute_facet_data(pts)
    if normals is None:
        return np.pi, -1, None
    
    max_idx = np.argmax(radii)
    max_ang = radii[max_idx]
    worst_pt = normals[max_idx].copy()
    
    # Map back to the index in the original hull.simplices if we need it for other purposes
    # But here we just return the point and the radius.
    # The 'worst_facet_index' is relative to the valid set. 
    # For the caller, we might need the actual indices in pts.
    # Let's return the indices of the vertices of the worst facet.
    worst_vertices = simplices[max_idx]
    
    return max_ang, worst_vertices, worst_pt


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
        normals, radii, simplices, valid = _compute_facet_data(pts)
        if normals is None:
            break
        
        circumcenters = normals
        
        new_pts = pts.copy()
        for i in range(len(pts)):
            # Find facets that contain point i
            # This is O(N * F) which is okay for 130 points
            # A more efficient way: precompute incidence
            # For now, simple loop
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


def _soft_max_optimize(pts, n_iter=100, beta=50.0, base_lr=0.05):
    """
    Optimizes using a soft-max surrogate.
    Gradient approximation: For each point i, sum over incident facets f:
    w_f * (c_f - p_i) / ||c_f - p_i||
    where w_f = exp(beta * r_f) / sum(exp(beta * r_j))
    """
    pts = pts.copy()
    
    # Precompute incidence list: for each point, which facets (indices in valid set) include it?
    # We will recompute this every iteration because the hull changes.
    
    for it in range(n_iter):
        normals, radii, simplices, valid = _compute_facet_data(pts)
        if normals is None:
            break
        
        n_facets = len(normals)
        if n_facets == 0:
            break
        
        # Compute weights
        # Shift for numerical stability
        r_max = np.max(radii)
        weights = np.exp(beta * (radii - r_max))
        sum_weights = np.sum(weights)
        if sum_weights < 1e-10:
            break
        weights /= sum_weights
        
        # Compute gradient for each point
        # grad[i] = sum_{f incident to i} w_f * direction(c_f - p_i)
        grad = np.zeros_like(pts)
        
        # Efficiently accumulate gradients
        # For each facet f, add contribution to its 3 vertices
        for f in range(n_facets):
            w = weights[f]
            if w < 1e-6: # Skip negligible facets
                continue
            c_f = normals[f]
            v_indices = simplices[f]
            for idx in v_indices:
                p_i = pts[idx]
                diff = c_f - p_i
                norm_diff = np.linalg.norm(diff)
                if norm_diff > 1e-8:
                    # Direction to move p_i to reduce distance to c_f
                    # We want to move p_i towards c_f
                    # So gradient of distance is (p_i - c_f)/norm. 
                    # We want to minimize distance, so we move in direction -grad_dist = (c_f - p_i)/norm
                    grad[idx] += w * (diff / norm_diff)
        
        # Project gradient to tangent space
        for i in range(len(pts)):
            radial = np.dot(grad[i], pts[i]) * pts[i]
            grad[i] -= radial
        
        # Adaptive learning rate
        lr = base_lr * (0.95 ** (it // 10))
        
        # Update
        new_pts = pts + lr * grad
        new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
        
        # Check improvement using hard max
        r_new, _, _ = _covering_radius(new_pts)
        r_old, _, _ = _covering_radius(pts)
        
        if r_new < r_old:
            pts = new_pts
        else:
            # Try smaller step
            new_pts = pts + 0.1 * lr * grad
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            r_new = _covering_radius(new_pts)[0]
            if r_new < r_old:
                pts = new_pts
            else:
                # If no improvement, we might be at a local minimum or step is too big
                # Break if we've tried enough
                if it > 20:
                    break

    return pts


def _polish(pts, n_iter=50):
    """High-precision polish phase. Uses very small steps on the single worst facet."""
    pts = pts.copy()
    for it in range(n_iter):
        radius, wv, wpt = _covering_radius(pts)
        if wv is None:
            break
        
        # wv contains indices of the 3 vertices
        # Try moving them towards wpt
        improved = False
        for alpha in [0.005, 0.002, 0.001]:
            trial = pts.copy()
            for idx in wv:
                new_pt = trial[idx] + alpha * (wpt - trial[idx])
                new_pt /= np.linalg.norm(new_pt)
                trial[idx] = new_pt
            
            r_trial, _, _ = _covering_radius(trial)
            if r_trial < radius:
                pts = trial
                improved = True
                break
        
        if not improved:
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
            # If Lloyd's didn't help, try a few more iterations with smaller steps? 
            # Or just break to save time for optimization.
            pts_lloyd = _lloyd_step(pts, n_iter=5)
            r_lloyd, _, _ = _covering_radius(pts_lloyd)
            if r_lloyd < best_radius:
                best_radius = r_lloyd
                best_pts = pts_lloyd.copy()
                pts = pts_lloyd.copy()
            else:
                break
    
    # Phase 3: Soft-Max Gradient Descent
    while time.time() < deadline - 15:
        pts_soft = _soft_max_optimize(pts, n_iter=50, beta=50.0, base_lr=0.05)
        r_soft, _, _ = _covering_radius(pts_soft)
        if r_soft < best_radius:
            best_radius = r_soft
            best_pts = pts_soft.copy()
            pts = pts_soft.copy()
        else:
            # Try with higher beta (sharper) or lower lr
            pts_soft = _soft_max_optimize(pts, n_iter=20, beta=100.0, base_lr=0.02)
            r_soft, _, _ = _covering_radius(pts_soft)
            if r_soft < best_radius:
                best_radius = r_soft
                best_pts = pts_soft.copy()
                pts = pts_soft.copy()
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
