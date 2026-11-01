import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(rng):
    """Generate random points on the sphere."""
    theta = rng.uniform(0, 2 * np.pi, N)
    z = rng.uniform(-1, 1, N)
    r = np.sqrt(np.clip(1 - z**2, 0, None))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), z])

def _hull_radii_and_poles(pts):
    """
    Compute covering radii and pole directions for all hull facets.
    Returns (radii, poles, faces, valid_mask) where radii/holes are for valid facets only.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.array([np.pi]), np.zeros((0,3)), faces, valid
        
        normals = cross[valid].copy()
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        return radii, normals, faces, valid
    except Exception:
        return np.array([np.pi]), np.zeros((0,3)), np.empty((0,3),dtype=int), np.zeros(0,dtype=bool)

def _subgradient_descent(pts, max_iters, deadline, rng, step_scale=0.01):
    """
    Subgradient descent on the covering radius.
    At each step, find the worst facet, compute the subgradient for its 3 vertices,
    and move them in the direction that reduces the covering radius.
    """
    pts = pts.copy()
    best_pts = pts.copy()
    best_r = _hull_radii_and_poles(pts)[0].max()
    
    step = step_scale
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        radii, poles, faces, valid = _hull_radii_and_poles(pts)
        if len(radii) == 0:
            break
            
        curr_r = radii.max()
        
        if curr_r < best_r - 1e-10:
            best_r = curr_r
            best_pts = pts.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 30:
                step *= 0.9
                no_improve = 0
                if step < 1e-6:
                    break
        
        # Find worst facet
        worst_idx = np.argmax(radii)
        worst_face = faces[valid][worst_idx]
        pole = poles[worst_idx]
        
        # Subgradient: for the worst facet, moving each vertex toward the pole
        # reduces the angular distance from pole to vertices
        # The subgradient direction for vertex v is: pole - pts[v], projected tangent
        cand = pts.copy()
        for v in worst_face:
            tangent_dir = pole - pts[v]
            # Project onto tangent plane at pts[v]
            tangent_dir = tangent_dir - np.dot(tangent_dir, pts[v]) * pts[v]
            norm_t = np.linalg.norm(tangent_dir)
            if norm_t > _EPS:
                tangent_dir /= norm_t
            else:
                # Random small perturbation if at pole
                tangent_dir = rng.normal(size=3)
                tangent_dir = tangent_dir - np.dot(tangent_dir, pts[v]) * pts[v]
                norm_t = np.linalg.norm(tangent_dir)
                if norm_t > _EPS:
                    tangent_dir /= norm_t
                else:
                    continue
            cand[v] = pts[v] + step * tangent_dir
            cand[v] /= (np.linalg.norm(cand[v]) + _EPS)
        
        # Evaluate
        new_r = _hull_radii_and_poles(cand)[0].max()
        if new_r < curr_r:
            pts = cand
        else:
            # Try a smaller step or different direction
            step *= 0.7
            
    return best_pts, best_r

def _multi_restart_search(deadline, n_restarts=8):
    """
    Run multiple restarts with different random initializations.
    Each restart does subgradient descent on the worst-facet covering radius.
    """
    rng = np.random.default_rng(12345)
    best_pts = None
    best_r = np.pi
    
    # First, try Fibonacci as a good baseline
    pts_fib = _initial_fib()
    r_fib = _hull_radii_and_poles(pts_fib)[0].max()
    if r_fib < best_r:
        best_r = r_fib
        best_pts = pts_fib.copy()
    
    # Do a quick subgradient descent from Fibonacci
    time_per_restart = (deadline - time.time()) / (n_restarts + 1)
    if time_per_restart > 5:
        pts_desc, r_desc = _subgradient_descent(pts_fib, max_iters=500, 
                                                 deadline=time.time() + time_per_restart * 0.8,
                                                 rng=rng, step_scale=0.015)
        if r_desc < best_r:
            best_r = r_desc
            best_pts = pts_desc.copy()
    
    # Now do random restarts
    remaining_time = deadline - time.time()
    n_actual_restarts = min(n_restarts, max(1, int(remaining_time / 10)))
    
    for restart in range(n_actual_restarts):
        if time.time() > deadline - 2:
            break
            
        pts_init = _random_init(rng)
        # Quick initial relaxation: a few steps of subgradient descent
        remaining = deadline - time.time() - 2
        max_iters = max(50, int(remaining / 0.1))
        
        pts_desc, r_desc = _subgradient_descent(pts_init, max_iters=max_iters,
                                                 deadline=time.time() + remaining * 0.9,
                                                 rng=rng, step_scale=0.02)
        if r_desc < best_r:
            best_r = r_desc
            best_pts = pts_desc.copy()
    
    return best_pts, best_r

def _refine_from_best(pts, deadline):
    """
    Final refinement: use a finer step and more iterations from the best point.
    """
    if time.time() > deadline - 3:
        return pts
    
    rng = np.random.default_rng(999)
    remaining = deadline - time.time() - 3
    max_iters = max(200, int(remaining / 0.05))
    
    pts_final, _ = _subgradient_descent(pts, max_iters=max_iters,
                                         deadline=time.time() + remaining * 0.95,
                                         rng=rng, step_scale=0.005)
    return pts_final

def construct():
    deadline = time.time() + 140.0  # Safety margin
    
    # Multi-restart subgradient descent
    pts, best_r = _multi_restart_search(deadline, n_restarts=8)
    
    # Final refinement
    pts = _refine_from_best(pts, deadline)
    
    # Final normalization
    pts = np.array(pts, dtype=float)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / (norms + _EPS)
    
    # Validity check
    if not np.all(np.isfinite(pts)) or pts.shape != (N, 3):
        return _initial_fib()
        
    return pts
