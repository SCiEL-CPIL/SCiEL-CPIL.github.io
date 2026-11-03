"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

_start_time = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _covering_radius_deg(pts):
    """Compute covering radius in degrees via convex hull Delaunay triangulation."""
    pts = np.asarray(pts, dtype=np.float64)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 90.0
    simplices = hull.simplices
    max_cr = 0.0
    for tri in simplices:
        v0, v1, v2 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        c = v0 + v1 + v2
        cn = np.linalg.norm(c)
        if cn < 1e-10:
            c = (v0 + v1) / 2.0
            cn = np.linalg.norm(c)
            if cn < 1e-10:
                continue
        c = c / cn
        d0 = np.arccos(np.clip(np.dot(c, v0), -1.0, 1.0))
        d1 = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
        d2 = np.arccos(np.clip(np.dot(c, v2), -1.0, 1.0))
        cr = max(d0, d1, d2)
        if cr > max_cr:
            max_cr = cr
    return np.degrees(max_cr)

def _fibonacci_init(seed=0):
    """Golden-angle spiral initialization."""
    rng = np.random.RandomState(seed)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.uniform(0, 2*np.pi)
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(seed=0):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _repulsion_step(pts, step_size=0.01, iters=100):
    """Fast repulsion-based optimization without convex hull at each step."""
    pts = pts.copy()
    for it in range(iters):
        if time.time() > _DEADLINE - 5:
            break
        # Compute pairwise distances
        dots = np.clip(pts @ pts.T, -1.0, 1.0)
        angles = np.arccos(dots)
        # Force: push points apart, stronger for closer points
        # Use inverse square law for repulsion
        with np.errstate(divide='ignore', invalid='ignore'):
            forces = np.zeros_like(pts)
            for i in range(N):
                # Exclude self
                dist_i = angles[i].copy()
                dist_i[i] = np.pi
                # Repulsion: 1/dist^2, but cap to avoid explosion
                rep = 1.0 / np.maximum(dist_i, 0.01)**2
                rep[i] = 0
                # Sum of unit vectors from other points, weighted by repulsion
                # Direction from j to i: pts[i] - pts[j], normalized
                for j in range(N):
                    if i == j:
                        continue
                    d = pts[i] - pts[j]
                    dn = np.linalg.norm(d)
                    if dn > 1e-10:
                        d = d / dn
                        forces[i] += rep[j] * d
        # Apply forces
        pts += step_size * forces
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _active_set_refine(pts, max_iters=200):
    """Refine using active set: only move points in worst facets."""
    pts = pts.copy()
    best_cr = _covering_radius_deg(pts)
    for it in range(max_iters):
        if time.time() > _DEADLINE - 3:
            break
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        simplices = hull.simplices
        max_cr = 0.0
        worst_tris = []
        for tri in simplices:
            v0, v1, v2 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            c = v0 + v1 + v2
            cn = np.linalg.norm(c)
            if cn < 1e-10:
                c = (v0 + v1) / 2.0
                cn = np.linalg.norm(c)
                if cn < 1e-10:
                    continue
            c = c / cn
            d0 = np.arccos(np.clip(np.dot(c, v0), -1.0, 1.0))
            d1 = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
            d2 = np.arccos(np.clip(np.dot(c, v2), -1.0, 1.0))
            cr = max(d0, d1, d2)
            if cr > max_cr:
                max_cr = cr
            if cr > max_cr * 0.95:  # Near-worst facets
                worst_tris.append((tri, c, cr))
        
        if not worst_tris:
            break
        
        # Move points in worst facets away from circumcenter
        step = 0.005
        for tri, c, cr in worst_tris:
            for idx in tri:
                pts[idx] -= step * c
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        cr_new = _covering_radius_deg(pts)
        if cr_new < best_cr:
            best_cr = cr_new
        else:
            # Revert if worse
            pts = pts - step * np.array([sum(c for t, c, _ in worst_tris if idx in t) for idx in range(N)])
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            break
    return pts, best_cr

def construct():
    best_pts = None
    best_cr = 180.0
    
    # Generate diverse initializations
    inits = []
    for seed in range(10):
        inits.append(_fibonacci_init(seed))
    for seed in range(10):
        inits.append(_random_init(seed))
    
    # Evaluate and select best initializations
    init_scores = []
    for pts in inits:
        if time.time() > _DEADLINE - 60:
            break
        cr = _covering_radius_deg(pts)
        init_scores.append((cr, pts))
    init_scores.sort(key=lambda x: x[0])
    
    # Fast repulsion optimization from top candidates
    for cr, pts in init_scores[:5]:
        if time.time() > _DEADLINE - 30:
            break
        try:
            # Fast repulsion phase
            rep_pts = _repulsion_step(pts, step_size=0.005, iters=50)
            rep_cr = _covering_radius_deg(rep_pts)
            
            # Active set refinement
            final_pts, final_cr = _active_set_refine(rep_pts, max_iters=100)
            
            if final_cr < best_cr:
                best_cr = final_cr
                best_pts = final_pts.copy()
        except Exception:
            continue
    
    # If we have time, try more from different inits
    if best_pts is not None and time.time() < _DEADLINE - 15:
        for cr, pts in init_scores[5:10]:
            if time.time() > _DEADLINE - 10:
                break
            try:
                rep_pts = _repulsion_step(pts, step_size=0.003, iters=30)
                final_pts, final_cr = _active_set_refine(rep_pts, max_iters=50)
                if final_cr < best_cr:
                    best_cr = final_cr
                    best_pts = final_pts.copy()
            except Exception:
                continue
    
    # Final polish
    if best_pts is not None and time.time() < _DEADLINE - 5:
        final_pts, final_cr = _active_set_refine(best_pts, max_iters=50)
        if final_cr < best_cr:
            best_cr = final_cr
            best_pts = final_pts.copy()
    
    # Final normalization and validation
    if best_pts is not None:
        best_pts = np.asarray(best_pts, dtype=np.float64)
        best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
        # Ensure no coincident points
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(best_pts[i] - best_pts[j])
                if d < 1e-6:
                    rng = np.random.RandomState(42 + i * 1000 + j)
                    best_pts[j] += 1e-5 * rng.randn(3)
                    best_pts[j] /= np.linalg.norm(best_pts[j])
        return best_pts
    
    # Fallback
    return _fibonacci_init(0)
# EVOLVE-BLOCK-END
