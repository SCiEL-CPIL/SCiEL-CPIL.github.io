import time
import numpy as np
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 110.0  # safety margin vs 120 s kill


def _to_sphere(pts):
    """Normalize points to the unit sphere (robust to near-zero norms)."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _facet_circumcenter(a, b, c):
    """Return the unit circumcenter of the spherical triangle (a,b,c).
    Returns None if degenerate.
    We select the circumcenter with the smaller angular distance to the vertices.
    """
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return None
    n = n / nn
    best_p = None
    best_r = np.pi
    for s in (1.0, -1.0):
        p = s * n
        da = np.arccos(np.clip(np.dot(p, a), -1.0, 1.0))
        db = np.arccos(np.clip(np.dot(p, b), -1.0, 1.0))
        dc = np.arccos(np.clip(np.dot(p, c), -1.0, 1.0))
        if abs(da - db) < 1e-6 and abs(da - dc) < 1e-6:
            if da < best_r:
                best_r = da
                best_p = p
    return best_p


def _covering_radius(pts):
    """Covering radius (radians): max facet circumradius of the convex hull."""
    pts = _to_sphere(pts)
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    if np.min(d2[np.triu_indices(len(pts), k=1)]) < 1e-14:
        return np.pi
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    if len(hull.simplices) == 0:
        return np.pi
    
    maxr = 0.0
    for idx in hull.simplices:
        a, b, c = pts[idx[0]], pts[idx[1]], pts[idx[2]]
        p = _facet_circumcenter(a, b, c)
        if p is not None:
            r = np.arccos(np.clip(np.dot(p, a), -1.0, 1.0))
            if r > maxr:
                maxr = r
    return maxr


def _fibonacci():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_pts(seed):
    rng = np.random.default_rng(seed)
    v = rng.randn(N, 3)
    return _to_sphere(v)


def _lloyd_relax(pts, steps=100, step_size=0.1):
    """Spherical Lloyd's algorithm: move points to centroids of Voronoi cells.
    Approximates Voronoi centroid by normalized sum of incident facet circumcenters.
    """
    pts = _to_sphere(pts.copy())
    best_cr = _covering_radius(pts)
    
    for step in range(steps):
        if time.time() > _DEADLINE - 5.0:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # Map: point index -> list of circumcenters of incident facets
        targets = [np.zeros(3) for _ in range(N)]
        counts = [0] * N
        
        for idx in hull.simplices:
            i, j, k = idx
            a, b, c = pts[i], pts[j], pts[k]
            p_center = _facet_circumcenter(a, b, c)
            if p_center is not None:
                targets[i] += p_center
                targets[j] += p_center
                targets[k] += p_center
                counts[i] += 1
                counts[j] += 1
                counts[k] += 1
        
        # Update points
        new_pts = pts.copy()
        for i in range(N):
            if counts[i] > 0:
                t = targets[i] / counts[i]
                tn = np.linalg.norm(t)
                if tn > 1e-12:
                    t = t / tn
                    # Move towards target
                    new_pts[i] = pts[i] + step_size * (t - pts[i])
        
        new_pts = _to_sphere(new_pts)
        
        # Check improvement
        cr = _covering_radius(new_pts)
        if cr < best_cr - 1e-12:
            pts = new_pts
            best_cr = cr
        else:
            # If no improvement, reduce step size
            step_size *= 0.9
            if step_size < 1e-4:
                break
                
    return pts, best_cr


def _greedy_refine(pts, step0=0.05):
    """Local greedy refinement: nudge each point toward its local farthest
    empty-cap circumcenter to reduce the covering radius."""
    pts = _to_sphere(pts.copy())
    best_cr = _covering_radius(pts)
    step = step0
    for it in range(40):
        if time.time() > _DEADLINE - 5.0:
            break
        improved = False
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        worst_c = None
        worst_r = 0.0
        for idx in hull.simplices:
            a, b, c = pts[idx[0]], pts[idx[1]], pts[idx[2]]
            p_center = _facet_circumcenter(a, b, c)
            if p_center is not None:
                r = np.arccos(np.clip(np.dot(p_center, a), -1.0, 1.0))
                if r > worst_r:
                    worst_r = r
                    worst_c = p_center
        if worst_c is None:
            break
        # nearest point to worst_c
        d = np.sum((pts - worst_c[None, :]) ** 2, axis=1)
        k = int(np.argmin(d))
        new = pts[k] + step * worst_c
        new = _to_sphere(new.reshape(1, 3))[0]
        cand = pts.copy()
        cand[k] = new
        cr = _covering_radius(cand)
        if cr < best_cr - 1e-9:
            pts = cand
            best_cr = cr
            improved = True
        else:
            step *= 0.8
        if not improved and step < 1e-4:
            break
    return pts, best_cr


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 110.0
    
    best_pts = None
    best_cr = np.inf
    
    # Start 1: Fibonacci
    p0 = _fibonacci()
    p_lloyd, cr_lloyd = _lloyd_relax(p0, steps=50, step_size=0.2)
    p_ref, cr_ref = _greedy_refine(p_lloyd, step0=0.05)
    if cr_ref < best_cr:
        best_cr = cr_ref
        best_pts = p_ref
        
    # Start 2-5: Random
    for seed in [42, 123, 456, 789]:
        if time.time() > _DEADLINE - 20.0:
            break
        p_rand = _random_pts(seed)
        p_lloyd, cr_lloyd = _lloyd_relax(p_rand, steps=50, step_size=0.2)
        p_ref, cr_ref = _greedy_refine(p_lloyd, step0=0.05)
        if cr_ref < best_cr:
            best_cr = cr_ref
            best_pts = p_ref

    if best_pts is None:
        return _fibonacci()
    return _to_sphere(best_pts)
# EVOLVE-BLOCK-END
