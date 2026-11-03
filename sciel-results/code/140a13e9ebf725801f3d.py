import numpy as np
import scipy.optimize as opt
import time
from scipy.spatial import ConvexHull

N = 50

def _to_sphere(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _facet_circumradius(a, b, c):
    """
    Compute the angular circumradius of the spherical triangle (a, b, c).
    Returns (radius, circumcenter) where circumcenter is the unit vector
    equidistant from a, b, c (the one with smaller angular distance).
    """
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return np.pi, np.zeros(3)
    n = n / nn
    
    best_r = np.pi
    best_c = n
    for s in (1.0, -1.0):
        p = s * n
        da = np.arccos(np.clip(np.dot(p, a), -1.0, 1.0))
        db = np.arccos(np.clip(np.dot(p, b), -1.0, 1.0))
        dc = np.arccos(np.clip(np.dot(p, c), -1.0, 1.0))
        # Check if p is equidistant from a, b, c
        if abs(da - db) < 1e-6 and abs(da - dc) < 1e-6:
            if da < best_r:
                best_r = da
                best_c = p
    return best_r, best_c

def _covering_radius(pts):
    """
    Compute the covering radius (in radians) via convex hull facet circumradii.
    Returns (radius, worst_circumcenter) or (pi, None) if invalid.
    """
    pts = _to_sphere(pts)
    
    # Check for coincident points
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    if np.min(d2[np.triu_indices(len(pts), k=1)]) < 1e-14:
        return np.pi, None
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, None
    
    if len(hull.simplices) == 0:
        return np.pi, None
    
    max_r = 0.0
    worst_c = None
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        r, cc = _facet_circumradius(a, b, c)
        if r > max_r:
            max_r = r
            worst_c = cc
    
    return max_r, worst_c

def _fibonacci_spiral(n, offset=0.0):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=None):
    """Generate n random points on the sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    return _to_sphere(pts)

def _lloyd_relax(pts, max_iter=100, deadline=None):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation).
    Iteratively moves each point toward the circumcenter of the worst facet
    (the one with the largest circumradius), weighted by how much it exceeds
    the average, to balance the covering radius.
    """
    pts = _to_sphere(pts.copy())
    current_cr, _ = _covering_radius(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        # Compute all facet circumradii and identify the worst
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # For each point, compute a target direction based on the facets it belongs to
        # Weighted by the circumradius of those facets
        targets = np.zeros_like(pts)
        weights = np.zeros(len(pts))
        
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            r, cc = _facet_circumradius(a, b, c)
            
            # The circumcenter cc is the point on the sphere farthest from the facet
            # We want to move the vertices of this facet toward cc to reduce the radius
            # Weight by r^2 to emphasize the worst facets
            w = r * r
            for idx in simplex:
                targets[idx] += w * cc
                weights[idx] += w
        
        # Normalize targets and blend with current positions
        improved = False
        for idx in range(len(pts)):
            if weights[idx] > 1e-12:
                target = targets[idx] / np.linalg.norm(targets[idx])
                # Move a fraction toward the target
                new_pt = 0.7 * pts[idx] + 0.3 * target
                new_pt = _to_sphere(new_pt.reshape(1, 3))[0]
                pts[idx] = new_pt
                improved = True
        
        if not improved:
            break
        
        new_cr, _ = _covering_radius(pts)
        if new_cr < current_cr - 1e-10:
            current_cr = new_cr
        else:
            # Revert if no improvement
            break
        
        if current_cr < 1e-8:
            break
    
    return pts, current_cr

def _local_refine(pts, max_iter=50, step0=0.05, deadline=None):
    """
    Local greedy refinement: nudge the nearest point to the worst circumcenter
    toward it to reduce the covering radius.
    """
    pts = _to_sphere(pts.copy())
    best_cr, worst_c = _covering_radius(pts)
    step = step0
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        if worst_c is None:
            break
        
        # Find the point nearest to the worst circumcenter
        d = np.sum((pts - worst_c[None, :]) ** 2, axis=1)
        k = int(np.argmin(d))
        
        # Move that point toward the circumcenter
        new_pt = pts[k] + step * worst_c
        new_pt = _to_sphere(new_pt.reshape(1, 3))[0]
        
        cand = pts.copy()
        cand[k] = new_pt
        cr, wc = _covering_radius(cand)
        
        if cr < best_cr - 1e-9:
            pts = cand
            best_cr = cr
            worst_c = wc
            step = min(step * 1.1, 0.2)
        else:
            step *= 0.8
            if step < 1e-5:
                break
    
    return pts, best_cr

def _simulated_annealing(pts, max_iter=200, T0=0.3, T_end=0.01, deadline=None):
    """
    Simulated annealing for covering radius minimization.
    """
    current_pts = _to_sphere(pts.copy())
    current_val, _ = _covering_radius(current_pts)
    best_pts = current_pts.copy()
    best_val = current_val
    
    T = T0
    cooling = (T_end / T0) ** (1.0 / max_iter)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        T *= cooling
        
        # Perturb a random point
        idx = np.random.randint(len(current_pts))
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        new_pts = current_pts.copy()
        new_pts[idx] = current_pts[idx] + 0.1 * direction
        new_pts[idx] = _to_sphere(new_pts[idx].reshape(1, 3))[0]
        
        new_val, _ = _covering_radius(new_pts)
        
        delta = new_val - current_val
        if delta < 0 or (T > 1e-6 and np.random.random() < np.exp(-delta / T)):
            current_pts = new_pts
            current_val = new_val
            if current_val < best_val:
                best_pts = current_pts.copy()
                best_val = current_val
    
    return best_pts, best_val

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    Uses convex hull-based covering radius, Lloyd's relaxation, local refinement,
    and simulated annealing with multi-start.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    np.random.seed(42)
    
    best_pts = None
    best_val = np.pi
    
    # Start with multiple diverse initializations
    starts = [_fibonacci_spiral(N)]
    
    # Add random starts with different seeds
    for seed in [12345, 67890, 24680, 13579]:
        starts.append(_random_points(N, seed=seed))
    
    # Add icosahedron-based start
    ico = np.array([
        [0, 1, 0], [0, -1, 0],
        [1, 0, 0], [-1, 0, 0],
        [0, 0, 1], [0, 0, -1],
        [1/np.sqrt(2), 1/np.sqrt(2), 0],
        [-1/np.sqrt(2), 1/np.sqrt(2), 0],
        [1/np.sqrt(2), -1/np.sqrt(2), 0],
        [-1/np.sqrt(2), -1/np.sqrt(2), 0],
        [0, 1/np.sqrt(2), 1/np.sqrt(2)],
        [0, -1/np.sqrt(2), 1/np.sqrt(2)],
        [0, 1/np.sqrt(2), -1/np.sqrt(2)],
        [0, -1/np.sqrt(2), -1/np.sqrt(2)],
        [1/np.sqrt(2), 0, 1/np.sqrt(2)],
        [-1/np.sqrt(2), 0, 1/np.sqrt(2)],
        [1/np.sqrt(2), 0, -1/np.sqrt(2)],
        [-1/np.sqrt(2), 0, -1/np.sqrt(2)],
        [1/np.sqrt(2), 1/np.sqrt(2), 0],
        [-1/np.sqrt(2), 1/np.sqrt(2), 0],
    ])
    ico = _to_sphere(ico)
    # Expand to 50 points by adding more
    extra = _random_points(30, seed=999)
    starts.append(_to_sphere(np.vstack([ico, extra])))
    
    for st in starts:
        if time.time() > deadline - 30.0:
            break
        
        # Apply Lloyd's relaxation
        lloyd_pts, lloyd_val = _lloyd_relax(st, max_iter=80, deadline=deadline)
        
        # Apply local refinement
        ref_pts, ref_val = _local_refine(lloyd_pts, max_iter=50, step0=0.05, deadline=deadline)
        
        # Apply simulated annealing
        sa_pts, sa_val = _simulated_annealing(ref_pts, max_iter=150, T0=0.2, T_end=0.005, deadline=deadline)
        
        if sa_val < best_val:
            best_val = sa_val
            best_pts = sa_pts.copy()
    
    # Final local refinement on the best result
    if best_pts is not None and time.time() < deadline - 10.0:
        final_pts, final_val = _local_refine(best_pts, max_iter=30, step0=0.03, deadline=deadline)
        if final_val < best_val:
            best_val = final_val
            best_pts = final_pts.copy()
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    best_pts = _to_sphere(best_pts)
    
    return best_pts
