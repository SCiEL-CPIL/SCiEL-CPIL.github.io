"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _covering_radius(pts):
    """Compute covering radius in degrees for points on unit sphere."""
    # Build convex hull
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # fallback
    
    facets = hull.simplices
    max_ang = 0.0
    for tri in facets:
        a, b, c = pts[tri]
        # Circumcenter of triangle on unit sphere
        # For a spherical triangle, the circumcenter (pole) is along the normal to the plane
        # The angular radius is the angle from the normal to any vertex
        normal = np.cross(a, b)
        if np.linalg.norm(normal) < 1e-15:
            # Degenerate triangle
            continue
        normal = normal / np.linalg.norm(normal)
        
        # The circumradius is the angle between the normal and any vertex
        # But we need to be careful: the circumcenter could be on either side
        # The correct circumcenter is the one that gives the smallest cap
        ang = np.arccos(np.clip(np.dot(normal, a), -1.0, 1.0))
        
        # Check if this is the correct side: the circumcenter should be such that
        # all points are on the same side or the cap is empty
        # For now, use the angle as-is; if it's > pi/2, the other side is smaller
        if ang > np.pi / 2:
            ang = np.pi - ang
        
        # Verify: the circumcenter should give a cap that doesn't contain other points
        # Actually, for the convex hull, the facet's circumcenter (on the correct side)
        # gives the covering radius for that facet
        # The angle from circumcenter to vertex is the circumradius
        max_ang = max(max_ang, ang)
    
    return np.degrees(max_ang)


def _soft_max(x, beta=100.0):
    """Soft maximum using log-sum-exp."""
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta


def _objective(pts, beta=50.0):
    """Smooth surrogate for covering radius (in radians)."""
    # For each point, compute angular distance to nearest other point
    # The covering radius is related to the maximum over all directions
    # A simpler approach: use the convex hull circumradii
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    
    facets = hull.simplices
    angles = []
    for tri in facets:
        a, b, c = pts[tri]
        normal = np.cross(a, b)
        nn = np.linalg.norm(normal)
        if nn < 1e-15:
            continue
        normal = normal / nn
        ang = np.arccos(np.clip(np.dot(normal, a), -1.0, 1.0))
        if ang > np.pi / 2:
            ang = np.pi - ang
        angles.append(ang)
    
    if not angles:
        return np.pi
    
    angles = np.array(angles)
    # Return soft max for smoothness
    return _soft_max(angles, beta)


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _gradient_ascent(pts, beta=50.0, lr=0.01, steps=500, time_limit=100.0):
    """Optimize using gradient ascent on negative soft-max covering radius."""
    start_time = time.time()
    best_pts = pts.copy()
    best_obj = _objective(best_pts, beta)
    
    for step in range(steps):
        if time.time() - start_time > time_limit:
            break
        
        # Compute objective and gradient via finite differences
        f0 = _objective(pts, beta)
        
        if f0 < best_obj:
            best_obj = f0
            best_pts = pts.copy()
        
        # Compute gradient numerically
        grad = np.zeros_like(pts)
        eps = 1e-4
        for i in range(N):
            for j in range(3):
                orig = pts[i, j]
                pts[i, j] = orig + eps
                pts[i] /= np.linalg.norm(pts[i])
                fp = _objective(pts, beta)
                pts[i, j] = orig - eps
                pts[i] /= np.linalg.norm(pts[i])
                fm = _objective(pts, beta)
                pts[i, j] = orig
                pts[i] /= np.linalg.norm(pts[i])
                grad[i, j] = (fp - fm) / (2 * eps)
        
        # Gradient ascent (we want to maximize negative covering radius, i.e., minimize it)
        # Actually we want to minimize the covering radius, so gradient descent on objective
        pts = pts - lr * grad
        pts = _normalize(pts)
    
    return best_pts, best_obj


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)


def _polish(pts, beta=200.0, lr=0.005, steps=2000, time_limit=30.0):
    """High-precision polish pass with tighter tolerance."""
    start_time = time.time()
    best_pts = pts.copy()
    best_obj = _objective(best_pts, beta)
    
    for step in range(steps):
        if time.time() - start_time > time_limit:
            break
        
        f0 = _objective(pts, beta)
        if f0 < best_obj:
            best_obj = f0
            best_pts = pts.copy()
        
        grad = np.zeros_like(pts)
        eps = 1e-5
        for i in range(N):
            for j in range(3):
                orig = pts[i, j]
                pts[i, j] = orig + eps
                pts[i] /= np.linalg.norm(pts[i])
                fp = _objective(pts, beta)
                pts[i, j] = orig - eps
                pts[i] /= np.linalg.norm(pts[i])
                fm = _objective(pts, beta)
                pts[i, j] = orig
                pts[i] /= np.linalg.norm(pts[i])
                grad[i, j] = (fp - fm) / (2 * eps)
        
        pts = pts - lr * grad
        pts = _normalize(pts)
    
    return best_pts, best_obj


def construct():
    global_start = time.time()
    deadline = global_start + 140.0  # Safety margin before 150s
    
    # Start with Fibonacci sphere
    best_pts = _fibonacci_sphere(N)
    best_radius = _covering_radius(best_pts)
    
    # Try multiple random restarts with gradient optimization
    n_restarts = 8
    for r in range(n_restarts):
        if time.time() > deadline - 40.0:
            break
        
        if r == 0:
            start_pts = _fibonacci_sphere(N)
        else:
            start_pts = _random_sphere(N, seed=42 + r)
        
        # Coarse optimization
        pts, obj = _gradient_ascent(start_pts, beta=50.0, lr=0.01, steps=300, 
                                     time_limit=10.0)
        
        # Check actual covering radius
        cr = _covering_radius(pts)
        if cr < best_radius:
            best_radius = cr
            best_pts = pts.copy()
        
        # Polish if time allows
        if time.time() > deadline - 25.0:
            break
        
        pts2, obj2 = _polish(pts, beta=200.0, lr=0.003, steps=500, time_limit=15.0)
        cr2 = _covering_radius(pts2)
        if cr2 < best_radius:
            best_radius = cr2
            best_pts = pts2.copy()
    
    # Final polish on best
    if time.time() < deadline - 5.0:
        final_pts, _ = _polish(best_pts, beta=500.0, lr=0.001, steps=1000, 
                                time_limit=20.0)
        cr_final = _covering_radius(final_pts)
        if cr_final < best_radius:
            best_pts = final_pts
            best_radius = cr_final
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
