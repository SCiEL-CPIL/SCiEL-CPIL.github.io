"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """Compute covering radius via convex hull facet circumradii."""
    try:
        hull = ConvexHull(pts)
        max_r = 0.0
        for simplex in hull.simplices:
            p = pts[simplex]
            # Compute circumradius of the simplex on unit sphere
            # For a triangle on unit sphere, circumradius angle = arccos of dot product with circumcenter
            a, b, c = p[0], p[1], p[2]
            # Find circumcenter of triangle abc
            # Solve for x such that |x-a|^2 = |x-b|^2 = |x-c|^2 with x on unit sphere
            # Project circumcenter to sphere
            # Using formula: circumcenter = (a+b+c + t*(a×b + b×c + c×a)) / norm
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            s = a + b + c
            t_vec = ab + bc + ca
            # Solve for t such that |s + t*t_vec|^2 = 1
            # |s|^2 + 2t*s·t_vec + t^2|t_vec|^2 = 1
            a_coeff = np.dot(t_vec, t_vec)
            b_coeff = 2.0 * np.dot(s, t_vec)
            c_coeff = np.dot(s, s) - 1.0
            if abs(a_coeff) < 1e-15:
                cc = s / np.linalg.norm(s)
            else:
                disc = b_coeff**2 - 4*a_coeff*c_coeff
                if disc < 0:
                    cc = s / np.linalg.norm(s)
                else:
                    sq = np.sqrt(disc)
                    t1 = (-b_coeff + sq) / (2*a_coeff)
                    t2 = (-b_coeff - sq) / (2*a_coeff)
                    cc1 = s + t1*t_vec
                    cc2 = s + t2*t_vec
                    n1 = np.linalg.norm(cc1)
                    n2 = np.linalg.norm(cc2)
                    if abs(n1 - 1.0) < abs(n2 - 1.0):
                        cc = cc1 / n1
                    else:
                        cc = cc2 / n2
            # Circumradius is max angle from cc to a, b, c
            dots = np.array([np.dot(cc, a), np.dot(cc, b), np.dot(cc, c)])
            dots = np.clip(dots, -1.0, 1.0)
            angles = np.arccos(dots)
            max_r = max(max_r, angles.max())
        return np.degrees(max_r)
    except Exception:
        return np.pi / 2.0  # 90 degrees as fallback


def _local_optimize(pts, max_iters=2000, step_size=0.1, time_limit=10.0):
    """Local optimization to minimize covering radius."""
    start_time = time.time()
    pts = _normalize(pts.copy())
    current_cr = _covering_radius(pts)
    best_pts = pts.copy()
    best_cr = current_cr
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        # Try small random perturbations
        improved = False
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
            idx = np.random.randint(len(pts))
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            new_pts = pts.copy()
            new_pts[idx] = pts[idx] + step_size * direction
            new_pts[idx] /= np.linalg.norm(new_pts[idx])
            new_cr = _covering_radius(new_pts)
            if new_cr < current_cr:
                pts = new_pts
                current_cr = new_cr
                if current_cr < best_cr:
                    best_pts = pts.copy()
                    best_cr = current_cr
                improved = True
        if not improved:
            step_size *= 0.8
    
    return _normalize(best_pts), best_cr


def _build_truncated_icosahedron():
    """Build a truncated icosahedron (soccer ball) with 60 vertices, then select 50."""
    # Start with icosahedron vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Truncated icosahedron has 60 vertices: 12 pentagonal faces, 20 hexagonal faces
    # We need to select 50 of the 60 vertices. 
    # Strategy: start with all 60, then iteratively remove the vertex whose removal
    # increases the covering radius the least.
    
    # Actually, let's just use a good initial configuration and optimize.
    # Use the icosahedron vertices plus their face centers (truncation points)
    
    # Better approach: use known good starting points and optimize
    # Start with Fibonacci spiral and optimize heavily
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi_angle = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi_angle), r * np.sin(phi_angle), z])
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, keep safety margin
    
    # Start with Fibonacci spiral
    pts = _build_truncated_icosahedron()
    pts, cr = _local_optimize(pts, max_iters=1000, step_size=0.15, time_limit=30.0)
    
    # Try multiple random restarts if time permits
    best_pts = pts
    best_cr = cr
    
    for restart in range(10):
        if time.time() > deadline - 20.0:
            break
        # Generate a new random starting configuration
        np.random.seed(42 + restart)
        pts_r = np.random.randn(N, 3)
        pts_r = _normalize(pts_r)
        pts_r, cr_r = _local_optimize(pts_r, max_iters=500, step_size=0.1, time_limit=10.0)
        if cr_r < best_cr:
            best_pts = pts_r
            best_cr = cr_r
    
    # Final refinement
    if time.time() < deadline - 5.0:
        best_pts, best_cr = _local_optimize(best_pts, max_iters=500, step_size=0.05, time_limit=15.0)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
