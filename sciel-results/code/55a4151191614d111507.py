"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle.
Strategy: fast vectorized active-set objective + global multi-start Nelder-Mead."""
import numpy as np
import random
import time
from scipy.optimize import minimize

N = 11

# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
AREA = np.sqrt(3) / 4.0  # area of the equilateral triangle


def _all_triangle_areas(points):
    """Vectorized computation of all C(n,3) triangle areas (normalized)."""
    n = len(points)
    # Build all index triples
    idx = np.array(list(range(n)))
    # Use broadcasting: for each i<j<k
    # Compute cross products
    areas = []
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            v1 = pj - pi
            for k in range(j + 1, n):
                pk = points[k]
                v2 = pk - pi
                areas.append(0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0]))
    return np.array(areas) / AREA


def _min_triangle_area_fast(points, active_set=None, threshold=0.02):
    """Compute min triangle area. If active_set is given, only check those triples first.
    active_set: list of (i,j,k) indices that were near-min in previous call.
    Returns (min_val, new_active_set)."""
    n = len(points)
    
    if active_set is not None and len(active_set) > 0:
        # Check active set first
        min_active = float('inf')
        for (i, j, k) in active_set:
            v1 = points[j] - points[i]
            v2 = points[k] - points[i]
            a = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0]) / AREA
            if a < min_active:
                min_active = a
        
        # Check all others to make sure none is smaller
        min_all = float('inf')
        new_active = []
        for i in range(n):
            pi = points[i]
            for j in range(i + 1, n):
                pj = points[j]
                v1 = pj - pi
                for k in range(j + 1, n):
                    pk = points[k]
                    v2 = pk - pi
                    a = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0]) / AREA
                    if a < min_all:
                        min_all = a
                    if a < min_all + threshold:
                        new_active.append((i, j, k))
        
        if min_all < min_active:
            # Active set missed the minimum
            return min_all, new_active
        else:
            return min_active, new_active
    else:
        # Full computation
        min_all = float('inf')
        new_active = []
        for i in range(n):
            pi = points[i]
            for j in range(i + 1, n):
                pj = points[j]
                v1 = pj - pi
                for k in range(j + 1, n):
                    pk = points[k]
                    v2 = pk - pi
                    a = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0]) / AREA
                    if a < min_all:
                        min_all = a
                    if a < min_all + threshold:
                        new_active.append((i, j, k))
        return min_all, new_active


def _objective_vec(x, active_set_ref, threshold=0.02):
    """Nelder-Mead objective: minimize negative min-triangle-area."""
    points = x.reshape(N, 2)
    # Project to triangle
    points = _project_to_triangle(points)
    val, new_active = _min_triangle_area_fast(points, active_set_ref[0], threshold)
    active_set_ref[0] = new_active
    return -val


def _project_to_triangle(points):
    """Project points to be inside the equilateral triangle."""
    pts = points.copy()
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Clip y >= 0
    mask_y = y < 0
    pts[mask_y, 1] = 0.0
    
    # Clip x range
    mask_x = (x < 0) | (x > 1)
    pts[mask_x, 0] = np.clip(x[mask_x], 0, 1)
    
    # Clip to triangle: y <= sqrt(3)*x for x<=0.5, y <= sqrt(3)*(1-x) for x>0.5
    h = np.sqrt(3) / 2.0
    for i in range(len(pts)):
        xi, yi = pts[i]
        if xi <= 0.5:
            max_y = np.sqrt(3) * xi
        else:
            max_y = np.sqrt(3) * (1.0 - xi)
        if yi > max_y:
            # Scale down y to fit
            if max_y > 1e-12:
                pts[i, 1] = max_y
            else:
                pts[i, 1] = 0.0
    return pts


def _is_inside_triangle(p):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -1e-9:
        return False
    if x < -1e-9 or x > 1.0 + 1e-9:
        return False
    max_y = np.sqrt(3) * x if x <= 0.5 else np.sqrt(3) * (1.0 - x)
    return y <= max_y + 1e-9


def _generate_structured_initial():
    """Generate a structured initial configuration based on known good patterns."""
    points = []
    # Vertices
    points.append(V0.copy())
    points.append(V1.copy())
    points.append(V2.copy())
    
    # Midpoints of edges
    points.append((V0 + V1) / 2.0)
    points.append((V1 + V2) / 2.0)
    points.append((V2 + V0) / 2.0)
    
    # Centroid
    points.append((V0 + V1 + V2) / 3.0)
    
    # Additional points
    h = np.sqrt(3) / 2.0
    points.append(np.array([0.5, h * 0.5]))
    points.append(np.array([0.5, h * 0.25]))
    
    # One more point
    points.append(np.array([0.25, h * 0.3]))
    
    points = points[:N]
    return np.array(points)


def _generate_random_points(rng, n=N):
    """Generate n random points inside the triangle using rejection sampling."""
    points = []
    while len(points) < n:
        u = rng.random()
        v = rng.random()
        if u + v <= 1.0:
            p = V0 * (1 - u - v) + V1 * u + V2 * v
            points.append(p)
    return np.array(points)


def _generate_grid_seeded(rng, n=N):
    """Generate points on a jittered grid inside the triangle."""
    points = []
    # Create a coarse grid
    for i in range(4):
        for j in range(4):
            if len(points) >= n:
                break
            # Barycentric coordinates
            a = (i + 0.5) / 4.0
            b = (j + 0.5) / 4.0
            if a + b <= 1.0:
                p = V0 * (1 - a - b) + V1 * a + V2 * b
                # Add small jitter
                p = p + np.array([rng.gauss(0, 0.02), rng.gauss(0, 0.02)])
                if _is_inside_triangle(p):
                    points.append(p)
    # Fill remaining with random
    while len(points) < n:
        points.append(_generate_random_points(rng, 1)[0])
    return np.array(points[:n])


def _nelder_mead_from(points, max_iter=500, deadline=None):
    """Run Nelder-Mead from a starting configuration."""
    x0 = points.flatten()
    active_ref = [None]  # mutable container for active set
    
    # Get initial active set
    val, active_set = _min_triangle_area_fast(points, None)
    active_ref[0] = active_set
    
    options = {
        'maxiter': max_iter,
        'xatol': 1e-8,
        'fatol': 1e-10,
        'adaptive': True
    }
    
    try:
        result = minimize(_objective_vec, x0, args=(active_ref,),
                         method='Nelder-Mead', options=options)
        best_points = _project_to_triangle(result.x.reshape(N, 2))
        best_val, _ = _min_triangle_area_fast(best_points, None)
        return best_points, best_val
    except Exception:
        return points, val


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points, normalized by triangle area."""
    val, _ = _min_triangle_area_fast(points, None)
    return val


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s
    
    best_points = None
    best_min = 0.0
    
    rng = random.Random(42)
    
    # Generate diverse starting points
    starts = []
    
    # Structured start
    starts.append(_generate_structured_initial())
    
    # Grid-seeded starts
    for _ in range(3):
        starts.append(_generate_grid_seeded(rng))
    
    # Random starts
    for _ in range(5):
        starts.append(_generate_random_points(rng))
    
    # Run Nelder-Mead from each start
    for idx, start_pts in enumerate(starts):
        if time.time() > deadline - 5:
            break
        
        pts, val = _nelder_mead_from(start_pts, max_iter=400, deadline=deadline)
        if val > best_min:
            best_min = val
            best_points = pts.copy()
    
    # If we have time, do a second round of Nelder-Mead from perturbed best
    if best_points is not None and time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            # Perturb best points
            perturbed = best_points.copy()
            for i in range(N):
                angle = rng.random() * 2 * np.pi
                dist = rng.random() * 0.05
                new_p = perturbed[i] + np.array([dist * np.cos(angle), dist * np.sin(angle)])
                if _is_inside_triangle(new_p):
                    perturbed[i] = new_p
            
            pts, val = _nelder_mead_from(perturbed, max_iter=300, deadline=deadline)
            if val > best_min:
                best_min = val
                best_points = pts.copy()
    
    # Final: one more Nelder-Mead from the best with tighter tolerance
    if best_points is not None and time.time() < deadline - 5:
        pts, val = _nelder_mead_from(best_points, max_iter=500, deadline=deadline)
        if val > best_min:
            best_min = val
            best_points = pts.copy()
    
    # Final check
    if best_points is None:
        best_points = _generate_structured_initial()
    
    return best_points
# EVOLVE-BLOCK-END
