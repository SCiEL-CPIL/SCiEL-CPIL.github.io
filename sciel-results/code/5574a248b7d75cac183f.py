"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 110.0  # safety margin before 120s kill


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """Compute the covering radius (in radians) using the convex hull facets.
    
    For each facet of the convex hull, find the circumcenter and check if it's
    within the cap (i.e., the circumcenter is on the same side as the facet normal).
    The covering radius is the max angular distance from the circumcenter to the
    facet vertices.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: use Delaunay triangulation on sphere
        try:
            # Project to 2D or use a grid-based approximation
            # Use a fine grid to estimate covering radius
            n_grid = 200
            lat = np.linspace(0, np.pi, n_grid)
            lon = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
            LAT, LON = np.meshgrid(lat, lon, indexing='ij')
            grid_pts = np.column_stack([
                np.sin(LAT) * np.cos(LON).ravel(),
                np.sin(LAT) * np.sin(LON).ravel(),
                np.cos(LAT).ravel()
            ])
            # Compute min angular distance from each grid point to nearest config point
            # Using dot product: cos(angle) = dot
            # angle = arccos(dot)
            # For efficiency, compute in batches
            max_angle = 0.0
            batch_size = 5000
            for i in range(0, len(grid_pts), batch_size):
                batch = grid_pts[i:i+batch_size]
                dots = batch @ pts.T  # (batch, N)
                dots = np.clip(dots, -1.0, 1.0)
                angles = np.arccos(dots)
                min_angles = np.min(angles, axis=1)
                max_angle = max(max_angle, np.max(min_angles))
            return max_angle
        except Exception:
            return np.pi  # worst case
            
    max_radius = 0.0
    for simplex in hull.simplices:
        facet_pts = pts[simplex]
        # Compute circumcenter of the facet points on the unit sphere
        # The circumcenter in 3D of points on the unit sphere lies along the normal
        # Find the point on the sphere equidistant from all facet vertices
        
        # For a triangle on the unit sphere, the circumcenter can be found by:
        # 1. Compute the normal to the plane containing the 3 points
        # 2. The circumcenter is along this normal (or its opposite)
        # 3. Find which direction gives the closest point to the vertices
        
        a, b, c = facet_pts
        # Normal to the plane
        ab = b - a
        ac = c - a
        n = np.cross(ab, ac)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        
        # The circumcenter on the sphere is either +n or -n (whichever is closer to the vertices)
        # Actually, for points on the unit sphere, the circumcenter of the spherical triangle
        # is the point on the great circle perpendicular to the plane... 
        # Let me think more carefully.
        
        # For 3 points on the unit sphere, the circumcenter is the point p on the unit sphere
        # such that |p-a| = |p-b| = |p-c|.
        # This means p·a = p·b = p·c (since |p|=|a|=|b|=|c|=1, |p-a|^2 = 2-2p·a)
        
        # So p·(a-b) = 0 and p·(a-c) = 0, meaning p is perpendicular to (a-b) and (a-c)
        # which means p is parallel to n = (a-b) × (a-c) or its negative.
        
        # Check both directions
        for sign in [1.0, -1.0]:
            p = sign * n
            # Check if this is a valid circumcenter (should be equidistant)
            da = np.linalg.norm(p - a)
            db = np.linalg.norm(p - b)
            dc = np.linalg.norm(p - c)
            if abs(da - db) < 1e-8 and abs(da - dc) < 1e-8:
                # The angular radius is arccos(p·a)
                dot_val = np.clip(p @ a, -1.0, 1.0)
                angular_radius = np.arccos(dot_val)
                max_radius = max(max_radius, angular_radius)
                break
    
    return max_radius


def _covering_radius_fast(pts):
    """Faster covering radius computation using Delaunay triangulation."""
    pts = _normalize(pts)
    try:
        # Use Delaunay triangulation in 3D
        tri = Delaunay(pts)
        max_radius = 0.0
        for simplex in tri.simplices:
            if len(simplex) != 3:
                continue
            facet_pts = pts[simplex]
            a, b, c = facet_pts
            ab = b - a
            ac = c - a
            n = np.cross(ab, ac)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            
            for sign in [1.0, -1.0]:
                p = sign * n
                dot_val = np.clip(p @ a, -1.0, 1.0)
                # Verify equidistance
                da2 = 2 - 2 * (p @ a)
                db2 = 2 - 2 * (p @ b)
                dc2 = 2 - 2 * (p @ c)
                if abs(da2 - db2) < 1e-8 and abs(da2 - dc2) < 1e-8:
                    angular_radius = np.arccos(dot_val)
                    max_radius = max(max_radius, angular_radius)
                    break
        return max_radius
    except Exception:
        return _covering_radius(pts)


def _find_worst_facet(pts):
    """Find the facet with the largest circumradius and return its circumcenter."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, 0.0
    
    max_radius = 0.0
    worst_circumcenter = None
    worst_simplex = None
    
    for simplex in hull.simplices:
        facet_pts = pts[simplex]
        a, b, c = facet_pts
        ab = b - a
        ac = c - a
        n = np.cross(ab, ac)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        
        for sign in [1.0, -1.0]:
            p = sign * n
            da2 = 2 - 2 * (p @ a)
            db2 = 2 - 2 * (p @ b)
            dc2 = 2 - 2 * (p @ c)
            if abs(da2 - db2) < 1e-8 and abs(da2 - dc2) < 1e-8:
                angular_radius = np.arccos(np.clip(p @ a, -1.0, 1.0))
                if angular_radius > max_radius:
                    max_radius = angular_radius
                    worst_circumcenter = p
                    worst_simplex = simplex
                break
    
    return worst_circumcenter, worst_simplex, max_radius


def _optimize_from_start(start_pts, max_time=30.0):
    """Optimize a starting configuration using gradient-like steps toward worst facets."""
    start_time = time.time()
    pts = _normalize(start_pts.copy())
    current_radius = _covering_radius_fast(pts)
    best_pts = pts.copy()
    best_radius = current_radius
    
    step_size = 0.05
    n_iter = 0
    
    while time.time() - start_time < max_time:
        circumcenter, simplex, radius = _find_worst_facet(pts)
        if circumcenter is None:
            break
        
        # Move points in the worst facet toward the circumcenter
        # to reduce the circumradius
        if simplex is not None:
            for idx in simplex:
                # Move point idx toward circumcenter
                diff = circumcenter - pts[idx]
                pts[idx] += step_size * diff
                # Normalize
                norm = np.linalg.norm(pts[idx])
                if norm > 1e-12:
                    pts[idx] /= norm
        
        # Also do a small random perturbation to escape local minima
        if n_iter % 10 == 5:
            noise = np.random.randn(len(pts), 3) * 0.01
            pts = _normalize(pts + noise)
        
        new_radius = _covering_radius_fast(pts)
        n_iter += 1
        
        if new_radius < current_radius:
            current_radius = new_radius
            if new_radius < best_radius:
                best_radius = new_radius
                best_pts = pts.copy()
            # Reduce step size slowly
            step_size = max(0.001, step_size * 0.995)
        else:
            # Revert and try smaller step
            pts = _normalize(start_pts.copy()) if new_radius > current_radius * 1.1 else pts
            step_size = max(0.0005, step_size * 0.8)
            # If stuck, perturb
            if n_iter % 20 == 15:
                idx_to_move = np.random.randint(len(pts))
                perturb = np.random.randn(3) * 0.05
                pts[idx_to_move] = _normalize(pts[idx_to_move] + perturb)[0]
    
    return best_pts, best_radius


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n):
    """Generate random points on the sphere."""
    pts = np.random.randn(n, 3)
    return _normalize(pts)


def _icosahedron_base():
    """Generate icosahedron vertices as a starting point."""
    t = (1 + np.sqrt(5)) / 2
    pts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    return _normalize(pts)


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    global_time_start = time.time()
    
    # Generate multiple starting configurations
    starts = []
    
    # Fibonacci sphere
    starts.append(_fibonacci_sphere(N))
    
    # Random starts
    for _ in range(5):
        starts.append(_random_sphere(N))
    
    # Icosahedron-based (expand to 50 points)
    ico = _icosahedron_base()
    # Add more points by subdividing
    expanded = ico.copy()
    while len(expanded) < N:
        # Find the pair with largest distance and add midpoint
        n = len(expanded)
        max_dist = 0
        best_pair = (0, 1)
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(expanded[i] - expanded[j])
                if d > max_dist:
                    max_dist = d
                    best_pair = (i, j)
        mid = _normalize((expanded[best_pair[0]] + expanded[best_pair[1]]) / 2)
        expanded = np.vstack([expanded, mid])
    expanded = expanded[:N]
    starts.append(expanded)
    
    # Optimize each start
    best_overall = None
    best_radius_overall = np.pi
    
    for i, start in enumerate(starts):
        elapsed = time.time() - global_time_start
        remaining = TIME_LIMIT - elapsed - 5  # 5s safety margin
        if remaining < 5:
            break
        
        time_per_start = max(5.0, remaining / (len(starts) - i))
        try:
            result_pts, result_radius = _optimize_from_start(start, max_time=time_per_start)
            if result_radius < best_radius_overall:
                best_radius_overall = result_radius
                best_overall = result_pts
        except Exception:
            continue
    
    # Final polish: try to improve the best result further
    if best_overall is not None and time.time() - global_time_start < TIME_LIMIT - 10:
        polish_time = TIME_LIMIT - time.time() - 5
        if polish_time > 5:
            try:
                polished_pts, polished_radius = _optimize_from_start(best_overall, max_time=polish_time)
                if polished_radius < best_radius_overall:
                    best_radius_overall = polished_radius
                    best_overall = polished_pts
            except Exception:
                pass
    
    if best_overall is None:
        best_overall = _fibonacci_sphere(N)
    
    return _normalize(best_overall)


# EVOLVE-BLOCK-START
# (The above code is the main implementation)
# EVOLVE-BLOCK-END
