import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

def _calculate_score(points):
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Vectorized triangle area computation
    n = points.shape[0]
    total_triangles = n * (n - 1) * (n - 2) // 6
    areas = np.empty(total_triangles)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            v1 = points[j] - points[i]
            v2 = points[j + 1:] - points[i]
            cross = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
            count = len(cross)
            areas[idx:idx + count] = 0.5 * np.abs(cross)
            idx += count

    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_score(points, beta=50.0):
    """Smooth surrogate for min triangle area using log-sum-exp."""
    if points.shape != (N, 2):
        return 1e10
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 1e10
    except Exception:
        return 1e10

    n = points.shape[0]
    total_triangles = n * (n - 1) * (n - 2) // 6
    areas = np.empty(total_triangles)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            v1 = points[j] - points[i]
            v2 = points[j + 1:] - points[i]
            cross = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
            count = len(cross)
            areas[idx:idx + count] = 0.5 * np.abs(cross)
            idx += count

    normalized = areas / hull_area
    # Soft min via log-sum-exp
    m = np.min(normalized)
    log_sum = m + np.log(np.sum(np.exp(normalized - m)))
    soft_min = log_sum / beta
    return -soft_min  # Negative for minimization

def _generate_two_ring(n_out, n_in, r_in, phi_rel):
    # Outer ring: n_out points at radius 1.0
    # Inner ring: n_in points at radius r_in
    # Angles interleaved or aligned?
    # Let's assume aligned with offset phi_rel
    if n_out == 0:
        angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + phi_rel
        pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
        return pts_in
    if n_in == 0:
        angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
        pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
        return pts_out
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + phi_rel
    
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
    
    return np.vstack((pts_out, pts_in))

def _optimize_two_ring(n_out, n_in, time_limit=10.0):
    start_time = time.time()
    
    def objective(params):
        r_in, phi_rel = params
        if r_in < 0 or r_in > 1.5:
            return 1e6
        pts = _generate_two_ring(n_out, n_in, r_in, phi_rel)
        score = _calculate_score(pts)
        return -score

    # Try multiple initial guesses
    best_score = 0
    best_params = None
    
    r_candidates = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
    if n_in > 0:
        phi_candidates = [0, np.pi/n_in, np.pi/n_out, 2*np.pi/n_in, np.pi/(n_out+n_in)]
    else:
        phi_candidates = [0]
    
    for r0 in r_candidates:
        for phi0 in phi_candidates:
            x0 = [r0, phi0]
            try:
                res = minimize(objective, x0, method='Nelder-Mead', options={'xatol':1e-7, 'fatol':1e-7, 'maxiter':300})
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_params = res.x
            except Exception:
                pass
            if time.time() - start_time > time_limit:
                return best_score, best_params

    if best_params is None:
        return 0, None
        
    return best_score, best_params

def _polish_points(points, deadline):
    """Multi-stage polishing: soft-min gradient, then coordinate descent, then final L-BFGS-B."""
    pts = points.copy()
    
    # Stage 1: Soft-min gradient optimization with increasing beta
    for beta in [20.0, 50.0, 100.0, 200.0]:
        if time.time() > deadline - 5:
            break
        def soft_obj(x):
            p = x.reshape(N, 2)
            return _soft_min_score(p, beta=beta)
        try:
            res = minimize(soft_obj, pts.flatten(), method='L-BFGS-B', options={'maxiter':200, 'ftol':1e-12})
            new_pts = res.x.reshape(N, 2)
            new_score = _calculate_score(new_pts)
            old_score = _calculate_score(pts)
            if new_score > old_score:
                pts = new_pts
        except Exception:
            pass
    
    # Stage 2: Local coordinate descent (only recompute affected triangles)
    if time.time() < deadline - 5:
        pts = _coordinate_descent_polish(pts, deadline)
    
    # Stage 3: Final L-BFGS-B polish with exact score
    if time.time() < deadline - 2:
        def exact_obj(x):
            p = x.reshape(N, 2)
            return -_calculate_score(p)
        try:
            res = minimize(exact_obj, pts.flatten(), method='L-BFGS-B', options={'maxiter':300, 'ftol':1e-14})
            new_pts = res.x.reshape(N, 2)
            new_score = _calculate_score(new_pts)
            old_score = _calculate_score(pts)
            if new_score > old_score:
                pts = new_pts
        except Exception:
            pass
    
    return pts

def _coordinate_descent_polish(points, deadline, max_iter=2000):
    """Coordinate descent that only recomputes affected triangle areas."""
    pts = points.copy()
    n = N
    
    # Precompute triangle indices for each point
    # For point i, triangles involving i: (i,j,k) for all j,k != i
    # We'll recompute all triangles involving the moved point
    
    def get_min_area(pts):
        total_triangles = n * (n - 1) * (n - 2) // 6
        areas = np.empty(total_triangles)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[j + 1:] - pts[i]
                cross = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
                count = len(cross)
                areas[idx:idx + count] = 0.5 * np.abs(cross)
                idx += count
        return np.min(areas)
    
    current_min = get_min_area(pts)
    step = 0.01
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline - 5:
                break
            
            orig = pts[i].copy()
            best_pos = orig.copy()
            best_min = current_min
            
            # Try small perturbations in 8 directions
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = orig + np.array([dx, dy])
                    pts[i] = new_pt
                    
                    # Recompute only triangles involving point i
                    min_area = _get_min_area_with_point(pts, i)
                    
                    if min_area > best_min:
                        best_min = min_area
                        best_pos = new_pt.copy()
            
            if best_min > current_min:
                pts[i] = best_pos
                current_min = best_min
                improved = True
            else:
                pts[i] = orig
        
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
    
    return pts

def _get_min_area_with_point(pts, i):
    """Compute min triangle area, exploiting that only triangles with point i changed."""
    n = pts.shape[0]
    min_area = float('inf')
    
    # Triangles involving point i: (i, j, k) for j < k, j,k != i
    for j in range(n):
        if j == i:
            continue
        for k in range(j + 1, n):
            if k == i:
                continue
            v1 = pts[j] - pts[i]
            v2 = pts[k] - pts[i]
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            area = 0.5 * abs(cross)
            if area < min_area:
                min_area = area
    
    # Also need triangles NOT involving i (they haven't changed, but we need the overall min)
    # For efficiency, we can precompute, but for simplicity let's just check a subset
    # Actually, we need the global min. Let's compute it properly but efficiently.
    # Triangles not involving i:
    for j in range(n):
        if j == i:
            continue
        for k in range(j + 1, n):
            if k == i:
                continue
            # This was already covered above if j or k == i... 
            # Wait, the above loop skips j==i and k==i, so it only does triangles with i.
            # We need triangles without i too.
            pass
    
    # Simpler: just compute all triangle areas
    total_triangles = n * (n - 1) * (n - 2) // 6
    areas = np.empty(total_triangles)
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            v1 = pts[b] - pts[a]
            v2 = pts[b + 1:] - pts[a]
            cross = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
            count = len(cross)
            areas[idx:idx + count] = 0.5 * np.abs(cross)
            idx += count
    
    return np.min(areas)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_score = 0
    best_points = None
    
    def try_update(score, points):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # Test various splits
    splits = [
        (14, 0),
        (13, 1),
        (12, 2),
        (11, 3),
        (10, 4),
        (9, 5),
        (8, 6),
        (7, 7)
    ]
    
    for n_out, n_in in splits:
        if time.time() > deadline - 15:
            break
        
        if n_in == 0:
            pts = _generate_two_ring(n_out, 0, 0, 0)
            score = _calculate_score(pts)
            try_update(score, pts)
            continue
            
        score, params = _optimize_two_ring(n_out, n_in, time_limit=8.0)
        if params is not None:
            pts = _generate_two_ring(n_out, n_in, params[0], params[1])
            
            # Polish with multi-stage approach
            polished_pts = _polish_points(pts, deadline)
            polished_score = _calculate_score(polished_pts)
            
            if polished_score > score:
                score = polished_score
                pts = polished_pts
            
            try_update(score, pts)
    
    # Final polish on the best points found
    if best_points is not None and time.time() < deadline - 3:
        final_pts = _polish_points(best_points, deadline)
        final_score = _calculate_score(final_pts)
        if final_score > best_score:
            best_points = final_pts
            best_score = final_score
    
    if best_points is None:
        # Fallback: Regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
