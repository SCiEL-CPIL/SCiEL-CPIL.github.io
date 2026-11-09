import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

def _triangle_areas(points):
    n = len(points)
    indices = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas

def _score(points):
    try:
        if np.any(~np.isfinite(points)):
            return 0.0
        if len(points) != 13 or points.shape != (13, 2):
            return 0.0
        hull = ConvexHull(points)
        a_hull = hull.volume
        if a_hull <= 0:
            return 0.0
        areas = _triangle_areas(points)
        a_min = np.min(areas)
        if a_min <= 0:
            return 0.0
        return a_min / a_hull
    except Exception:
        return 0.0

def _soft_min_area(points, beta=50.0):
    """Compute a smooth approximation of the minimum triangle area using log-sum-exp."""
    areas = _triangle_areas(points)
    # Soft min: (1/beta) * log(sum(exp(-beta * areas)))
    # For numerical stability, subtract max
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    return -log_sum / beta

def _smooth_objective(x, beta=50.0):
    """
    Smooth objective for optimization: minimize -soft_min_area / hull_area
    Using a smooth surrogate for the nonsmooth min operation.
    """
    pts = x.reshape(13, 2)
    if not np.all(np.isfinite(pts)):
        return 1e6
    
    try:
        hull = ConvexHull(pts)
        a_hull = hull.volume
        if a_hull <= 0:
            return 1e6
        
        soft_min = _soft_min_area(pts, beta)
        if soft_min <= 0:
            return 1e6
        
        return -soft_min / a_hull
    except Exception:
        return 1e6

def _generate_structured_points(r2, theta):
    r1 = 1.0
    pts = np.zeros((13, 2))
    pts[0] = [0, 0]
    for i in range(6):
        angle = i * np.pi / 3.0
        pts[i+1] = [r1 * np.cos(angle), r1 * np.sin(angle)]
    for i in range(6):
        angle = i * np.pi / 3.0 + theta
        pts[i+7] = [r2 * np.cos(angle), r2 * np.sin(angle)]
    return pts

def _generate_regular_polygon(n):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _find_min_triangle(points):
    """Find indices of the minimum-area triangle."""
    areas = _triangle_areas(points)
    min_idx = np.argmin(areas)
    indices = np.array([(i, j, k) for i in range(13) for j in range(i+1, 13) for k in range(j+1, 13)])
    return indices[min_idx]

def _geometric_ascent(points, max_iter=500, deadline=None):
    """
    Targeted geometric ascent: identify the min-area triangle and move its vertices
    to increase that area while maintaining overall configuration validity.
    """
    pts = points.copy()
    best_score = _score(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Find the minimum area triangle
        min_tri = _find_min_triangle(pts)
        i, j, k = min_tri
        
        # Get the three points
        p1, p2, p3 = pts[i], pts[j], pts[k]
        
        # Compute the centroid of the min triangle
        centroid = (p1 + p2 + p3) / 3.0
        
        # For each vertex of the min triangle, try moving it away from the other two
        # to increase the area of this specific triangle
        best_move = None
        best_move_score = best_score
        
        # Try moving each vertex in several directions
        for vi, vj in [(i, j), (i, k), (j, i), (j, k), (k, i), (k, j)]:
            p_v = pts[vi].copy()
            p_other = pts[vj]
            
            # Direction away from the other vertex
            direction = p_v - p_other
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            direction = direction / norm
            
            # Try several step sizes
            for step in [0.01, 0.05, 0.1, 0.2, 0.5]:
                new_pts = pts.copy()
                new_pts[vi] = p_v + step * direction
                
                # Check validity
                if not np.all(np.isfinite(new_pts)):
                    continue
                try:
                    hull = ConvexHull(new_pts)
                    if hull.volume <= 0:
                        continue
                except Exception:
                    continue
                
                new_score = _score(new_pts)
                if new_score > best_move_score + 1e-10:
                    best_move_score = new_score
                    best_move = new_pts
        
        if best_move is not None:
            pts = best_move
            best_score = best_move_score
        else:
            # No improvement found with simple moves, try random perturbations of min triangle vertices
            improved = False
            for _ in range(10):
                trial = pts.copy()
                for vi in [i, j, k]:
                    trial[vi] += np.random.randn(2) * 0.1
                if np.all(np.isfinite(trial)):
                    try:
                        hull = ConvexHull(trial)
                        if hull.volume > 0:
                            s = _score(trial)
                            if s > best_score + 1e-10:
                                pts = trial
                                best_score = s
                                improved = True
                                break
                    except Exception:
                        pass
            
            if not improved:
                # Try small random moves of all points
                trial = pts + np.random.randn(13, 2) * 0.05
                if np.all(np.isfinite(trial)):
                    try:
                        hull = ConvexHull(trial)
                        if hull.volume > 0:
                            s = _score(trial)
                            if s > best_score + 1e-10:
                                pts = trial
                                best_score = s
                    except Exception:
                        pass
    
    return pts, best_score

def _smooth_optimize(points, deadline=None, beta_start=10.0, beta_end=100.0):
    """
    Use smooth surrogate optimization with annealed beta for the soft-min.
    This allows gradient-based methods to navigate the nonsmooth landscape.
    """
    from scipy.optimize import minimize
    
    pts = points.copy()
    best_score = _score(pts)
    best_pts = pts.copy()
    
    # Anneal beta from low to high for increasing precision
    n_steps = 5
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        beta = beta_start + (beta_end - beta_start) * step / (n_steps - 1)
        
        def obj(x):
            return _smooth_objective(x, beta)
        
        def grad(x):
            # Numerical gradient
            eps = 1e-7
            g = np.zeros_like(x)
            for i in range(len(x)):
                xp = x.copy()
                xm = x.copy()
                xp[i] += eps
                xm[i] -= eps
                g[i] = (obj(xp) - obj(xm)) / (2 * eps)
            return g
        
        try:
            res = minimize(obj, pts.flatten(), method='L-BFGS-B', jac=grad,
                          options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-9})
            new_pts = res.x.reshape(13, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                pts = new_pts.copy()
        except Exception:
            pass
    
    return best_pts, best_score

def heilbronn_convex13() -> np.ndarray:
    deadline = time.time() + 580
    
    # Multiple diverse initializations
    candidates = []
    
    # Structured: center + 2 hexagons with various parameters
    for r2 in [1.5, 2.0, 2.5, 3.0]:
        for theta in [np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi/5]:
            candidates.append(_generate_structured_points(r2, theta))
    
    # Regular 13-gon
    candidates.append(_generate_regular_polygon(13))
    
    # Regular 12-gon + center
    pts_12 = _generate_regular_polygon(12)
    pts_12_center = np.vstack([np.array([[0, 0]]), pts_12])
    candidates.append(pts_12_center)
    
    # Two concentric hexagons with offset
    for r1, r2, offset in [(1.0, 2.0, 0.1), (1.0, 2.5, 0.2), (1.2, 2.2, 0.15)]:
        pts = np.zeros((13, 2))
        pts[0] = [offset, 0]
        for i in range(6):
            angle = i * np.pi / 3.0
            pts[i+1] = pts[0] + r1 * np.array([np.cos(angle), np.sin(angle)])
        for i in range(6):
            angle = i * np.pi / 3.0 + np.pi/6
            pts[i+7] = pts[0] + r2 * np.array([np.cos(angle), np.sin(angle)])
        candidates.append(pts)
    
    # Evaluate all candidates
    best_pts = candidates[0]
    best_score = _score(best_pts)
    for cand in candidates[1:]:
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_pts = cand
    
    # Phase 1: Geometric ascent from best candidate
    if time.time() < deadline - 120:
        try:
            pts_asc, score_asc = _geometric_ascent(best_pts, max_iter=300, deadline=deadline - 120)
            if score_asc > best_score:
                best_score = score_asc
                best_pts = pts_asc
        except Exception:
            pass
    
    # Phase 2: Smooth surrogate optimization with annealed beta
    if time.time() < deadline - 60:
        try:
            pts_smooth, score_smooth = _smooth_optimize(best_pts, deadline=deadline - 60)
            if score_smooth > best_score:
                best_score = score_smooth
                best_pts = pts_smooth
        except Exception:
            pass
    
    # Phase 2b: Nelder-Mead refinement on the best point
    if time.time() < deadline - 30:
        def _obj(x):
            pts = x.reshape(13, 2)
            return -_score(pts)
        
        try:
            res = minimize(_obj, best_pts.flatten(), method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
            if res.fun < -best_score:
                pts_opt = res.x.reshape(13, 2)
                score_opt = _score(pts_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_pts = pts_opt
        except Exception:
            pass
    
    # Phase 3: Try geometric ascent again from refined point
    if time.time() < deadline - 20:
        try:
            pts_asc2, score_asc2 = _geometric_ascent(best_pts, max_iter=100, deadline=deadline - 20)
            if score_asc2 > best_score:
                best_score = score_asc2
                best_pts = pts_asc2
        except Exception:
            pass
    
    # Phase 4: Final smooth polish with high beta
    if time.time() < deadline - 5:
        try:
            def _obj_high_beta(x):
                return _smooth_objective(x, beta=200.0)
            
            def _grad_high_beta(x):
                eps = 1e-8
                g = np.zeros_like(x)
                for i in range(len(x)):
                    xp = x.copy()
                    xm = x.copy()
                    xp[i] += eps
                    xm[i] -= eps
                    g[i] = (_obj_high_beta(xp) - _obj_high_beta(xm)) / (2 * eps)
                return g
            
            res = minimize(_obj_high_beta, best_pts.flatten(), method='L-BFGS-B', 
                          jac=_grad_high_beta, options={'maxiter': 100, 'gtol': 1e-8})
            new_pts = res.x.reshape(13, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
        except Exception:
            pass
    
    # Final safety check
    final_score = _score(best_pts)
    if final_score <= 0:
        best_pts = _generate_regular_polygon(13)
    
    return best_pts
