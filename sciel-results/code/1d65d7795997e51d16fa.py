import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

def _triangle_indices(n=13):
    """Precompute all triangle index combinations."""
    return np.array(list(combinations(range(n), 3)))

_TRI_INDICES = _triangle_indices(13)

def _triangle_areas(points):
    """Compute areas of all triangles formed by 3 points. Returns array of shape (286,)."""
    n = len(points)
    idx = _TRI_INDICES
    
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    cross = (p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
             p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
             p3[:, 0] * (p1[:, 1] - p2[:, 1]))
    areas = 0.5 * np.abs(cross)
    return areas

def _score(points):
    """Compute A_min / A_hull. Returns 0 if invalid."""
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

def _subgradient_min_area(points):
    """
    Compute subgradient of -min_triangle_area with respect to point coordinates.
    Returns a flat array of shape (26,) representing d(-A_min)/d(x).
    The subgradient of min is the negative of the gradient of the active (minimum) triangle.
    """
    n = len(points)
    idx = _TRI_INDICES
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    # cross = x1(y2-y3) + x2(y3-y1) + x3(y1-y2)
    # area = 0.5 * |cross|
    cross = (p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
             p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
             p3[:, 0] * (p1[:, 1] - p2[:, 1]))
    
    # For the active minimum triangle, we need the sign of cross
    # If cross > 0, area = 0.5*cross, d(area)/d(points) is straightforward
    # If cross < 0, area = -0.5*cross
    
    min_idx = np.argmin(np.abs(cross))
    sign = np.sign(cross[min_idx])
    if sign == 0:
        sign = 1.0
    
    # Gradient of 0.5 * sign * cross with respect to each point
    # cross = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
    # d(cross)/dx1 = (y2-y3), d(cross)/dy1 = x3-x2
    # d(cross)/dx2 = (y3-y1), d(cross)/dy2 = x1-x3
    # d(cross)/dx3 = (y1-y2), d(cross)/dy3 = x2-x1
    
    i, j, k = idx[min_idx]
    grad = np.zeros((n, 2))
    
    # For point i (p1):
    grad[i, 0] = 0.5 * sign * (p2[min_idx, 1] - p3[min_idx, 1])
    grad[i, 1] = 0.5 * sign * (p3[min_idx, 0] - p2[min_idx, 0])
    
    # For point j (p2):
    grad[j, 0] = 0.5 * sign * (p3[min_idx, 1] - p1[min_idx, 1])
    grad[j, 1] = 0.5 * sign * (p1[min_idx, 0] - p3[min_idx, 0])
    
    # For point k (p3):
    grad[k, 0] = 0.5 * sign * (p1[min_idx, 1] - p2[min_idx, 1])
    grad[k, 1] = 0.5 * sign * (p2[min_idx, 0] - p1[min_idx, 0])
    
    return grad

def _soft_min(areas, beta):
    """Soft minimum using log-sum-exp: -log(sum(exp(-beta*areas)))/beta approximates min for large beta."""
    # For numerical stability
    min_val = np.min(areas)
    exp_terms = np.exp(-beta * (areas - min_val))
    log_sum = np.log(np.sum(exp_terms))
    return min_val - np.log(log_sum) / beta

def _epigraph_objective(x, beta):
    """
    Epigraph formulation: minimize t such that t >= all triangle areas.
    We minimize t - lambda * (sum of soft-violations) to encourage all areas >= t.
    Actually, we want to MAXIMIZE t subject to all areas >= t.
    Equivalent: minimize -t + penalty for areas < t.
    
    Better: use the soft-min directly as a smooth approximation.
    For large beta, soft_min(areas, beta) -> min(areas).
    We maximize soft_min(areas, beta) / hull_area.
    """
    pts = x.reshape(13, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    
    areas = _triangle_areas(pts)
    a_min_soft = _soft_min(areas, beta)
    if a_min_soft < 1e-15:
        return 1e10
    
    try:
        hull = ConvexHull(pts)
        a_hull = hull.volume
        if a_hull < 1e-15:
            return 1e10
        score = a_min_soft / a_hull
        return -score
    except Exception:
        return 1e10

def _high_precision_polish(points, deadline=None):
    """
    High-precision polish using SLSQP with tight tolerances.
    Maximizes A_min / A_hull via the soft-min surrogate with very large beta.
    """
    n = 13
    best_points = points.copy()
    best_score = _score(points)
    
    # Use very high beta for near-exact min approximation
    for beta in [1000, 2000, 5000]:
        if deadline is not None and time.time() > deadline - 30:
            break
        
        def obj(x):
            pts = x.reshape(n, 2)
            if np.any(~np.isfinite(pts)):
                return 1e10
            areas = _triangle_areas(pts)
            a_min_soft = _soft_min(areas, beta)
            if a_min_soft < 1e-15:
                return 1e10
            try:
                hull = ConvexHull(pts)
                a_hull = hull.volume
                if a_hull < 1e-15:
                    return 1e10
                return -(a_min_soft / a_hull)
            except Exception:
                return 1e10
        
        x0 = best_points.flatten()
        try:
            result = minimize(obj, x0, method='SLSQP',
                            options={'maxiter': 1000, 'ftol': 1e-14})
            new_points = result.x.reshape(n, 2)
            new_score = _score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
        except Exception:
            pass
        
        # Also try Nelder-Mead with tight tolerances
        try:
            result = minimize(obj, x0, method='nelder-mead',
                            options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
            new_points = result.x.reshape(n, 2)
            new_score = _score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
        except Exception:
            pass
    
    return best_points, best_score

def _subgradient_ascent(points, steps=2000, lr_init=0.01, lr_decay=0.999, deadline=None):
    """
    Subgradient ascent on A_min / A_hull.
    We maximize the ratio, so we do subgradient ascent on A_min while keeping track of A_hull.
    Actually, let's maximize A_min - lambda * A_hull or use the ratio directly.
    
    Simpler: maximize A_min / A_hull. The subgradient of the ratio is complex.
    Instead, let's maximize A_min (since the ratio is scale-invariant, we can fix the hull area
    by normalizing, or just maximize A_min and normalize at the end).
    
    Better approach: Use the fact that the objective is scale-invariant.
    We can fix one point or normalize the configuration.
    
    Let's use: maximize A_min subject to A_hull = 1 (or some fixed value).
    Or simply: do subgradient ascent on A_min, then normalize.
    """
    best_points = points.copy()
    best_score = _score(points)
    lr = lr_init
    
    for step in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        
        areas = _triangle_areas(points)
        a_min = np.min(areas)
        
        # Subgradient of A_min w.r.t. points
        grad = _subgradient_min_area(points)
        
        # Normalize gradient to have unit norm for stable step size
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-15:
            grad = grad / grad_norm
        
        # Update
        points = points + lr * grad
        
        # Keep points in a reasonable range
        # Normalize to keep hull area approximately constant
        try:
            hull = ConvexHull(points)
            a_hull = hull.volume
            if a_hull > 1e-10:
                # Scale to keep area ~1
                scale = 1.0 / np.sqrt(a_hull)
                points = points * scale
        except Exception:
            pass
        
        # Check score
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        lr *= lr_decay
    
    return best_points, best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    deadline = time.time() + 580  # Safety margin
    
    # Start with regular 13-gon as initial guess
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = _score(best_points)
    
    def _objective_flat(x):
        """Objective function for optimization (to minimize negative score)."""
        pts = x.reshape(n, 2)
        if np.any(~np.isfinite(pts)):
            return 1e10
        
        areas = _triangle_areas(pts)
        a_min = np.min(areas)
        if a_min < 1e-15:
            return 1e10
        
        try:
            hull = ConvexHull(pts)
            a_hull = hull.volume
            if a_hull < 1e-15:
                return 1e10
            score = a_min / a_hull
            return -score
        except Exception:
            return 1e10
    
    def _objective_smooth(x, beta):
        """Smooth objective using soft-min."""
        pts = x.reshape(n, 2)
        if np.any(~np.isfinite(pts)):
            return 1e10
        
        areas = _triangle_areas(pts)
        a_min_soft = _soft_min(areas, beta)
        if a_min_soft < 1e-15:
            return 1e10
        
        try:
            hull = ConvexHull(pts)
            a_hull = hull.volume
            if a_hull < 1e-15:
                return 1e10
            score = a_min_soft / a_hull
            return -score
        except Exception:
            return 1e10
    
    # Phase 1: Subgradient ascent from regular polygon
    if time.time() < deadline - 300:
        sg_points, sg_score = _subgradient_ascent(
            best_points.copy(), steps=3000, lr_init=0.05, lr_decay=0.9995,
            deadline=deadline - 250
        )
        if sg_score > best_score:
            best_score = sg_score
            best_points = sg_points
    
    # Phase 2: Nelder-Mead refinement
    if time.time() < deadline - 200:
        x0 = best_points.flatten()
        try:
            result = minimize(_objective_flat, x0, method='nelder-mead',
                            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            new_points = result.x.reshape(n, 2)
            new_score = _score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
        except Exception:
            pass
    
    # Phase 3: Soft-min optimization with increasing beta
    for beta in [20, 50, 100, 200, 500, 1000, 2000]:
        if time.time() > deadline - 150:
            break
        x0 = best_points.flatten()
        try:
            result = minimize(
                lambda x, b=beta: _objective_smooth(x, b),
                x0, method='nelder-mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            new_points = result.x.reshape(n, 2)
            new_score = _score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
        except Exception:
            pass
    
    # Phase 4: Multiple diverse starting points with subgradient + NM
    rng = np.random.default_rng(123)
    diverse_starts = []
    
    # Regular polygon with different rotations
    for rot in [0, np.pi/13, 2*np.pi/13, np.pi/7]:
        a = angles + rot
        diverse_starts.append(np.column_stack((np.cos(a), np.sin(a))))
    
    # Grid-like arrangements
    for r in range(3):
        pts = rng.uniform(-1, 1, size=(n, 2))
        # Ensure convex hull is valid
        try:
            ConvexHull(pts)
            diverse_starts.append(pts)
        except Exception:
            pass
    
    # Perturbed best points
    for r in range(3):
        pts = best_points + rng.normal(0, 0.3, size=(n, 2))
        try:
            ConvexHull(pts)
            diverse_starts.append(pts)
        except Exception:
            pass
    
    for start_pts in diverse_starts:
        if time.time() > deadline - 100:
            break
        
        # Subgradient ascent
        if time.time() < deadline - 80:
            sg_pts, sg_sc = _subgradient_ascent(
                start_pts.copy(), steps=1000, lr_init=0.03, lr_decay=0.9995,
                deadline=deadline - 60
            )
            if sg_sc > best_score:
                best_score = sg_sc
                best_points = sg_pts
        
        # NM refinement
        if time.time() < deadline - 40:
            x0 = best_points.flatten()
            try:
                result = minimize(_objective_flat, x0, method='nelder-mead',
                                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
                new_points = result.x.reshape(n, 2)
                new_score = _score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points
            except Exception:
                pass
    
    # Final high-precision polish
    if time.time() < deadline - 30:
        polished_points, polished_score = _high_precision_polish(
            best_points.copy(), deadline=deadline - 10
        )
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_points
    
    # Ensure the result is valid
    if _score(best_points) <= 0:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_points
