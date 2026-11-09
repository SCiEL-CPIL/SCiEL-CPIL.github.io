import numpy as np
import time
from scipy.spatial import ConvexHull

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

def _generate_regular_polygon(n):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _generate_halton_sequence(n, dims=2):
    pts = np.zeros((n, dims))
    for i in range(n):
        for dim in range(dims):
            f = 1.0
            num = i + 1
            base = dim + 2
            while num > 0:
                num, rem = divmod(num, base)
                f *= base
                pts[i, dim] += rem / f
    return pts

def _normalize_points(pts):
    """Normalize points to have unit hull area and center at origin."""
    hull = ConvexHull(pts)
    a_hull = hull.volume
    if a_hull <= 0:
        return pts
    # Center at origin
    pts_centered = pts - np.mean(pts, axis=0)
    # Scale to unit hull area
    pts_scaled = pts_centered * np.sqrt(1.0 / a_hull)
    return pts_scaled

def _gradient_ascent(pts, max_iter=5000, lr=0.01, deadline=None):
    """Gradient ascent on the min-triangle-area ratio."""
    n = len(pts)
    # Precompute triangle indices
    indices = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        pts = pts.copy()
        areas = _triangle_areas(pts)
        a_min = np.min(areas)
        
        if a_min <= 0:
            # Try to fix by moving points apart
            for i in range(n):
                for j in range(n):
                    if i != j:
                        diff = pts[i] - pts[j]
                        dist = np.linalg.norm(diff)
                        if dist < 1e-6:
                            pts[i] += rng.normal(0, 0.01, 2)
            continue
        
        # Find which triangles have area close to minimum
        threshold = a_min * 1.01
        min_triangles = np.where(areas <= threshold)[0]
        
        # Compute gradient with respect to each point
        grad = np.zeros((n, 2))
        
        for tri_idx in min_triangles:
            i, j, k = indices[tri_idx]
            p1, p2, p3 = pts[i], pts[j], pts[k]
            
            # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
            # Gradient of area w.r.t. p1: (-(y2-y3)/2, (x2-x3)/2)
            # Gradient w.r.t. p2: ((y3-y1)/2, -(x3-x1)/2)
            # Gradient w.r.t. p3: (-(y1-y2)/2, (x1-x2)/2)
            
            # Sign based on orientation
            cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
            sign = np.sign(cross) if cross != 0 else 1.0
            
            # Gradient contributions
            g1 = np.array([-(p2[1]-p3[1]), p2[0]-p3[0]]) * 0.5 * sign
            g2 = np.array([(p3[1]-p1[1]), -(p3[0]-p1[0])]) * 0.5 * sign
            g3 = np.array([-(p1[1]-p2[1]), p1[0]-p2[0]]) * 0.5 * sign
            
            grad[i] += g1
            grad[j] += g2
            grad[k] += g3
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad /= grad_norm
        
        # Update points
        pts = pts + lr * grad
        
        # Keep points in a reasonable range
        pts = np.clip(pts, -10, 10)
    
    return pts

def heilbronn_convex13() -> np.ndarray:
    deadline = time.time() + 580  # Safety margin
    rng = np.random.default_rng(42)
    
    # Initial candidates
    candidates = []
    candidates.append(_generate_regular_polygon(13))
    
    # Halton sequence
    candidates.append(_generate_halton_sequence(13))
    
    # Random candidates
    for _ in range(20):
        pts = rng.uniform(-2, 2, (13, 2))
        candidates.append(pts)
    
    # Evaluate all candidates
    best_pts = candidates[0]
    best_score = _score(best_pts)
    for c in candidates[1:]:
        s = _score(c)
        if s > best_score:
            best_score = s
            best_pts = c
    
    # Multi-start gradient ascent
    starts = []
    for c in candidates:
        starts.append(c.copy())
        # Add perturbed versions
        for _ in range(3):
            noise = rng.normal(0, 0.2, (13, 2))
            starts.append(c + noise)
    
    rng.shuffle(starts)
    
    for start in starts:
        if time.time() > deadline - 30:
            break
        
        try:
            # Normalize starting points
            start_norm = _normalize_points(start)
            
            # Run gradient ascent
            pts_opt = _gradient_ascent(start_norm, max_iter=3000, lr=0.02, deadline=deadline)
            
            # Also try without normalization
            pts_opt2 = _gradient_ascent(start, max_iter=3000, lr=0.02, deadline=deadline)
            
            for pts_try in [pts_opt, pts_opt2]:
                score_try = _score(pts_try)
                if score_try > best_score:
                    best_score = score_try
                    best_pts = pts_try.copy()
        except Exception:
            pass
    
    # Final fallback
    if _score(best_pts) <= 0:
        best_pts = _generate_regular_polygon(13)
    
    return best_pts
