# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _normalize(polygon):
    """Subtract centroid and divide by mean distance to centroid."""
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = dists.mean()
    if mean_dist <= 0.01:
        return centered, mean_dist
    return centered / mean_dist, mean_dist

def _check_strictly_convex(polygon):
    """Check if polygon is strictly convex."""
    n = len(polygon)
    cross_products = []
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-9:
            return False
        cross_products.append(cross)
    
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return False
        
    angles = []
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        a1 = np.arctan2(v1[1], v1[0])
        a2 = np.arctan2(v2[1], v2[0])
        da = a2 - a1
        while da > np.pi:
            da -= 2 * np.pi
        while da < -np.pi:
            da += 2 * np.pi
        angles.append(da)
    total_turn = sum(angles)
    if abs(abs(total_turn) - 2 * np.pi) > 1e-4:
        return False
    return True

def _vertex_score(v_idx, polygon):
    """Compute the vertex score for vertex at v_idx in polygon."""
    n = len(polygon)
    v = polygon[v_idx]
    diffs = polygon - v
    dists = np.linalg.norm(diffs, axis=1)
    dists[v_idx] = np.inf
    other_dists = dists[~np.isinf(dists)]
    sorted_dists = np.sort(other_dists)
    
    if sorted_dists[0] < 0.05:
        return 0.0
        
    gaps = np.diff(sorted_dists)
    k = np.argmin(gaps)
    D = (sorted_dists[k] + sorted_dists[k + 1]) / 2
    if D < 0.1:
        return 0.0
        
    if k == 0:
        selected = sorted_dists[0:4]
    elif k == 17:
        selected = sorted_dists[15:19]
    else:
        selected = sorted_dists[k - 1:k + 3]
        
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)

def _score_polygon(polygon):
    """Compute the minimum vertex score over all vertices."""
    polygon, mean_dist = _normalize(polygon)
    if not _check_strictly_convex(polygon):
        return 0.0
    scores = [_vertex_score(i, polygon) for i in range(len(polygon))]
    return min(scores)

def _optimize_polygon(start_time, max_time=600.0):
    n = 20
    deadline = start_time + max_time - 30.0
    
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    best_polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    best_score = _score_polygon(best_polygon)
    
    def objective(params):
        base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        angles = base_angles + params
        angles = angles - np.mean(angles)
        angles = np.sort(angles)
        angles = angles % (2 * np.pi)
        angles = np.sort(angles)
        
        if np.any(np.diff(np.concatenate([angles, [angles[0] + 2*np.pi]])) < 1e-6):
            return 0.0
            
        x = np.cos(angles)
        y = np.sin(angles)
        polygon = np.stack([x, y], axis=1)
        
        score = _score_polygon(polygon)
        return -score
    
    np.random.seed(42)
    
    for trial in range(5):
        if time.time() > deadline:
            break
            
        if trial == 0:
            x0 = np.zeros(n)
        else:
            x0 = np.random.randn(n) * 0.1
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            if res.fun < -best_score:
                best_score = -res.fun
                base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
                angles = base_angles + res.x
                angles = angles - np.mean(angles)
                angles = np.sort(angles)
                angles = angles % (2 * np.pi)
                angles = np.sort(angles)
                best_polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        except Exception:
            pass

    return best_polygon, best_score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    polygon, score = _optimize_polygon(start_time)
    return polygon

# EVOLVE-BLOCK-END
