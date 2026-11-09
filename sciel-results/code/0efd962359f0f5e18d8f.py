# EVOLVE-BLOCK-START
import numpy as np
import time


def _normalize(polygon):
    """Subtract centroid and divide by mean distance to centroid."""
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = dists.mean()
    if mean_dist <= 0.01:
        return centered
    return centered / mean_dist


def _is_valid_convex(polygon):
    """Check strictly convex: no repeated points, all turns same sign nonzero, total turning +-2*pi."""
    n = len(polygon)
    if n < 3:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(polygon[i] - polygon[j]) < 1e-10:
                return False
    cross_prods = []
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        cross_prods.append(cross)
    cross_prods = np.array(cross_prods)
    if np.any(np.abs(cross_prods) < 1e-12):
        return False
    if np.any(cross_prods > 0) and np.any(cross_prods < 0):
        return False
    total_turn = 0.0
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        angle1 = np.arctan2(e1[1], e1[0])
        angle2 = np.arctan2(e2[1], e2[0])
        d_angle = angle2 - angle1
        while d_angle > np.pi:
            d_angle -= 2 * np.pi
        while d_angle < -np.pi:
            d_angle += 2 * np.pi
        total_turn += d_angle
    if abs(abs(total_turn) - 2 * np.pi) > 1e-6:
        return False
    return True


def _vertex_score(vertex, polygon):
    """Compute the score for a single vertex."""
    n = len(polygon)
    diffs = polygon - vertex
    dists = np.linalg.norm(diffs, axis=1)
    idx = np.argmin(dists)
    dists[idx] = np.inf
    dists_sorted = np.sort(dists)
    if np.any(dists_sorted < 0.05):
        return 0.0
    gaps = dists_sorted[1:] - dists_sorted[:-1]
    k = np.argmin(gaps)
    D = (dists_sorted[k] + dists_sorted[k + 1]) / 2
    if D < 0.1:
        return 0.0
    if k == 0:
        four_dists = dists_sorted[0:4]
    elif k == 17:
        four_dists = dists_sorted[15:19]
    else:
        four_dists = dists_sorted[k - 1:k + 3]
    ratios = np.maximum(four_dists / D, D / four_dists)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)


def _polygon_score(polygon):
    """Compute the minimum vertex score over all vertices."""
    n = len(polygon)
    scores = []
    for i in range(n):
        scores.append(_vertex_score(polygon[i], polygon))
    return min(scores)


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices where every vertex has approximately
    4 other vertices equidistant from it.
    """
    start_time = time.time()
    deadline = start_time + 590.0
    
    n = 20
    
    # Start with regular 20-gon
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    polygon = _normalize(polygon)
    
    best_polygon = polygon.copy()
    best_score = _polygon_score(polygon)
    
    rng = np.random.default_rng(42)
    
    # Multi-scale local search with decreasing step sizes
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    for step in step_sizes:
        if time.time() > deadline - 5.0:
            break
        
        # Number of iterations proportional to 1/step, capped
        num_iters = int(min(500, max(50, 2.0 / step)))
        
        for _ in range(num_iters):
            if time.time() > deadline - 2.0:
                break
            
            vi = rng.integers(0, n)
            delta = rng.normal(0, step, 2)
            
            candidate = best_polygon.copy()
            candidate[vi] += delta
            candidate = _normalize(candidate)
            
            if not _is_valid_convex(candidate):
                continue
            
            cand_score = _polygon_score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_polygon = candidate
    
    best_polygon = _normalize(best_polygon)
    if not _is_valid_convex(best_polygon):
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        best_polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        best_polygon = _normalize(best_polygon)
    
    return best_polygon


# EVOLVE-BLOCK-END
