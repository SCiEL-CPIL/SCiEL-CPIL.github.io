# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def _score_polygon(polygon):
    """Compute the minimum vertex score for a given polygon."""
    n = len(polygon)
    # Normalize
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_d = dists_to_centroids.mean() if False else np.linalg.norm(centered, axis=1).mean()
    if mean_d <= 0.01:
        return 0.0
    centered = centered / mean_d
    
    # Check convexity: no repeated points, all turns same sign, total turn = ±2π
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(centered[i] - centered[j]) < 1e-10:
                return 0.0
    
    # Compute cross products for turn direction
    cross_products = []
    for i in range(n):
        p0 = centered[i]
        p1 = centered[(i+1) % n]
        p2 = centered[(i+2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross = e1[0]*e2[1] - e1[1]*e2[0]
        cross_products.append(cross)
    
    # All must be nonzero and same sign
    nonzero = [c for c in cross_products if abs(c) > 1e-12]
    if len(nonzero) != n:
        return 0.0
    if any(c > 0 for c in nonzero) and any(c < 0 for c in nonzero):
        return 0.0
    
    # Check total turning angle
    angles = []
    for i in range(n):
        p0 = centered[i]
        p1 = centered[(i+1) % n]
        p2 = centered[(i+2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        norm_e1 = np.linalg.norm(e1)
        norm_e2 = np.linalg.norm(e2)
        if norm_e1 < 1e-12 or norm_e2 < 1e-12:
            return 0.0
        cos_angle = np.dot(e1, e2) / (norm_e1 * norm_e2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if cross_products[i] > 0:
            angle = angle
        else:
            angle = -angle
        angles.append(angle)
    
    total_turn = sum(angles)
    if abs(abs(total_turn) - 2*np.pi) > 1e-6:
        return 0.0
    
    # Compute vertex scores
    min_score = 1.0
    for i in range(n):
        v = centered[i]
        d = np.linalg.norm(centered - v, axis=1)
        d[i] = np.inf
        d_sorted = np.sort(d)[:19]
        
        if d_sorted[0] < 0.05:
            min_score = 0.0
            break
        
        gaps = np.diff(d_sorted)
        k = np.argmin(gaps)
        
        D = (d_sorted[k] + d_sorted[k+1]) / 2
        if D < 0.1:
            min_score = 0.0
            break
        
        if k == 0:
            four_d = d_sorted[0:4]
        elif k == 17:
            four_d = d_sorted[15:19]
        else:
            four_d = d_sorted[k-1:k+3]
        
        ratios = np.maximum(four_d / D, D / four_d)
        raw = np.mean(ratios)
        
        vertex_score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
        min_score = min(min_score, vertex_score)
    
    return min_score


def _fast_min_vertex_score(angles_20):
    """
    Fast scoring: given 20 angles (radians), construct unit-circle points,
    normalize, and compute min vertex score. Returns negative score if invalid.
    """
    n = 20
    angles = np.sort(angles_20)
    # Ensure angles are in [0, 2*pi)
    angles = angles % (2 * np.pi)
    angles = np.sort(angles)
    
    # Build polygon on unit circle
    x = np.cos(angles)
    y = np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    
    # Normalize
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_d = dists_to_centroid.mean()
    if mean_d <= 0.01:
        return 0.0
    centered = centered / mean_d
    
    # Quick convexity check via cross products
    cross_products = np.zeros(n)
    for i in range(n):
        p0 = centered[i]
        p1 = centered[(i+1) % n]
        p2 = centered[(i+2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross_products[i] = e1[0]*e2[1] - e1[1]*e2[0]
    
    if np.any(np.abs(cross_products) < 1e-12):
        return 0.0
    if np.any(cross_products > 0) and np.any(cross_products < 0):
        return 0.0
    
    # Total turning check
    total_turn = 0.0
    for i in range(n):
        p0 = centered[i]
        p1 = centered[(i+1) % n]
        p2 = centered[(i+2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        n1 = np.linalg.norm(e1)
        n2 = np.linalg.norm(e2)
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        cos_a = np.dot(e1, e2) / (n1 * n2)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        a = np.arccos(cos_a)
        if cross_products[i] < 0:
            a = -a
        total_turn += a
    
    if abs(abs(total_turn) - 2*np.pi) > 1e-5:
        return 0.0
    
    # Compute min vertex score
    min_score = 1.0
    for i in range(n):
        v = centered[i]
        d = np.linalg.norm(centered - v, axis=1)
        d[i] = np.inf
        d_sorted = np.sort(d)[:19]
        
        if d_sorted[0] < 0.05:
            return 0.0
        
        gaps = np.diff(d_sorted)
        k = np.argmin(gaps)
        D = (d_sorted[k] + d_sorted[k+1]) / 2
        if D < 0.1:
            return 0.0
        
        if k == 0:
            four_d = d_sorted[0:4]
        elif k == 17:
            four_d = d_sorted[15:19]
        else:
            four_d = d_sorted[k-1:k+3]
        
        ratios = np.maximum(four_d / D, D / four_d)
        raw = np.mean(ratios)
        vertex_score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
        min_score = min(min_score, vertex_score)
    
    return min_score


def equidistant_points_polygon20() -> np.ndarray:
    """
    Use differential evolution to optimize 20 angular positions on a circle
    to maximize the minimum vertex equidistance score.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin
    
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Get baseline score from regular polygon
    x = np.cos(base_angles)
    y = np.sin(base_angles)
    regular_poly = np.stack([x, y], axis=1)
    best_score = _score_polygon(regular_poly)
    best_polygon = regular_poly.copy()
    
    # Objective for differential evolution: maximize min vertex score
    def objective(angles_flat):
        if time.time() > deadline:
            return 0.0
        return -_fast_min_vertex_score(angles_flat)
    
    # Bounds: each angle in [0, 2*pi)
    bounds = [(0.0, 2.0 * np.pi)] * n
    
    # Run differential evolution
    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-4,
            mutation=(0.3, 1.0),
            recombination=0.7,
            polish=False,
            init='sobol',
            updating='deferred'
        )
        
        if result.fun < 0:  # valid result (negative of score)
            opt_angles = result.x
            x = np.cos(opt_angles)
            y = np.sin(opt_angles)
            opt_poly = np.stack([x, y], axis=1)
            opt_score = _score_polygon(opt_poly)
            if opt_score > best_score:
                best_score = opt_score
                best_polygon = opt_poly.copy()
    except Exception:
        pass
    
    # If time remains, do local refinement around best
    if best_polygon is not None and time.time() < deadline - 10.0:
        rng = np.random.default_rng(123)
        for _ in range(3000):
            if time.time() > deadline - 5.0:
                break
            # Small angular perturbation
            angles_now = np.arctan2(best_polygon[:, 1], best_polygon[:, 0])
            angles_now = np.sort(angles_now)
            noise = rng.normal(0, 0.03, n)
            new_angles = np.sort(angles_now + noise)
            new_angles = new_angles % (2 * np.pi)
            new_angles = np.sort(new_angles)
            x = np.cos(new_angles)
            y = np.sin(new_angles)
            perturbed = np.stack([x, y], axis=1)
            s = _score_polygon(perturbed)
            if s > best_score:
                best_score = s
                best_polygon = perturbed.copy()
    
    return best_polygon


# EVOLVE-BLOCK-END
