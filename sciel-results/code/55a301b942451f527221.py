# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import math

def _score_polygon(polygon):
    """
    Compute the minimum vertex score for a given polygon.
    Returns 0.0 if the polygon is invalid.
    """
    n = len(polygon)
    if n != 20:
        return 0.0
    
    # Check for finite coordinates
    if not np.all(np.isfinite(polygon)):
        return 0.0
    
    # Normalize: subtract centroid, divide by mean distance to centroid
    centroid = np.mean(polygon, axis=0)
    normalized = polygon - centroid
    dists_to_centroid = np.linalg.norm(normalized, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist <= 0.01:
        return 0.0
    
    normalized = normalized / mean_dist
    
    # Check for repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(normalized[i] - normalized[j]) < 1e-10:
                return 0.0
    
    # Check strict convexity
    # Compute cross products of consecutive edges
    cross_products = []
    for i in range(n):
        p0 = normalized[i]
        p1 = normalized[(i+1) % n]
        p2 = normalized[(i+2) % n]
        edge1 = p1 - p0
        edge2 = p2 - p1
        cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]
        cross_products.append(cross)
    
    cross_products = np.array(cross_products)
    
    # All cross products must be nonzero and of the same sign
    if np.any(np.abs(cross_products) < 1e-12):
        return 0.0
    
    signs = np.sign(cross_products)
    if len(np.unique(signs)) > 1:
        return 0.0
    
    # Total turning should be ±2π
    # For a convex polygon, the sum of exterior angles is 2π
    # We can verify this by checking that the polygon is simple and convex
    # A simpler check: for strictly convex polygon with consistent turning,
    # the total turning is automatically 2π (or -2π)
    
    # Compute vertex scores
    min_score = float('inf')
    
    for i in range(n):
        v = normalized[i]
        # Compute distances from v to all other vertices
        other_indices = [j for j in range(n) if j != i]
        dists = np.sort(np.linalg.norm(normalized[other_indices] - v, axis=1))
        
        # Check if any distance < 0.05
        if dists[0] < 0.05:
            min_score = 0.0
            break
        
        # Find k: first index minimizing d_{k+1} - d_k
        gaps = np.diff(dists)
        k = np.argmin(gaps)
        
        # D = (d_k + d_{k+1}) / 2
        D = (dists[k] + dists[k+1]) / 2.0
        
        if D < 0.1:
            min_score = 0.0
            break
        
        # Take four distances
        if k == 0:
            selected = dists[0:4]
        elif k == 17:
            selected = dists[15:19]
        else:
            selected = dists[k-1:k+3]
        
        # raw = mean over these four d of max(d/D, D/d)
        ratios = np.maximum(selected / D, D / selected)
        raw = np.mean(ratios)
        
        # vertex score = min(1, 1/raw)
        vertex_score = min(1.0, 1.0 / raw)
        
        min_score = min(min_score, vertex_score)
    
    return min_score if min_score != float('inf') else 0.0


def _neg_score(params):
    """
    Negative of the score for optimization (we maximize score, so minimize -score).
    params: 40-dimensional vector of vertex coordinates.
    """
    polygon = params.reshape(20, 2)
    score = _score_polygon(polygon)
    return -score


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices optimized for the equidistance scoring.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit
    
    # Start with regular polygon
    n = 20
    radius = 1.0
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    best_polygon = np.stack([x, y], axis=1)
    best_score = _score_polygon(best_polygon)
    
    # Try local optimization from regular polygon
    initial_params = best_polygon.flatten()
    
    # Use Nelder-Mead for local optimization
    try:
        result = minimize(
            _neg_score,
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if result.success or result.fun < -best_score:
            candidate = result.x.reshape(20, 2)
            candidate_score = _score_polygon(candidate)
            if candidate_score > best_score:
                best_polygon = candidate
                best_score = candidate_score
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 60.0:
        return best_polygon
    
    # Try differential evolution for global search
    try:
        bounds = [(-5, 5)] * 40
        
        def _de_score(params):
            if time.time() > deadline - 30.0:
                return 0.0
            polygon = params.reshape(20, 2)
            return -_score_polygon(polygon)
        
        de_result = differential_evolution(
            _de_score,
            bounds,
            seed=42,
            maxiter=100,
            tol=1e-8,
            polish=True,
            update_interval=10
        )
        
        if de_result.success or de_result.fun < -best_score:
            candidate = de_result.x.reshape(20, 2)
            candidate_score = _score_polygon(candidate)
            if candidate_score > best_score:
                best_polygon = candidate
                best_score = candidate_score
    except Exception:
        pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                _neg_score,
                best_polygon.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if result.fun < -best_score:
                candidate = result.x.reshape(20, 2)
                candidate_score = _score_polygon(candidate)
                if candidate_score > best_score:
                    best_polygon = candidate
                    best_score = candidate_score
        except Exception:
            pass
    
    return best_polygon


# EVOLVE-BLOCK-END
