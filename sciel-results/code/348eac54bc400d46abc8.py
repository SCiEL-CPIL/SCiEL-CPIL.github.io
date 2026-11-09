# EVOLVE-BLOCK-START
import numpy as np
import time


def _score_polygon(poly: np.ndarray) -> float:
    """
    Compute the minimum vertex score over all vertices of a polygon.
    Returns 0.0 if the polygon is invalid (not strictly convex, etc.).
    """
    n = len(poly)
    if n != 20:
        return 0.0
    
    # Check for finite coordinates
    if not np.all(np.isfinite(poly)):
        return 0.0
    
    # Check for repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(poly[i] - poly[j]) < 1e-12:
                return 0.0
    
    # Check strict convexity: all turns same sign, nonzero
    # Compute cross products of consecutive edges
    min_cross = float('inf')
    max_cross = float('-inf')
    for i in range(n):
        p0 = poly[i]
        p1 = poly[(i + 1) % n]
        p2 = poly[(i + 2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        if abs(cross) < 1e-15:
            return 0.0
        min_cross = min(min_cross, cross)
        max_cross = max(max_cross, cross)
    
    # All crosses must be same sign
    if min_cross * max_cross <= 0:
        return 0.0
    
    # Normalize: subtract centroid, divide by mean distance to centroid
    centroid = np.mean(poly, axis=0)
    centered = poly - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    normalized = centered / mean_dist
    
    # Compute vertex scores
    min_vertex_score = float('inf')
    for i in range(n):
        v = normalized[i]
        # Distances to all other vertices
        diffs = normalized - v
        dists = np.linalg.norm(diffs, axis=1)
        # Exclude self-distance (index i)
        dists = np.delete(dists, i)
        dists = np.sort(dists)
        
        # d_0 <= ... <= d_18
        if dists[0] < 0.05:
            min_vertex_score = 0.0
            break
        
        # Find k: first index minimizing d_{k+1} - d_k
        gaps = np.diff(dists)
        k = np.argmin(gaps)
        
        D = (dists[k] + dists[k + 1]) / 2
        if D < 0.1:
            min_vertex_score = 0.0
            break
        
        # Take four distances
        if k == 0:
            four_dists = dists[0:4]
        elif k == 17:
            four_dists = dists[15:19]
        else:
            four_dists = dists[k - 1:k + 3]
        
        # raw = mean of max(d/D, D/d) over these four
        ratios = np.maximum(four_dists / D, D / four_dists)
        raw = np.mean(ratios)
        vertex_score = min(1.0, 1.0 / raw)
        min_vertex_score = min(min_vertex_score, vertex_score)
    
    return min_vertex_score


def _neg_score(poly: np.ndarray) -> float:
    """Negative score for minimization."""
    return -_score_polygon(poly)


def _is_strictly_convex(poly: np.ndarray) -> bool:
    """Check if polygon is strictly convex."""
    n = len(poly)
    if n < 3:
        return False
    for i in range(n):
        p0 = poly[i]
        p1 = poly[(i + 1) % n]
        p2 = poly[(i + 2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        if abs(cross) < 1e-15:
            return False
    # Check all same sign
    crosses = []
    for i in range(n):
        p0 = poly[i]
        p1 = poly[(i + 1) % n]
        p2 = poly[(i + 2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        crosses.append(e1[0] * e2[1] - e1[1] * e2[0])
    crosses = np.array(crosses)
    if np.any(crosses > 0) and np.any(crosses < 0):
        return False
    return True


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon with equidistant vertices.
    Uses radial function r(theta) = 1 + epsilon * cos(4*theta) with optimized epsilon.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit
    
    n = 20
    best_poly = None
    best_score = -1.0
    
    # Initial: regular polygon
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    regular_poly = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    best_poly = regular_poly.copy()
    best_score = _score_polygon(regular_poly)
    
    # Try radial modulation r(theta) = 1 + eps * cos(4*theta)
    # This creates 4-fold symmetry with clusters
    eps_values = np.linspace(-0.3, 0.3, 61)
    
    for eps in eps_values:
        if time.time() > deadline - 10.0:
            break
        
        r = 1.0 + eps * np.cos(4.0 * angles)
        poly = r[:, np.newaxis] * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        
        if not _is_strictly_convex(poly):
            continue
        
        score = _score_polygon(poly)
        if score > best_score:
            best_score = score
            best_poly = poly.copy()
    
    # Local refinement around best epsilon
    if best_score > 0:
        # Find which epsilon gave best score
        # Re-derive from best_poly: it's on a modulated circle
        # Actually, let's just do a finer search around promising epsilons
        
        # Finer grid search
        eps_fine = np.linspace(-0.3, 0.3, 601)
        for eps in eps_fine:
            if time.time() > deadline - 10.0:
                break
            
            r = 1.0 + eps * np.cos(4.0 * angles)
            poly = r[:, np.newaxis] * np.stack([np.cos(angles), np.sin(angles)], axis=1)
            
            if not _is_strictly_convex(poly):
                continue
            
            score = _score_polygon(poly)
            if score > best_score:
                best_score = score
                best_poly = poly.copy()
        
        # Gradient-free local optimization (Nelder-Mead style)
        # Try perturbing the best solution
        if time.time() < deadline - 30.0:
            # Use scipy if available
            try:
                from scipy.optimize import minimize
                import numpy as np
                
                # Parameterize: small angular perturbations + radial perturbations
                # Keep it simple: optimize eps more precisely
                def objective(eps):
                    r = 1.0 + eps * np.cos(4.0 * angles)
                    poly = r[:, np.newaxis] * np.stack([np.cos(angles), np.sin(angles)], axis=1)
                    if not _is_strictly_convex(poly):
                        return 0.0
                    return -_score_polygon(poly)
                
                # Start from current best
                # Estimate eps from best_poly
                # For now, just try a few starting points
                for eps_start in [0.05, 0.1, 0.15, 0.2, -0.05, -0.1, -0.15, -0.2]:
                    if time.time() > deadline - 15.0:
                        break
                    
                    result = minimize(objective, [eps_start], method='Nelder-Mead',
                                      options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 200})
                    
                    if time.time() > deadline - 10.0:
                        break
                    
                    eps_opt = result.x[0]
                    r = 1.0 + eps_opt * np.cos(4.0 * angles)
                    poly = r[:, np.newaxis] * np.stack([np.cos(angles), np.sin(angles)], axis=1)
                    
                    if _is_strictly_convex(poly):
                        score = _score_polygon(poly)
                        if score > best_score:
                            best_score = score
                            best_poly = poly.copy()
            except Exception:
                pass
    
    # Final polish: try to improve with small angular adjustments
    if time.time() < deadline - 5.0 and best_poly is not None:
        # Try small rotations
        for rot in np.linspace(-0.05, 0.05, 21):
            if time.time() > deadline - 2.0:
                break
            theta = np.arctan2(best_poly[:, 1], best_poly[:, 0])
            r = np.linalg.norm(best_poly, axis=1)
            new_angles = theta + rot
            new_poly = np.stack([r * np.cos(new_angles), r * np.sin(new_angles)], axis=1)
            if _is_strictly_convex(new_poly):
                score = _score_polygon(new_poly)
                if score > best_score:
                    best_score = score
                    best_poly = new_poly.copy()
    
    return best_poly


# EVOLVE-BLOCK-END
