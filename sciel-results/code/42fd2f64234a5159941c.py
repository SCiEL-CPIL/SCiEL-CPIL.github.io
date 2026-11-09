# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-END


def _normalize(polygon):
    """Subtract centroid, divide by mean distance to centroid."""
    c = polygon.mean(axis=0)
    p = polygon - c
    dists = np.linalg.norm(p, axis=1)
    mean_d = dists.mean()
    if mean_d <= 0.01:
        return None
    return p / mean_d


def _is_strictly_convex(p):
    """Check strictly convex: no repeated points, all turns same sign nonzero, total turn +-2pi."""
    n = len(p)
    # Check no repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(p[i] - p[j]) < 1e-10:
                return False
    # Compute cross products of consecutive edges
    cross_products = []
    for i in range(n):
        e1 = p[(i + 1) % n] - p[i]
        e2 = p[(i + 2) % n] - p[(i + 1) % n]
        cp = e1[0] * e2[1] - e1[1] * e2[0]
        cross_products.append(cp)
    cross_products = np.array(cross_products)
    # All must be nonzero and same sign
    if np.any(np.abs(cross_products) < 1e-10):
        return False
    signs = np.sign(cross_products)
    if len(set(signs)) != 1:
        return False
    # Total turning should be +-2*pi
    angles = np.arctan2(cross_products, 
                        np.array([(p[(i+1)%n] - p[i]) @ (p[(i+2)%n] - p[(i+1)%n]) for i in range(n)]))
    total_turn = np.sum(angles)
    if abs(abs(total_turn) - 2 * np.pi) > 1e-5:
        return False
    return True


def _vertex_score(v, polygon):
    """Compute the score for a single vertex v in the polygon."""
    dists = np.linalg.norm(polygon - v, axis=1)
    other_dists = np.sort(dists[dists > 1e-10])
    
    if len(other_dists) < 19:
        return 0.0
    
    d = other_dists[:19]
    
    if d[0] < 0.05:
        return 0.0
    
    gaps = np.diff(d)
    k = np.argmin(gaps)
    
    D = (d[k] + d[k + 1]) / 2
    
    if D < 0.1:
        return 0.0
    
    if k == 0:
        four_d = d[0:4]
    elif k == 17:
        four_d = d[15:19]
    else:
        four_d = d[k-1:k+3]
    
    ratios = np.maximum(four_d / D, D / four_d)
    raw = np.mean(ratios)
    
    return min(1.0, 1.0 / raw)


def _polygon_score(polygon):
    """Compute the overall score: minimum vertex score over all vertices."""
    n = len(polygon)
    scores = np.zeros(n)
    for i in range(n):
        scores[i] = _vertex_score(polygon[i], polygon)
    return np.min(scores)


def _evaluate(polygon):
    """Full evaluation: normalize, check convexity, compute score."""
    normed = _normalize(polygon)
    if normed is None:
        return 0.0
    if not _is_strictly_convex(normed):
        return 0.0
    return _polygon_score(normed)


def _parametric_polygon(coeffs):
    """
    Generate a convex 20-gon from parametric coefficients.
    coeffs = [a1, b1, a2, b2, a3, b3, a4, b4, a5, b5]
    r_i = 1 + sum_{k=1..5} (a_k * cos(k * theta_i) + b_k * sin(k * theta_i))
    theta_i = 2*pi*i/20
    """
    n = 20
    theta = 2 * np.pi * np.arange(n) / n
    r = np.ones(n)
    for k in range(1, 6):
        a_k = coeffs[2*(k-1)]
        b_k = coeffs[2*(k-1)+1]
        r += a_k * np.cos(k * theta) + b_k * np.sin(k * theta)
    
    # Ensure positive radii for convexity
    r = np.maximum(r, 0.3)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=1)


def _objective_smoothed(coeffs, sigma=0.02):
    """
    Smoothed objective: negative of soft-min of vertex scores.
    Lower is better for optimizer.
    """
    poly = _parametric_polygon(coeffs)
    normed = _normalize(poly)
    if normed is None:
        return 1000.0
    if not _is_strictly_convex(normed):
        return 1000.0
    
    n = len(normed)
    scores = np.zeros(n)
    for i in range(n):
        scores[i] = _vertex_score(normed[i], normed)
    
    # Soft minimum: -log(sum(exp(-scores/sigma))) * sigma
    # This is smooth and approximates -min(scores) as sigma -> 0
    min_score = np.min(scores)
    # Use a smooth approximation: min + sigma * log(mean(exp((min - scores)/sigma)))
    # Actually simpler: just use the min but with a smoothing term
    # Let's use: -min(scores) + penalty for variance
    # Better: use the actual min but the parametric family should keep scores relatively uniform
    
    return -min_score


def _objective_exact(coeffs):
    """Exact objective: negative of minimum vertex score."""
    poly = _parametric_polygon(coeffs)
    normed = _normalize(poly)
    if normed is None:
        return 1000.0
    if not _is_strictly_convex(normed):
        return 1000.0
    
    n = len(normed)
    scores = np.zeros(n)
    for i in range(n):
        scores[i] = _vertex_score(normed[i], normed)
    
    return -np.min(scores)


def _search_parametric(time_budget=600, deadline_margin=30):
    """Search for best polygon using parametric family with multi-start L-BFGS-B."""
    start_time = time.time()
    deadline = start_time + time_budget - deadline_margin
    
    best_coeffs = np.zeros(10)
    best_score = _evaluate(_parametric_polygon(best_coeffs))
    
    # Multi-start optimization
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial coefficients
        if restart == 0:
            init = np.zeros(10)
        else:
            # Mix of random and perturbed best
            if restart % 3 == 0:
                # Pure random
                init = np.random.uniform(-0.15, 0.15, 10)
            elif restart % 3 == 1:
                # Perturb best
                init = best_coeffs + np.random.normal(0, 0.05, 10)
            else:
                # Random with smaller amplitude for higher harmonics
                init = np.random.uniform(-0.1, 0.1, 10)
                init[4:] *= 0.5  # Dampen higher harmonics
        
        # Evaluate initial
        init_score = _evaluate(_parametric_polygon(init))
        if init_score > best_score:
            best_score = init_score
            best_coeffs = init.copy()
        
        # Run L-BFGS-B
        try:
            result = minimize(
                _objective_smoothed,
                init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.fun < 0:  # Valid result
                result_score = _evaluate(_parametric_polygon(result.x))
                if result_score > best_score:
                    best_score = result_score
                    best_coeffs = result.x.copy()
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline:
            break
    
    # Final local refinement with exact objective
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _objective_exact,
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.fun < 0:
                result_score = _evaluate(_parametric_polygon(result.x))
                if result_score > best_score:
                    best_score = result_score
                    best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return _parametric_polygon(best_coeffs), best_score


def _regular_polygon(n=20, radius=1.0):
    """Create a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that tries to be a counterexample to the
    conjecture "every convex polygon has a vertex v with no 4 other vertices equidistant
    from v". The goal is a polygon where EVERY vertex has (at least) 4 other vertices that
    are approximately equidistant from it. Quality is the minimum, over all vertices, of a
    per-vertex equidistance score in [0, 1] (1.0 = perfect counterexample).
    
    Uses a parametric family with Fourier harmonics and multi-start L-BFGS-B optimization.
    
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    # Use parametric search with time budget
    result, score = _search_parametric(time_budget=600, deadline_margin=30)
    
    # Validate the result
    final_score = _evaluate(result)
    if final_score <= 0:
        # Fallback to regular polygon
        return _regular_polygon(20)
    
    return result


# EVOLVE-BLOCK-START
import numpy as np
import time

# EVOLVE-BLOCK-END
