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
    # Compute angles of edges
    angles = np.arctan2(cross_products, 
                        np.array([(p[(i+1)%n] - p[i]) @ (p[(i+2)%n] - p[(i+1)%n]) for i in range(n)]))
    total_turn = np.sum(angles)
    if abs(abs(total_turn) - 2 * np.pi) > 1e-5:
        return False
    return True

def _vertex_score(v, polygon):
    """Compute the score for a single vertex v in the polygon."""
    # Compute distances from v to all other vertices
    dists = np.linalg.norm(polygon - v, axis=1)
    # Exclude the vertex itself (distance 0)
    other_dists = np.sort(dists[dists > 1e-10])
    
    if len(other_dists) < 19:
        return 0.0
    
    d = other_dists[:19]
    
    # If any distance < 0.05, score is 0
    if d[0] < 0.05:
        return 0.0
    
    # Find k: first index minimizing d_{k+1} - d_k
    gaps = np.diff(d)
    k = np.argmin(gaps)
    
    # D = (d_k + d_{k+1}) / 2
    D = (d[k] + d[k + 1]) / 2
    
    # If D < 0.1, score is 0
    if D < 0.1:
        return 0.0
    
    # Choose four distances
    if k == 0:
        four_d = d[0:4]
    elif k == 17:
        four_d = d[15:19]
    else:
        four_d = d[k-1:k+3]
    
    # raw = mean of max(d/D, D/d) over the four distances
    ratios = np.maximum(four_d / D, D / four_d)
    raw = np.mean(ratios)
    
    return min(1.0, 1.0 / raw)

def _polygon_score(polygon):
    """Compute the overall score: minimum vertex score over all vertices."""
    n = len(polygon)
    scores = []
    for i in range(n):
        s = _vertex_score(polygon[i], polygon)
        scores.append(s)
    return min(scores)

def _evaluate(polygon):
    """Full evaluation: normalize, check convexity, compute score."""
    normed = _normalize(polygon)
    if normed is None:
        return 0.0
    if not _is_strictly_convex(normed):
        return 0.0
    return _polygon_score(normed)

def _regular_polygon(n=20, radius=1.0):
    """Create a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)

def _parametric_polygon(params):
    """
    Generate a polygon from parameters.
    params: [a2, a4, a6, a8, a10]
    Radius modulation: r(i) = 1 + a2*cos(2*theta_i) + a4*cos(4*theta_i) + ...
    This structure helps maintain convexity for small amplitudes and allows fine-tuning.
    """
    n = 20
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    r = np.ones(n)
    for j, k in enumerate([2, 4, 6, 8, 10]):
        r += params[j] * np.cos(k * angles)
        
    # Ensure radii are positive
    r = np.maximum(r, 0.5)
    
    return np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)

def _objective(params):
    """
    Objective function for minimization (we minimize negative score).
    """
    try:
        poly = _parametric_polygon(params)
        score = _evaluate(poly)
        return -score
    except Exception:
        return 10.0 # Large penalty for invalid

def _search_parametric(time_budget=600):
    """
    Search for optimal parameters using L-BFGS-B with multiple restarts.
    """
    start_time = time.time()
    deadline = start_time + time_budget - 30 # Safety margin
    
    best_score = -1
    best_params = None
    
    # Initial regular polygon score
    reg_poly = _regular_polygon(20)
    best_score = _evaluate(reg_poly)
    best_poly = reg_poly.copy()
    
    # Define bounds for parameters to ensure convexity (roughly)
    # Amplitudes should be small enough to not create self-intersections or non-convexity
    bounds = [(-0.2, 0.2) for _ in range(5)]
    
    n_restarts = 50
    restart_count = 0
    
    while time.time() < deadline:
        restart_count += 1
        # Generate random initial parameters
        # Bias towards small values to start in valid region
        init_params = np.random.uniform(-0.05, 0.05, 5)
        
        # Optimize
        res = minimize(_objective, init_params, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 100, 'ftol': 1e-8})
        
        if res.success:
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x
                best_poly = _parametric_polygon(res.x)
        
        # If we've used a significant portion of time, reduce restarts or stop
        if restart_count >= n_restarts:
            break
            
    # If no improvement found, return regular polygon
    if best_params is None or best_score <= _evaluate(reg_poly):
        return reg_poly
    
    return best_poly

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that tries to be a counterexample to the
    conjecture "every convex polygon has a vertex v with no 4 other vertices equidistant
    from v". The goal is a polygon where EVERY vertex has (at least) 4 other vertices that
    are approximately equidistant from it. Quality is the minimum, over all vertices, of a
    per-vertex equidistance score in [0, 1] (1.0 = perfect counterexample).
    
    Uses a parametric search approach with L-BFGS-B optimization.
    
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    # Use parametric search with time budget
    result = _search_parametric(time_budget=600)
    
    # Validate the result
    score = _evaluate(result)
    if score <= 0:
        # Fallback to regular polygon
        return _regular_polygon(20)
    
    return result

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-END
