# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

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

def _objective(params, n=20):
    """
    Objective function for optimization.
    params: array of length 2*n. First n are log-radii, next n are angle deltas.
    Returns negative score (to be minimized).
    """
    # Unpack
    log_r = params[:n]
    d_theta = params[n:]
    
    # Reconstruct angles
    # d_theta must be positive. We use softplus to ensure positivity? 
    # Or just clip? Softplus is smoother.
    # Let's use exp(d_theta) to ensure positive deltas.
    # But we need sum of angles to be 2*pi.
    # Actually, the angles are relative.
    # Let theta_0 = 0.
    # theta_i = sum_{j=0}^{i-1} exp(d_theta_j)
    # Total angle = sum exp(d_theta_j). We can scale the final polygon, so absolute scale doesn't matter for convexity/score (normalization handles it).
    # However, the relative angles matter.
    
    theta_deltas = np.exp(d_theta)
    angles = np.cumsum(theta_deltas)
    angles = np.append(0, angles)[:-1] # theta_0 = 0
    
    # Radii
    radii = np.exp(log_r)
    
    # Construct polygon
    poly = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    
    # Evaluate
    score = _evaluate(poly)
    
    # If invalid, return a large penalty
    if score <= 0:
        return 100.0
        
    return -score

def _optimize_from_start(start_params, n=20, max_iter=1000, time_limit=10):
    """Run Nelder-Mead optimization from a given start point."""
    start_time = time.time()
    
    # We need to check time inside the callback? SciPy doesn't easily support time-based stopping in minimize.
    # We will just run a fixed number of iterations or rely on the outer loop.
    # For better control, we can implement a custom step or just use a small max_iter.
    
    try:
        res = scipy.optimize.minimize(
            _objective, 
            start_params, 
            args=(n,),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False}
        )
        return res.x, -res.fun
    except Exception:
        return start_params, 0.0

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that tries to be a counterexample to the
    conjecture "every convex polygon has a vertex v with no 4 other vertices equidistant
    from v". The goal is a polygon where EVERY vertex has (at least) 4 other vertices that
    are approximately equidistant from it. Quality is the minimum, over all vertices, of a
    per-vertex equidistance score in [0, 1] (1.0 = perfect counterexample).
    
    Uses a search approach: generate many varied candidate answers and keep the best one
    found within the time budget.
    
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    n = 20
    start_time = time.time()
    deadline = start_time + 600 - 30 # 30s safety margin
    
    best_poly = _regular_polygon(n)
    best_score = _evaluate(best_poly)
    
    # Parameterization: log_r (n) and d_theta (n)
    # Initial guess: regular polygon
    regular = _regular_polygon(n)
    # Extract radii and angles from regular polygon
    dists_from_origin = np.linalg.norm(regular, axis=1)
    angles_reg = np.arctan2(regular[:, 1], regular[:, 0])
    # Ensure angles are sorted and continuous
    angles_reg = np.sort(angles_reg)
    # Handle wrap around
    if angles_reg[0] > angles_reg[-1]:
        # This shouldn't happen with sort, but just in case
        pass
    
    # Calculate deltas
    d_angles_reg = np.diff(np.append(angles_reg, angles_reg[0] + 2*np.pi))
    
    init_log_r = np.log(dists_from_origin)
    init_d_theta = np.log(d_angles_reg)
    
    # Multi-start optimization
    n_restarts = 50
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Perturb initial parameters
        noise_scale = 0.5 if i == 0 else 0.2
        curr_params = init_params + np.random.normal(0, noise_scale, 2*n)
        
        # Ensure d_theta corresponds to positive angles (log space handles this)
        # Ensure radii are positive (log space handles this)
        
        # Run optimization
        opt_params, opt_score = _optimize_from_start(curr_params, n=n, max_iter=200)
        
        if opt_score > best_score:
            best_score = opt_score
            # Reconstruct polygon from best params
            log_r = opt_params[:n]
            d_theta = opt_params[n:]
            theta_deltas = np.exp(d_theta)
            angles = np.cumsum(theta_deltas)
            angles = np.append(0, angles)[:-1]
            radii = np.exp(log_r)
            best_poly = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
            
        # If we found a very good score, we can try to refine it further with more iterations
        if best_score > 1.5 and i < 5:
             # Do a few more rounds of local search with tighter tolerance
             pass

    # Final validation
    final_score = _evaluate(best_poly)
    if final_score <= 0:
        return _regular_polygon(n)
        
    return best_poly

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

# EVOLVE-BLOCK-END
