# EVOLVE-BLOCK-START
import numpy as np
import time

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


def _symmetric_polygon20(r_inner, r_outer):
    """
    Generate a 20-gon with D10 symmetry.
    Vertices alternate between radius r_inner and r_outer.
    Angles are fixed at 18 degrees apart.
    """
    n = 20
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Alternate radii: even indices get r_outer, odd get r_inner
    # Or vice versa, doesn't matter due to symmetry
    radii = np.array([r_outer if i % 2 == 0 else r_inner for i in range(n)])
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))


def _search_symmetric():
    """
    Search for the best symmetric 20-gon.
    We optimize the ratio of inner to outer radius.
    Due to normalization, absolute scale doesn't matter, only the ratio.
    Let r_outer = 1.0, vary r_inner in (0, 1).
    """
    best_score = -1.0
    best_poly = _regular_polygon(20)
    
    # Grid search over the ratio
    # r_inner must be > 0 for convexity (strictly)
    # If r_inner is too close to r_outer, it's regular (score ~1.0)
    # If r_inner is too small, it might violate convexity or distance constraints
    # Let's sample a dense grid
    ratios = np.linspace(0.1, 0.99, 200)
    
    for r_in in ratios:
        poly = _symmetric_polygon20(r_in, 1.0)
        score = _evaluate(poly)
        if score > best_score:
            best_score = score
            best_poly = poly.copy()
            
    # Refine with golden section or fine grid around best
    if best_score > 0:
        # Find the ratio that gave the best score roughly
        # We can just do a fine grid around the best found region if we tracked it, 
        # but for simplicity, a dense grid of 1000 points is fast enough.
        ratios_fine = np.linspace(0.1, 0.99, 1000)
        for r_in in ratios_fine:
            poly = _symmetric_polygon20(r_in, 1.0)
            score = _evaluate(poly)
            if score > best_score:
                best_score = score
                best_poly = poly.copy()
                
    return best_poly, best_score


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Uses symmetry-constrained optimization to find a high-scoring configuration.
    """
    start_time = time.time()
    
    # 1. Try symmetric search (very fast, likely finds global optimum in this class)
    sym_poly, sym_score = _search_symmetric()
    
    # 2. Check time
    if time.time() - start_time > 500:
        return sym_poly if sym_score > 0 else _regular_polygon(20)
        
    # 3. Local refinement of the symmetric polygon
    # Even though the symmetric class is 1D, we can break symmetry slightly 
    # to see if we can improve, but usually symmetry is optimal for these metrics.
    # Let's do a quick local search from the best symmetric point.
    
    current_poly = sym_poly.copy()
    current_score = sym_score
    
    # If symmetric score is already very high, return it
    if current_score > 0.95:
        return current_poly
        
    # Simple local search
    step_size = 0.05
    for _ in range(1000):
        if time.time() - start_time > 580:
            break
        # Perturb
        perturbed = current_poly + np.random.normal(0, step_size, current_poly.shape)
        score = _evaluate(perturbed)
        if score > current_score + 1e-6:
            current_poly = perturbed
            current_score = score
            if current_score > sym_score:
                sym_score = current_score
                sym_poly = current_poly.copy()
        else:
            step_size *= 0.95
            
    return current_poly


# EVOLVE-BLOCK-START
import numpy as np
import time

# EVOLVE-BLOCK-END
