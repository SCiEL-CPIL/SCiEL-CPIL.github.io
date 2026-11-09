# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 620.0  # Safety margin before 630s kill


def _is_valid_convex(pts):
    """Check if points form a strictly convex polygon."""
    n = len(pts)
    if n < 3:
        return False
    # Check for repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-12:
                return False
    # Compute cross products of consecutive edges
    cross_signs = []
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        p2 = pts[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-12:
            return False
        cross_signs.append(np.sign(cross))
    # All cross products must have the same sign
    if not (all(s == cross_signs[0] for s in cross_signs)):
        return False
    # Total turning should be ±2π (for a simple convex polygon this is automatic
    # if all turns are same sign, but let's verify)
    total_turn = 0.0
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        p2 = pts[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])
        turn = angle2 - angle1
        # Normalize to [-π, π]
        while turn > np.pi:
            turn -= 2 * np.pi
        while turn < -np.pi:
            turn += 2 * np.pi
        total_turn += turn
    if abs(abs(total_turn) - 2 * np.pi) > 1e-6:
        return False
    return True


def _vertex_score(pts, v_idx):
    """Compute the vertex score for vertex v_idx."""
    n = len(pts)
    v = pts[v_idx]
    # Distances to all other vertices
    dists = np.zeros(n - 1)
    idx = 0
    for j in range(n):
        if j != v_idx:
            dists[idx] = np.linalg.norm(v - pts[j])
            idx += 1
    # Sort distances
    d_sorted = np.sort(dists)
    # Check if any distance < 0.05
    if d_sorted[0] < 0.05:
        return 0.0
    # Find k: first index minimizing d_{k+1} - d_k
    gaps = np.diff(d_sorted)
    k = np.argmin(gaps)
    D = (d_sorted[k] + d_sorted[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    # Select four distances
    if k == 0:
        selected = d_sorted[0:4]
    elif k == 17:
        selected = d_sorted[15:19]
    else:
        selected = d_sorted[k - 1:k + 3]
    # raw = mean over these four d of max(d/D, D/d)
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    score = min(1.0, 1.0 / raw)
    return score


def _polygon_score(pts):
    """Compute the overall polygon score (minimum vertex score)."""
    if not _is_valid_convex(pts):
        return 0.0
    n = len(pts)
    scores = [_vertex_score(pts, i) for i in range(n)]
    return min(scores)


def _normalize(pts):
    """Normalize: subtract centroid, divide by mean distance to centroid."""
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    dists = np.linalg.norm(pts, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return pts
    return pts / mean_dist


def _regular_20_gon():
    """Generate a regular 20-gon."""
    n = 20
    radius = 1.0
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.stack([x, y], axis=1)


def _perturb_regular(angles_offset, radii):
    """Generate a polygon with given angle offsets and radii."""
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    angles = base_angles + angles_offset
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)


def _optimize_from_start(start_pts, max_iter=200, timeout=30.0):
    """Optimize a polygon to maximize the minimum vertex score."""
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > timeout:
            return 1e6
        n = 20
        angles = params[:n]
        radii = params[n:2*n]
        pts = _perturb_regular(angles, radii)
        pts = _normalize(pts)
        if not _is_valid_convex(pts):
            return 1e6
        score = _polygon_score(pts)
        return -score  # Minimize negative score
    
    # Initial parameters
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    # Extract angles and radii from start_pts
    centroid = np.mean(start_pts, axis=0)
    pts_centered = start_pts - centroid
    radii_init = np.linalg.norm(pts_centered, axis=1)
    angles_init = np.arctan2(pts_centered[:, 1], pts_centered[:, 0])
    # Adjust to match base_angles
    angle_diffs = angles_init - base_angles
    # Unwrap
    angle_diffs = np.unwrap(angle_diffs)
    
    x0 = np.concatenate([angle_diffs, radii_init])
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6})
    
    if result.fun < 1e5:
        best_params = result.x
        best_pts = _perturb_regular(best_params[:n], best_params[n:2*n])
        best_pts = _normalize(best_pts)
        best_score = -result.fun
        return best_pts, best_score
    
    return start_pts, 0.0


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that tries to be a counterexample to the
    conjecture "every convex polygon has a vertex v with no 4 other vertices equidistant
    from v". The goal is a polygon where EVERY vertex has (at least) 4 other vertices that
    are approximately equidistant from it. Quality is the minimum, over all vertices, of a
    per-vertex equidistance score in [0, 1] (1.0 = perfect counterexample).

    This implementation uses optimization to perturb a regular 20-gon to create
    4-fold equidistant structure at each vertex.

    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    # Start with regular 20-gon
    regular = _regular_20_gon()
    regular = _normalize(regular)
    best_pts = regular
    best_score = _polygon_score(regular)
    
    # Try various perturbations and optimize
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Strategy: try to create symmetry where each vertex is equidistant to 4 others
    # For a regular 20-gon, vertex i is at distance 2R*sin(k*pi/20) from vertex i+k
    # We want to make distances to vertices i±a, i±b approximately equal for some a,b
    
    # Try different perturbation patterns
    perturbations = []
    
    # Pattern 1: Small random perturbations
    np.random.seed(42)
    for trial in range(5):
        if time.time() - _START_TIME > TIME_LIMIT - 10:
            break
        angle_pert = np.random.randn(n) * 0.01
        radius_pert = np.random.randn(n) * 0.01
        start = _perturb_regular(angle_pert, np.ones(n) + radius_pert)
        start = _normalize(start)
        if _is_valid_convex(start):
            pts, score = _optimize_from_start(start, max_iter=100, timeout=10.0)
            if score > best_score:
                best_score = score
                best_pts = pts
    
    # Pattern 2: Structured perturbations to create 4-fold symmetry
    # Try to make vertices i and i+10 have similar distances to their neighbors
    for trial in range(3):
        if time.time() - _START_TIME > TIME_LIMIT - 10:
            break
        # Alternate radii slightly
        radii = np.ones(n)
        for i in range(n):
            if i % 2 == 0:
                radii[i] = 1.0 + 0.02 * trial
            else:
                radii[i] = 1.0 - 0.02 * trial
        start = _perturb_regular(np.zeros(n), radii)
        start = _normalize(start)
        if _is_valid_convex(start):
            pts, score = _optimize_from_start(start, max_iter=100, timeout=10.0)
            if score > best_score:
                best_score = score
                best_pts = pts
    
    # Pattern 3: Group-based perturbations
    # Divide 20 vertices into 5 groups of 4, perturb each group
    for trial in range(3):
        if time.time() - _START_TIME > TIME_LIMIT - 10:
            break
        angles = np.zeros(n)
        radii = np.ones(n)
        for g in range(5):
            group_offset = np.random.randn(4) * 0.005 * (trial + 1)
            for j in range(4):
                idx = g * 4 + j
                angles[idx] = group_offset[j]
                radii[idx] = 1.0 + np.random.randn() * 0.005 * (trial + 1)
        start = _perturb_regular(angles, radii)
        start = _normalize(start)
        if _is_valid_convex(start):
            pts, score = _optimize_from_start(start, max_iter=100, timeout=10.0)
            if score > best_score:
                best_score = score
                best_pts = pts
    
    # Final check
    if not _is_valid_convex(best_pts):
        return regular
    
    return best_pts


# EVOLVE-BLOCK-END
