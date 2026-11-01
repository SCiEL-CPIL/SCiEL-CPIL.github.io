import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import time

N = 14

# Precompute triangle indices for vectorized area calculation
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]

def compute_score(points):
    """Compute min triangle area / convex hull area, vectorized."""
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    return min_area / hull_area

def neg_score(x):
    points = x.reshape((N, 2))
    return -compute_score(points)

def soft_neg_score(x, sharpness=50.0):
    """
    Soft-minimization of triangle areas normalized by hull area.
    Encourages maximizing the minimum area by penalizing small areas more heavily.
    """
    points = x.reshape((N, 2))
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 1e10
    except Exception:
        return 1e10
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    normalized_areas = areas / hull_area
    
    # Soft min: -log(sum(exp(-sharpness * areas))) / sharpness
    # This is a smooth approximation of the minimum.
    # We want to maximize the min, so we minimize the negative soft min.
    # -soft_min = log(sum(exp(-sharpness * a))) / sharpness
    # To avoid overflow, subtract max.
    max_a = np.max(normalized_areas)
    exp_terms = np.exp(-sharpness * (normalized_areas - max_a))
    soft_min_val = max_a - np.log(np.sum(exp_terms)) / sharpness
    
    return -soft_min_val

def _generate_two_heptagon_config(params):
    """
    Generate points from a two-heptagon configuration.
    params: [R, s, theta, u0, u1, ..., u6]
    R: radius of outer heptagon
    s: scale factor for inner heptagon (0 <= s <= 1)
    theta: rotation of inner heptagon relative to outer
    u0..u6: angular offsets for inner points relative to their "ideal" positions
             or perhaps radial adjustments?
             
    Let's use a more flexible parameterization:
    Outer heptagon: fixed at unit circle (scale R=1 for simplicity, or optimize R).
    Inner heptagon: radius r, rotation phi.
    But the directive suggests: "allow the inner heptagon to degenerate into a set of points constrained to lie on the edges of the outer heptagon via a 1D parameter per point"
    
    Let's try:
    Outer: 7 points on a circle of radius 1. Angles: 2*pi*k/7.
    Inner: 7 points. Each point i is defined by:
    - It lies on the edge of the outer heptagon between vertex i and i+1?
    - Or is it a separate ring?
    
    The directive says: "Represent the 14 points as the union of a regular heptagon and a second regular heptagon that is radially scaled and angularly shifted, but allow the inner heptagon to degenerate into a set of points constrained to lie on the edges of the outer heptagon via a 1D parameter per point"
    
    This is a bit ambiguous. "Degenerate into a set of points constrained to lie on the edges" might mean that instead of a regular inner heptagon, we place 7 points, each on a specific edge of the outer heptagon.
    Let's assume:
    Outer: 7 vertices of a regular heptagon.
    Inner: 7 points, each lying on one of the 7 edges of the outer heptagon.
    Parameter for each inner point: t_i in [0, 1], where t_i=0 is vertex i and t_i=1 is vertex i+1.
    Plus a global rotation for the whole set? The outer heptagon orientation doesn't matter due to rotation invariance, so we can fix it.
    But maybe we can rotate the "inner" set relative to the outer? If they are constrained to edges, rotation is implicit in the choice of which edge and position.
    
    Let's try this specific structure:
    Outer: P_k = (cos(2*pi*k/7), sin(2*pi*k/7)) for k=0..6
    Inner: Q_k = (1-t_k)*P_k + t_k*P_{(k+1)%7} for k=0..6
    
    Parameters: t_0, ..., t_6 (7 parameters)
    This is a very low-dimensional space (7D).
    
    Let's also consider a more general "two-ring" approach where both rings are regular polygons but we optimize radii and rotation.
    Outer: radius R1, rotation 0.
    Inner: radius R2, rotation phi.
    Points: 7 on outer, 7 on inner.
    Parameters: R1, R2, phi (3 parameters).
    
    Let's combine:
    We will optimize a mix.
    Structure A: Two regular heptagons (R1, R2, phi).
    Structure B: Outer heptagon + 7 points on edges (t0..t6).
    
    Let's implement Structure B first as it's the specific directive.
    """
    # params: [t0, t1, t2, t3, t4, t5, t6]
    t = params[:7]
    # Clip t to [0, 1]
    t = np.clip(t, 0.0, 1.0)
    
    # Outer heptagon vertices
    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer_x = np.cos(angles)
    outer_y = np.sin(angles)
    outer_pts = np.column_stack((outer_x, outer_y))
    
    # Inner points on edges
    # Edge k connects outer_pts[k] and outer_pts[(k+1)%7]
    inner_pts = np.zeros((7, 2))
    for k in range(7):
        p1 = outer_pts[k]
        p2 = outer_pts[(k + 1) % 7]
        inner_pts[k] = (1 - t[k]) * p1 + t[k] * p2
        
    return np.vstack([outer_pts, inner_pts])

def neg_soft_two_heptagon(params):
    pts = _generate_two_heptagon_config(params)
    return soft_neg_score(pts, sharpness=30.0)

def _generate_two_ring_config(params):
    """
    Two regular heptagons.
    params: [R1, R2, phi]
    R1: radius outer (fixed to 1? or optimize? Let's fix R1=1 to reduce dims, or optimize R1, R2, phi)
    Actually, scale invariance means we can fix one radius. Let's fix R1=1.
    Then optimize R2 (0 to 1) and phi (0 to 2*pi/7, since 2*pi/7 is the symmetry period).
    """
    # params: [R2, phi]
    R2 = params[0]
    phi = params[1]
    
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer_pts = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
    
    angles_inner = angles_outer + phi
    inner_pts = R2 * np.column_stack((np.cos(angles_inner), np.sin(angles_inner)))
    
    return np.vstack([outer_pts, inner_pts])

def neg_soft_two_ring(params):
    pts = _generate_two_ring_config(params)
    return soft_neg_score(pts, sharpness=30.0)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_points = None
    best_score = -float('inf')
    
    def try_update(points):
        nonlocal best_points, best_score
        if points is None:
            return
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s

    # Strategy 1: Optimize the "points on edges" structure (7D)
    if time.time() < deadline - 5:
        # Differential Evolution on 7D space
        bounds = [(0, 1)] * 7
        try:
            res = differential_evolution(
                neg_soft_two_heptagon,
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-5,
                popsize=10,
                mutation=(0.5, 1.5),
                recombination=0.9,
                polish=True,
                init='latinhypercube'
            )
            best_params = res.x
            pts = _generate_two_heptagon_config(best_params)
            try_update(pts)
        except Exception:
            pass

    # Strategy 2: Optimize the "two regular heptagons" structure (2D)
    if time.time() < deadline - 5:
        bounds = [(0.1, 1.0), (0, 2 * np.pi / 7)]
        try:
            res = differential_evolution(
                neg_soft_two_ring,
                bounds,
                seed=43,
                maxiter=100,
                tol=1e-5,
                popsize=10,
                mutation=(0.5, 1.5),
                recombination=0.9,
                polish=True,
                init='latinhypercube'
            )
            best_params = res.x
            pts = _generate_two_ring_config(best_params)
            try_update(pts)
        except Exception:
            pass

    # Strategy 3: General 28D optimization starting from best found so far
    if best_points is not None and time.time() < deadline - 10:
        # Use the best points found so far as initial guess for full space optimization
        x0 = best_points.flatten()
        
        # Use Nelder-Mead for local refinement
        try:
            res = minimize(
                neg_soft_score,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False}
            )
            if res.fun < -best_score: # Note: neg_soft_score is not the same scale as neg_score
                # We need to compare actual scores
                pts_candidate = res.x.reshape((N, 2))
                s_candidate = compute_score(pts_candidate)
                if s_candidate > best_score:
                    best_points = pts_candidate
                    best_score = s_candidate
        except Exception:
            pass

    # Strategy 4: If time remains, try a few random restarts in full space with soft score
    if time.time() < deadline - 10:
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            # Random initial points
            x0 = np.random.uniform(-1, 1, (N, 2)).flatten()
            try:
                res = minimize(
                    neg_soft_score,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False}
                )
                pts_candidate = res.x.reshape((N, 2))
                s_candidate = compute_score(pts_candidate)
                if s_candidate > best_score:
                    best_points = pts_candidate
                    best_score = s_candidate
            except Exception:
                pass

    # Fallback
    if best_points is None:
        # Regular 14-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points

# Test to ensure it runs
if __name__ == "__main__":
    pts = construct()
    print(f"Score: {compute_score(pts)}")
    print(f"Points:\n{pts}")
