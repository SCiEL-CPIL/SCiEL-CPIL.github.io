import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

# Precompute triple indices for vectorized area computation
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

    min_area = 0.5 * np.min(np.abs(cross))
    return min_area / hull_area

def _soft_min_score(points, temperature=0.05):
    """Smooth surrogate using log-sum-exp for the minimum triangle area."""
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    
    # Soft min: min(x) approx -T * log(sum(exp(-x/T)))
    # We want to maximize min(areas) / hull_area
    # So we minimize - (soft_min(areas) / hull_area)
    
    # To avoid overflow in exp, shift by min_area
    shifted = areas - min_area
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    soft_min_val = min_area - temperature * log_sum
    
    return soft_min_val / hull_area

def _generate_points_from_fourier(params, K=3):
    """
    Generate N points from Fourier coefficients.
    params: [a1, b1, ..., aK, bK, global_scale]
    The radius is r(theta) = global_scale * exp(sum(ak cos(k theta) + bk sin(k theta)))
    Angles are fixed: theta_i = 2*pi*i/N
    """
    a_coeffs = params[:K]
    b_coeffs = params[K:2*K]
    global_scale = params[2*K]
    
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    log_r = np.zeros(N)
    for k in range(1, K+1):
        log_r += a_coeffs[k-1] * np.cos(k * theta)
        log_r += b_coeffs[k-1] * np.sin(k * theta)
        
    r = global_scale * np.exp(log_r)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return np.column_stack((x, y))

def _optimize_fourier(time_limit=20.0, K=3):
    """
    Optimize Fourier coefficients to maximize soft-min score.
    """
    start_time = time.time()
    
    def neg_soft_score(params):
        if time.time() - start_time > time_limit:
            return 1e6
        try:
            pts = _generate_points_from_fourier(params, K)
            # Check convexity roughly: all cross products of consecutive edges should have same sign
            # For a convex polygon centered at origin with r>0, this is likely if coeffs are small.
            # We rely on the optimizer staying in a valid region.
            score = _soft_min_score(pts, temperature=0.05)
            return -score
        except Exception:
            return 1e6

    best_score = 0
    best_params = None
    rng = np.random.RandomState(42)
    
    # Number of restarts
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit - 2:
            break
        
        # Initialize parameters
        if restart == 0:
            # Start with regular polygon (all coeffs 0)
            params0 = np.zeros(2*K + 1)
            params0[2*K] = 1.0 # scale
        else:
            # Random small perturbations
            noise_scale = 0.1 + 0.05 * (restart % 5)
            a0 = rng.normal(0, noise_scale, K)
            b0 = rng.normal(0, noise_scale, K)
            scale0 = 1.0 + rng.normal(0, 0.1)
            params0 = np.concatenate([a0, b0, [scale0]])
        
        try:
            res = minimize(neg_soft_score, params0, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20})
            
            if res.fun < -best_score:
                best_score = -res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is not None:
        pts = _generate_points_from_fourier(best_params, K)
        final_score = _calculate_score(pts)
        return final_score, pts
    
    return 0.0, None

def _polish_points_cartesian(points, time_limit=5.0):
    """
    Fine-tune points in Cartesian space using coordinate descent on the exact min area.
    """
    start_time = time.time()
    pts = points.copy()
    
    # Compute current areas
    p_i = pts[TRIPLE_INDICES[:, 0]]
    p_j = pts[TRIPLE_INDICES[:, 1]]
    p_k = pts[TRIPLE_INDICES[:, 2]]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    min_area = np.min(areas)
    
    # Precompute triples for each point
    POINT_TRIPLES = [[] for _ in range(N)]
    for idx, (i, j, k) in enumerate(TRIPLE_INDICES):
        POINT_TRIPLES[i].append(idx)
        POINT_TRIPLES[j].append(idx)
        POINT_TRIPLES[k].append(idx)
    POINT_TRIPLES = [np.array(pt) for pt in POINT_TRIPLES]
    
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for step in step_sizes:
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(N):
                tri_idx = POINT_TRIPLES[i]
                best_move = None
                best_min_area = min_area
                p_i_curr = pts[i]
                
                # Try 8 directions + opposite
                directions = []
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx != 0 or dy != 0:
                            directions.append(np.array([dx, dy]))
                
                for d in directions:
                    # Forward
                    new_p = p_i_curr + d
                    new_areas = areas.copy()
                    for t_idx in tri_idx:
                        i_t, j_t, k_t = TRIPLE_INDICES[t_idx]
                        p_a = pts[i_t] if i_t != i else new_p
                        p_b = pts[j_t] if j_t != i else new_p
                        p_c = pts[k_t] if k_t != i else new_p
                        v1_t = p_b - p_a
                        v2_t = p_c - p_a
                        cross_t = v1_t[0] * v2_t[1] - v1_t[1] * v2_t[0]
                        new_areas[t_idx] = 0.5 * np.abs(cross_t)
                    
                    curr_min = np.min(new_areas)
                    if curr_min > best_min_area:
                        best_min_area = curr_min
                        best_move = new_p
                    
                    # Backward
                    new_p2 = p_i_curr - d
                    new_areas2 = areas.copy()
                    for t_idx in tri_idx:
                        i_t, j_t, k_t = TRIPLE_INDICES[t_idx]
                        p_a = pts[i_t] if i_t != i else new_p2
                        p_b = pts[j_t] if j_t != i else new_p2
                        p_c = pts[k_t] if k_t != i else new_p2
                        v1_t = p_b - p_a
                        v2_t = p_c - p_a
                        cross_t = v1_t[0] * v2_t[1] - v1_t[1] * v2_t[0]
                        new_areas2[t_idx] = 0.5 * np.abs(cross_t)
                    
                    curr_min2 = np.min(new_areas2)
                    if curr_min2 > best_min_area:
                        best_min_area = curr_min2
                        best_move = new_p2
                
                if best_move is not None:
                    pts[i] = best_move
                    improved = True
                    # Recompute all areas
                    p_i = pts[TRIPLE_INDICES[:, 0]]
                    p_j = pts[TRIPLE_INDICES[:, 1]]
                    p_k = pts[TRIPLE_INDICES[:, 2]]
                    v1 = p_j - p_i
                    v2 = p_k - p_i
                    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
                    areas = 0.5 * np.abs(cross)
                    min_area = np.min(areas)
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 0
    best_points = None
    
    def try_update(score, points):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()
            
    # 1. Baseline: Regular Polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    score_reg = _calculate_score(pts_reg)
    try_update(score_reg, pts_reg)
    
    # 2. Fourier Optimization
    if time.time() < deadline - 30:
        # Use a moderate amount of time for Fourier search
        fourier_time = min(40.0, deadline - time.time() - 20)
        f_score, f_pts = _optimize_fourier(time_limit=fourier_time, K=3)
        if f_pts is not None:
            try_update(f_score, f_pts)
            
            # Polish the Fourier result in Cartesian space
            if time.time() < deadline - 10:
                polish_time = min(15.0, deadline - time.time() - 5)
                polished_pts = _polish_points_cartesian(f_pts, time_limit=polish_time)
                polished_score = _calculate_score(polished_pts)
                try_update(polished_score, polished_pts)
                
    # 3. Try higher Fourier modes if time permits
    if time.time() < deadline - 20:
        f2_score, f2_pts = _optimize_fourier(time_limit=min(20.0, deadline - time.time() - 10), K=4)
        if f2_pts is not None:
            try_update(f2_score, f2_pts)
            if time.time() < deadline - 5:
                polished_pts2 = _polish_points_cartesian(f2_pts, time_limit=min(10.0, deadline - time.time() - 2))
                polished_score2 = _calculate_score(polished_pts2)
                try_update(polished_score2, polished_pts2)

    # 4. Final polish on the absolute best found
    if best_points is not None and time.time() < deadline - 2:
        remaining = deadline - time.time() - 1
        final_pts = _polish_points_cartesian(best_points, time_limit=remaining)
        final_score = _calculate_score(final_pts)
        try_update(final_score, final_pts)

    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
