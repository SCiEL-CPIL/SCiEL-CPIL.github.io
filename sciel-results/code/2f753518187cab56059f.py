"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

# Pre-compute indices for N=14 for fast triangle area calculation
_IDX_CACHE = {}
def _get_indices(n):
    if n not in _IDX_CACHE:
        i_idx, j_idx, k_idx = [], [], []
        for i in range(n-2):
            for j in range(i+1, n-1):
                for k in range(j+1, n):
                    i_idx.append(i)
                    j_idx.append(j)
                    k_idx.append(k)
        _IDX_CACHE[n] = (np.array(i_idx), np.array(j_idx), np.array(k_idx))
    return _IDX_CACHE[n]

def _min_triangle_area(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    i_idx, j_idx, k_idx = _get_indices(n)
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _convex_hull_area(pts):
    try:
        if np.allclose(pts[:, 0], pts[0, 0]) or np.allclose(pts[:, 1], pts[0, 1]):
            return 1e-10
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10

def _score(pts):
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area

def _soft_min_area(pts, beta):
    n = len(pts)
    if n < 3:
        return 0.0
    i_idx, j_idx, k_idx = _get_indices(n)
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    min_a = np.min(areas)
    shifted = areas - min_a
    max_shift = np.max(shifted)
    if max_shift > 50:
        return min_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return min_a - (1.0 / beta) * log_sum

def _soft_score(pts, beta):
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-9:
        return 0.0
    soft_min = _soft_min_area(pts, beta)
    return soft_min / hull_area

def _make_ring(n_pts, radius, angle_offset=0.0):
    """Generate n_pts points on a circle of given radius with angular offset."""
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + angle_offset
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _three_ring_points(r1, r2, a1, a2, a3):
    """
    14 points in three rings: 5 inner, 4 middle, 5 outer.
    r1, r2: radii of inner and middle rings (outer is 1.0)
    a1, a2, a3: angular offsets for each ring
    """
    pts1 = _make_ring(5, r1, a1)
    pts2 = _make_ring(4, r2, a2)
    pts3 = _make_ring(5, 1.0, a3)
    return np.vstack([pts1, pts2, pts3])

def _two_ring_points(r, alpha):
    """14 points: 7 inner at radius r, 7 outer at radius 1.0, staggered by alpha."""
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_inner = angles_outer + alpha
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = np.column_stack([r * np.cos(angles_inner), r * np.sin(angles_inner)])
    return np.vstack([outer_pts, inner_pts])

def _coordinate_descent_polish(points, time_limit=2.0):
    """
    Iteratively adjust each point to improve the minimum triangle area.
    Uses fast local updates for triples involving the moved point.
    """
    start_time = time.time()
    pts = points.copy()
    
    # Precompute current areas
    i_idx, j_idx, k_idx = _get_indices(N)
    p_i = pts[i_idx]
    p_j = pts[j_idx]
    p_k = pts[k_idx]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    min_area = np.min(areas)
    
    # Precompute which triples involve each point
    POINT_TRIPLES = [[] for _ in range(N)]
    for idx in range(len(i_idx)):
        POINT_TRIPLES[i_idx[idx]].append(idx)
        POINT_TRIPLES[j_idx[idx]].append(idx)
        POINT_TRIPLES[k_idx[idx]].append(idx)
    POINT_TRIPLES = [np.array(pt) for pt in POINT_TRIPLES]
    
    step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for step in step_sizes:
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(N):
                # Get indices of triples involving point i
                tri_idx = POINT_TRIPLES[i]
                
                best_move = None
                best_min_area = min_area
                
                # Current position
                p_i_curr = pts[i]
                
                # Try 8 directions
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        new_p = p_i_curr + np.array([dx, dy])
                        
                        # Update areas for triples involving i
                        new_areas = areas.copy()
                        
                        for t_idx in tri_idx:
                            i_t = i_idx[t_idx]
                            j_t = j_idx[t_idx]
                            k_t = k_idx[t_idx]
                            # Get points
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
                
                if best_move is not None:
                    pts[i] = best_move
                    min_area = best_min_area
                    improved = True
                    
                    # Recompute all areas
                    p_i = pts[i_idx]
                    p_j = pts[j_idx]
                    p_k = pts[k_idx]
                    v1 = p_j - p_i
                    v2 = p_k - p_i
                    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
                    areas = 0.5 * np.abs(cross)
                    min_area = np.min(areas)
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_score = -np.inf
    best_pts = None

    def update_best(pts):
        nonlocal best_score, best_pts
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
            return True
        return False

    # 1. Baseline: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    update_best(reg_pts)

    # 2. Three-ring optimization: 5-4-5 configuration
    # Parameters: r1 (inner radius), r2 (middle radius), a1, a2, a3 (angular offsets)
    def neg_three_ring(params):
        r1, r2, a1, a2, a3 = params
        pts = _three_ring_points(r1, r2, a1, a2, a3)
        return -_score(pts)

    if time.time() < deadline - 50.0:
        try:
            bounds = [(0.1, 0.6), (0.5, 0.85), (0.0, 2*np.pi/5), (0.0, 2*np.pi/4), (0.0, 2*np.pi/5)]
            res = differential_evolution(
                neg_three_ring,
                bounds,
                seed=42,
                maxiter=80,
                tol=1e-8,
                polish=True
            )
            if res.fun < 0:
                pts = _three_ring_points(*res.x)
                update_best(pts)
        except Exception:
            pass

    # 3. Two-ring optimization: 7-7 configuration
    def neg_two_ring(params):
        r, alpha = params
        pts = _two_ring_points(r, alpha)
        return -_score(pts)

    if time.time() < deadline - 40.0:
        try:
            bounds = [(0.2, 0.9), (0.0, 2*np.pi/7)]
            res = differential_evolution(
                neg_two_ring,
                bounds,
                seed=42,
                maxiter=80,
                tol=1e-8,
                polish=True
            )
            if res.fun < 0:
                pts = _two_ring_points(*res.x)
                update_best(pts)
        except Exception:
            pass

    # 4. Alternative three-ring: 4-5-5 configuration
    def _three_ring_455(r1, r2, a1, a2, a3):
        pts1 = _make_ring(4, r1, a1)
        pts2 = _make_ring(5, r2, a2)
        pts3 = _make_ring(5, 1.0, a3)
        return np.vstack([pts1, pts2, pts3])

    def neg_three_ring_455(params):
        r1, r2, a1, a2, a3 = params
        pts = _three_ring_455(r1, r2, a1, a2, a3)
        return -_score(pts)

    if time.time() < deadline - 30.0:
        try:
            bounds = [(0.1, 0.5), (0.5, 0.85), (0.0, 2*np.pi/4), (0.0, 2*np.pi/5), (0.0, 2*np.pi/5)]
            res = differential_evolution(
                neg_three_ring_455,
                bounds,
                seed=123,
                maxiter=60,
                tol=1e-8,
                polish=True
            )
            if res.fun < 0:
                pts = _three_ring_455(*res.x)
                update_best(pts)
        except Exception:
            pass

    # 5. Local refinement using soft-min continuation
    if best_pts is not None and time.time() < deadline - 20.0:
        def neg_soft_score_full(x, beta):
            pts = x.reshape(-1, 2)
            return -_soft_score(pts, beta)

        current_pts = best_pts.copy()
        current_score = best_score

        betas = [5.0, 20.0, 100.0, 500.0]
        for beta in betas:
            if time.time() > deadline - 10.0:
                break
            try:
                res = minimize(
                    neg_soft_score_full,
                    current_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                new_score = _score(res.x.reshape(-1, 2))
                if new_score > current_score:
                    current_pts = res.x.reshape(-1, 2)
                    current_score = new_score
                    update_best(current_pts)
            except Exception:
                pass

        # Final polish with hard score
        if time.time() < deadline - 5.0:
            try:
                def neg_hard_score_full(x):
                    pts = x.reshape(-1, 2)
                    return -_score(pts)
                res = minimize(
                    neg_hard_score_full,
                    current_pts.flatten(),
                    method='Powell',
                    options={'maxiter': 100, 'xtol': 1e-9, 'ftol': 1e-12}
                )
                if res.fun < -current_score:
                    current_pts = res.x.reshape(-1, 2)
                    current_score = -res.fun
                    update_best(current_pts)
            except Exception:
                pass

    # 6. Coordinate descent polish
    if best_pts is not None and time.time() < deadline - 10.0:
        try:
            polished_pts = _coordinate_descent_polish(best_pts, time_limit=8.0)
            update_best(polished_pts)
        except Exception:
            pass

    # 7. Random restarts with local refinement
    if time.time() < deadline - 15.0:
        rng = np.random.RandomState(123)
        for restart in range(5):
            if time.time() > deadline - 10.0:
                break
            noise = rng.normal(0, 0.05, size=(N, 2))
            init_pts = best_pts + noise
            update_best(init_pts)
            if time.time() > deadline - 8.0:
                continue
            try:
                def neg_soft_restart(x, beta=50.0):
                    pts = x.reshape(-1, 2)
                    return -_soft_score(pts, beta)
                res = minimize(
                    neg_soft_restart,
                    init_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                if res.fun < -best_score:
                    new_pts = res.x.reshape(-1, 2)
                    update_best(new_pts)
            except Exception:
                pass

    if best_pts is None:
        return reg_pts
    return best_pts
# EVOLVE-BLOCK-END
