import numpy as np
import time
import warnings
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

warnings.filterwarnings('ignore')

N = 13

_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _normalize(points):
    """Center and normalize points."""
    centroid = np.mean(points, axis=0)
    pts = points - centroid
    # Normalize so hull area = 1 for consistent comparison
    hull_area = _hull_area(pts)
    if hull_area > 1e-12:
        pts = pts / np.sqrt(hull_area)
    return pts

def _objective_full(x_flat):
    """Full objective: minimize negative score."""
    points = x_flat.reshape(N, 2)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _logsumexp_smooth_obj(x_flat, alpha=50.0):
    """Smooth surrogate using log-sum-exp for soft-min of triangle areas.
    
    Uses -log(sum(exp(-alpha * areas))) as a smooth approximation to -min(areas).
    This is smoother than power-mean for large alpha.
    """
    points = x_flat.reshape(N, 2)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    if np.allclose(points, 0):
        return 1e10
    
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    areas = np.maximum(areas, 1e-15)
    
    # Soft min via log-sum-exp: -log(sum(exp(-alpha*a))) ≈ -min(a) - (1/alpha)*log(n)
    neg_alpha_areas = -alpha * areas
    max_val = np.max(neg_alpha_areas)
    soft_min_neg = -max_val - np.log(np.sum(np.exp(neg_alpha_areas - max_val)))
    # soft_min_neg ≈ -min(areas)
    soft_min_val = -soft_min_neg
    
    score = soft_min_val / hull_area
    if score <= 0:
        return 1e10
    return -score

def _coordinate_descent_polish(x_flat, deadline, best_score_ref):
    """Iterate over coordinates, doing line search along each axis."""
    best_x = x_flat.copy()
    best_s = _score(best_x.reshape(N, 2))
    
    for coord in range(N * 2):
        if time.time() > deadline - 3:
            break
        
        point_idx = coord // 2
        dim = coord % 2
        
        # Try perturbations in decreasing step sizes
        for step in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]:
            for direction in [-1, 1]:
                candidate = best_x.copy()
                candidate[coord] += direction * step
                
                s = _score(candidate.reshape(N, 2))
                if s > best_s:
                    best_s = s
                    best_x = candidate.copy()
                else:
                    break
    
    return best_x, best_s

def _local_polish(x_flat, deadline, best_score_ref):
    """Multi-stage local polish: smooth gradient-free, then coordinate descent, then NM."""
    best_x = x_flat.copy()
    best_s = _score(best_x.reshape(N, 2))
    
    # Stage 1: Smooth optimization with log-sum-exp
    if time.time() < deadline - 15:
        try:
            for alpha in [30.0, 80.0, 150.0]:
                if time.time() > deadline - 10:
                    break
                res = minimize(
                    lambda x: _logsumexp_smooth_obj(x, alpha=alpha),
                    best_x,
                    method='Powell',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                s = _score(res.x.reshape(N, 2))
                if s > best_s:
                    best_s = s
                    best_x = res.x.copy()
        except Exception:
            pass
    
    # Stage 2: Coordinate descent
    if time.time() < deadline - 5:
        best_x, best_s = _coordinate_descent_polish(best_x, deadline, best_s)
    
    # Stage 3: Nelder-Mead on hard objective
    if time.time() < deadline - 2:
        try:
            res = minimize(
                _objective_full,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_s:
                best_s = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x, best_s

def _make_ring_config(n_outer, n_inner, r_inner, angle_offset):
    """Create a two-ring configuration."""
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_inner, endpoint=False)
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    return np.vstack([outer_pts, inner_pts])

def _optimize_from_start(x0, deadline, best_score_ref):
    """Run multi-phase optimization from a starting point."""
    best_x, best_s = _local_polish(x0, deadline, best_score_ref)
    return best_x, best_s

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None or len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]).flatten())
    
    # 2. Two-ring configurations with various parameters
    for n_out in [6, 7, 8]:
        n_in = N - n_out
        for r_in in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for offset in [0, np.pi / n_out, np.pi / (2 * n_out)]:
                pts = _make_ring_config(n_out, n_in, r_in, offset)
                inits.append(pts.flatten())
    
    # 3. Three-ring configurations
    for n1, n2, n3 in [(5, 4, 4), (6, 5, 2), (7, 3, 3), (4, 4, 5)]:
        for r2 in [0.5, 0.6, 0.7]:
            for r3 in [0.2, 0.3, 0.4]:
                a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
                a2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n1
                a3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + np.pi / n2
                pts = np.vstack([
                    np.column_stack([np.cos(a1), np.sin(a1)]),
                    np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
                    np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)])
                ])
                inits.append(pts.flatten())
    
    # 4. Random starts
    rng = np.random.default_rng(12345)
    for _ in range(20):
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts.flatten())
    
    # 5. Near-regular with small perturbations
    for pert in [0.01, 0.03, 0.05]:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        r_pert = 1.0 + rng.normal(0, pert, N)
        a_pert = angles + rng.normal(0, pert / 2, N)
        pts = np.column_stack([r_pert * np.cos(a_pert), r_pert * np.sin(a_pert)])
        inits.append(pts.flatten())
    
    # Shuffle inits to diversify
    rng2 = np.random.default_rng(999)
    indices = rng2.permutation(len(inits))
    inits = [inits[i] for i in indices]
    
    # Optimize from each start
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 12:
            break
        
        t_start = time.time()
        x_opt, s_opt = _optimize_from_start(x0, deadline, best_score)
        _update_best(x_opt.reshape(N, 2))
        
        elapsed = time.time() - t_start
        if i % 5 == 0:
            print(f"Init {i}: score={best_score:.6f}, elapsed={elapsed:.1f}s", flush=True)
    
    # Final multi-stage polish on best
    if best_points is not None and time.time() < deadline - 3:
        try:
            # Tighter Nelder-Mead
            res_final = minimize(
                _objective_full,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            _update_best(res_final.x.reshape(N, 2))
        except Exception:
            pass
        
        # Coordinate descent on the best
        if best_points is not None and time.time() < deadline - 1:
            try:
                bx, bs = _coordinate_descent_polish(best_points.flatten(), deadline, best_score)
                _update_best(bx.reshape(N, 2))
            except Exception:
                pass
    
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
