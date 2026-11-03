"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

DEADLINE = 140.0  # Safety margin before 150s kill


def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0


def _min_triangle_area_vec(points):
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        xi, yi = x[i], y[i]
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - xi
            dy_ki = y[j+1:] - yi
            cross = (x[j] - xi) * dy_ki - dx_ki * (y[j] - yi)
            local_min = 0.5 * np.min(np.abs(cross))
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-14:
                    return min_area
    return min_area


def _all_triangle_areas(points):
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n - 2):
        xi, yi = x[i], y[i]
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - xi
            dy_ki = y[j+1:] - yi
            cross = (x[j] - xi) * dy_ki - dx_ki * (y[j] - yi)
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)


def _score(pts):
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area_vec(pts)
    return min_area / hull_a


def _neg_score(pts_flat):
    pts = pts_flat.reshape(-1, 2)
    return -_score(pts)


def _soft_obj(pts_flat, beta=150.0):
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val


def _multi_beta_soft_obj(pts_flat, betas=(50.0, 150.0, 300.0)):
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    total = 0.0
    for beta in betas:
        z = -beta * areas
        max_z = np.max(z)
        if max_z < -500:
            log_sum = max_z
        else:
            log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
        soft_min_val = -log_sum / beta
        if soft_min_val < 1e-12:
            return 1e10
        total += hull_area / soft_min_val
    return total / len(betas)


def _normalize_unit_area(pts):
    hull_a = _hull_area(pts)
    if hull_a > 0:
        return pts / np.sqrt(hull_a)
    return pts


# ---------- Lattice parameterization (from current program) ----------
def _lattice_params_to_points(params):
    """params: [b1x, b1y, b2x, b2y, u1..u7, v1..v7] -> 14 points (7 pairs ±)"""
    b1 = np.array([params[0], params[1]])
    b2 = np.array([params[2], params[3]])
    u = params[4:11]
    v = params[11:18]
    half = np.array([u[i]*b1 + v[i]*b2 for i in range(7)])
    return np.vstack([half, -half])


def _objective_lattice(params):
    if np.any(~np.isfinite(params)):
        return 1e10
    pts = _lattice_params_to_points(params)
    if np.any(~np.isfinite(pts)):
        return 1e10
    m = _min_triangle_area_vec(pts)
    if m < 1e-12:
        return 1e10
    h = _hull_area(pts)
    if h < 1e-10:
        return 1e10
    return h / m


def _objective_lattice_soft(params, beta=150.0):
    if np.any(~np.isfinite(params)):
        return 1e10
    pts = _lattice_params_to_points(params)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_triangle_areas(pts)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    mz = np.max(z)
    if mz < -500:
        lse = mz
    else:
        lse = mz + np.log(np.sum(np.exp(z - mz)))
    sm = -lse / beta
    if sm < 1e-12:
        return 1e10
    h = _hull_area(pts)
    if h < 1e-10:
        return 1e10
    return h / sm


def _hex_initializers():
    """Generate diverse lattice-based initial parameters."""
    inits = []
    s3 = np.sqrt(3)
    
    patterns = [
        [(0,0), (1,0), (0,1), (1,1), (2,0), (0,2), (2,1)],
        [(0,0), (1,0), (0,1), (1,1), (2,0), (0,2), (1,2)],
        [(0,0), (1,0), (0,1), (1,1), (2,0), (0,2), (2,2)],
        [(1,0), (0,1), (1,1), (2,0), (0,2), (2,1), (1,2)],
        [(0,0), (2,0), (0,2), (2,2), (1,0), (0,1), (1,1)],
        [(1,1), (2,0), (0,2), (2,2), (1,0), (0,1), (3,1)],
        [(0,0), (1,0), (0,1), (1,1), (2,1), (1,2), (2,2)],
        [(1,0), (1,1), (2,0), (2,1), (0,1), (0,2), (1,2)],
    ]
    
    for pattern in patterns:
        u = np.array([p[0] for p in pattern], dtype=float)
        v = np.array([p[1] for p in pattern], dtype=float)
        b1 = np.array([1.0, 0.0])
        b2 = np.array([0.5, s3/2.0])
        params = np.concatenate([[b1[0], b1[1], b2[0], b2[1]], u, v])
        inits.append(params)
    
    for rot in [0.1, -0.1, 0.2, -0.2, 0.3]:
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])
        b1 = R @ np.array([1.0, 0.0])
        b2 = R @ np.array([0.5, s3/2.0])
        pattern = [(0,0), (1,0), (0,1), (1,1), (2,0), (0,2), (1,2)]
        u = np.array([p[0] for p in pattern], dtype=float)
        v = np.array([p[1] for p in pattern], dtype=float)
        params = np.concatenate([[b1[0], b1[1], b2[0], b2[1]], u, v])
        inits.append(params)
    
    rng = np.random.default_rng(7)
    for seed in range(12):
        rng = np.random.default_rng(seed * 13 + 3)
        b1 = np.array([1.0, 0.0]) + rng.normal(0, 0.15, 2)
        b2 = np.array([0.5, s3/2.0]) + rng.normal(0, 0.15, 2)
        pattern = [(0,0), (1,0), (0,1), (1,1), (2,0), (0,2), (1,2)]
        u = np.array([p[0] for p in pattern], dtype=float) + rng.normal(0, 0.1, 7)
        v = np.array([p[1] for p in pattern], dtype=float) + rng.normal(0, 0.1, 7)
        params = np.concatenate([[b1[0], b1[1], b2[0], b2[1]], u, v])
        inits.append(params)
    
    return inits


# ---------- Free-point initializers (from inspiration) ----------
def _initial_regular_polygon(n):
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return _normalize_unit_area(pts)


def _initial_mixed(n):
    outer_n = 10
    inner_n = n - outer_n
    angles = 2 * np.pi * np.arange(outer_n) / outer_n
    outer = np.column_stack([np.cos(angles), np.sin(angles)])
    inner_angles = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    inner = 0.4 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    pts = np.vstack([outer, inner])
    return _normalize_unit_area(pts)


def _initial_hexagonal(n):
    pts = [[0.0, 0.0]]
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append([np.cos(angle), np.sin(angle)])
    for i in range(7):
        angle = 2 * np.pi * i / 7 + np.pi / 7
        r = 1.8
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    pts = np.array(pts)
    return _normalize_unit_area(pts)


# ---------- Directed push-pull local search ----------
def _worst_triangle_indices(points):
    """Find indices of the triangle with minimum area."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    min_area = float('inf')
    min_idx = (0, 1, 2)
    for i in range(n - 2):
        xi, yi = x[i], y[i]
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - xi
            dy_ki = y[j+1:] - yi
            cross = (x[j] - xi) * dy_ki - dx_ki * (y[j] - yi)
            abs_cross = np.abs(cross)
            k_local = np.argmin(abs_cross)
            k = j + 1 + k_local
            area = 0.5 * abs_cross[k_local]
            if area < min_area:
                min_area = area
                min_idx = (i, j, k)
    return min_idx


def _push_pull_step(points, step_size=0.05, n_iter=50, deadline=None):
    """
    Directed push-pull: identify worst triangle, push its vertices apart
    to increase its area while maintaining overall structure.
    """
    pts = points.copy()
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline - 2.0:
            break
        
        i, j, k = _worst_triangle_indices(pts)
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        
        # Centroid of the worst triangle
        centroid = (pi + pj + pk) / 3.0
        
        # Push each vertex away from centroid
        for idx, p in [(i, pi), (j, pj), (k, pk)]:
            direction = p - centroid
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
                pts[idx] = p + step_size * direction
        
        # Also push other points slightly away from the worst triangle centroid
        # to avoid creating new small triangles
        for idx in range(len(pts)):
            if idx not in (i, j, k):
                direction = pts[idx] - centroid
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction = direction / norm
                    pts[idx] = pts[idx] + 0.3 * step_size * direction
        
        # Normalize to maintain unit area
        pts = _normalize_unit_area(pts)
        
        # Check if score improved
        current_score = _score(pts)
        if current_score <= 0:
            break
    
    return pts


# ---------- Main ----------
def construct():
    start_time = time.time()
    
    best_score = 0.0
    best_pts = None
    
    def _consider(pts):
        nonlocal best_score, best_pts
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Phase 1: Generate initial candidates from both lattice and free-point approaches
    candidates = []
    
    # Lattice-based initializers
    lattice_inits = _hex_initializers()
    for params in lattice_inits[:8]:  # Limit to keep time bounded
        if time.time() - start_time > DEADLINE - 60:
            break
        pts = _lattice_params_to_points(params)
        candidates.append(pts)
    
    # Free-point initializers
    candidates.append(_initial_regular_polygon(N))
    candidates.append(_initial_mixed(N))
    candidates.append(_initial_hexagonal(N))
    
    # Add perturbed versions
    rng = np.random.RandomState(42)
    base = _initial_regular_polygon(N)
    for _ in range(3):
        if time.time() - start_time > DEADLINE - 60:
            break
        noise = rng.normal(0, 0.05, size=base.shape)
        perturbed = base + noise
        candidates.append(_normalize_unit_area(perturbed))
    
    # Phase 2: Optimize each candidate using multi-stage approach
    for cand in candidates:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        x0 = cand.flatten()
        
        # Stage A: Multi-beta soft objective (smooth landscape)
        if time.time() - start_time < DEADLINE - 15:
            try:
                res = minimize(_multi_beta_soft_obj, x0, method='Nelder-Mead',
                              options={'maxiter': 1500, 'xatol': 1e-6, 'fatol': 1e-8,
                                      'adaptive': True})
                if res.x is not None:
                    _consider(res.x.reshape(-1, 2))
                    x0 = res.x
            except Exception:
                pass
        
        # Stage B: Single beta soft objective
        if time.time() - start_time < DEADLINE - 12:
            try:
                res = minimize(lambda x: _soft_obj(x, beta=200.0), x0, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8,
                                      'adaptive': True})
                if res.x is not None:
                    _consider(res.x.reshape(-1, 2))
                    x0 = res.x
            except Exception:
                pass
        
        # Stage C: Hard objective for precise local optimization
        if time.time() - start_time < DEADLINE - 10:
            try:
                res = minimize(_neg_score, x0, method='Nelder-Mead',
                              options={'maxiter': 800, 'xatol': 1e-6, 'fatol': 1e-8,
                                      'adaptive': True})
                if res.x is not None:
                    _consider(res.x.reshape(-1, 2))
            except Exception:
                pass
        
        # Stage D: Directed push-pull local search
        if time.time() - start_time < DEADLINE - 8:
            if best_pts is not None:
                pushed = _push_pull_step(best_pts, step_size=0.03, n_iter=30,
                                         deadline=start_time + DEADLINE)
                _consider(pushed)
    
    # Phase 3: Global refinement on best solution found
    if best_pts is not None and time.time() - start_time < DEADLINE - 15:
        x0 = best_pts.flatten()
        
        # More aggressive multi-beta optimization
        try:
            res = minimize(_multi_beta_soft_obj, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9,
                                  'adaptive': True})
            if res.x is not None:
                _consider(res.x.reshape(-1, 2))
        except Exception:
            pass
        
        # Hard objective polish
        if time.time() - start_time < DEADLINE - 10:
            try:
                res = minimize(_neg_score, x0, method='Nelder-Mead',
                              options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-9,
                                      'adaptive': True})
                if res.x is not None:
                    _consider(res.x.reshape(-1, 2))
            except Exception:
                pass
    
    # Phase 4: Final directed push-pull on best solution
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        pushed = _push_pull_step(best_pts, step_size=0.02, n_iter=40,
                                 deadline=start_time + DEADLINE)
        _consider(pushed)
    
    # Fallback
    if best_pts is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    final_points = _normalize_unit_area(best_pts)
    
    # Safety check
    try:
        h = _hull_area(final_points)
        if h < 1e-10:
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))
    except Exception:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    return final_points
# EVOLVE-BLOCK-END
