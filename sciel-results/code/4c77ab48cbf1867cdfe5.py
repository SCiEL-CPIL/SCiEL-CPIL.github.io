"""
Heilbronn n=14: Maximize minimum triangle area / convex hull area.

Structural change: parameterize the 14 points as vertices of a strictly convex
polygon via edge-length + internal-angle representation, then optimize with
multi-start SLSQP on a soft-min objective, followed by hard-objective polish.
This avoids radial/Fourier bias and allows asymmetric convex shapes.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14
_START_TIME = time.time()
DEADLINE = 138.0


def _time_left():
    return DEADLINE - (time.time() - _START_TIME)


# Precompute triangle indices
_idx = np.arange(N)
_II, _JJ, _KK = np.meshgrid(_idx, _idx, _idx, indexing='ij')
_mask = (_II < _JJ) & (_JJ < _KK)
_T_I = _II[_mask].ravel()
_T_J = _JJ[_mask].ravel()
_T_K = _KK[_mask].ravel()
N_TRI = len(_T_I)


def _all_areas(points):
    pi = points[_T_I]
    pj = points[_T_J]
    pk = points[_T_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    return 0.5 * np.abs(v1x * v2y - v1y * v2x)


def _score(points):
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    if hull_area < 1e-12:
        return 0.0
    areas = _all_areas(points)
    min_a = np.min(areas)
    if min_a < 1e-14:
        return 0.0
    return min_a / hull_area


def _polygon_from_params(params):
    """
    params: (N+13,) = [e0,...,e13, a0,...,a12]
    e_i > 0 are edge lengths, a_i > 0 are internal angles summing to (N-2)*pi.
    Returns (N,2) points forming a strictly convex polygon.
    """
    edges = np.abs(params[:N]) + 0.01
    angles_raw = params[N:]
    angles = np.abs(angles_raw) + 0.01
    # Normalize angles to sum to (N-2)*pi
    target = (N - 2) * np.pi
    angles = angles * (target / angles.sum())
    
    # Turn angles (exterior) = pi - internal angle
    turn = np.pi - angles
    # Close the polygon: total turn should be 2*pi
    # Adjust last turn to close
    total_turn = turn.sum()
    turn[-1] = 2 * np.pi - turn[:-1].sum()
    
    # Build vertices by walking edges with turning angles
    pts = np.zeros((N, 2))
    pts[0] = [0.0, 0.0]
    heading = 0.0
    for i in range(N):
        if i > 0:
            heading += turn[i - 1]
        pts[i + 1 if i + 1 < N else 0] += np.array([np.cos(heading), np.sin(heading)]) * edges[i]
    
    # Actually let's rebuild properly
    pts = np.zeros((N, 2))
    heading = 0.0
    for i in range(N):
        pts[i] = pts[i - 1 if i > 0 else N - 1] + np.array([np.cos(heading), np.sin(heading)]) * edges[i]
        heading += turn[i]
    
    # Center and return
    centroid = pts.mean(axis=0)
    return pts - centroid


def _params_to_points_flat(params):
    """Convert flat param vector to flat point array."""
    pts = _polygon_from_params(params)
    return pts.flatten()


def _objective_soft(p, beta=40.0):
    points = _params_to_points_flat(p).reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 1e10
    if hull_area < 1e-10:
        return 1e10
    areas = _all_areas(points)
    min_a = np.min(areas)
    shifted = areas - min_a
    soft_min_val = min_a - (1.0 / beta) * np.log(np.sum(np.exp(-beta * shifted)))
    score = soft_min_val / hull_area
    return -score


def _objective_hard(p):
    points = _params_to_points_flat(p).reshape(N, 2)
    return -_score(points)


def _objective_soft_direct(p, beta=40.0):
    """Soft objective directly on 28-dim point coordinates."""
    points = p.reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 1e10
    if hull_area < 1e-10:
        return 1e10
    areas = _all_areas(points)
    min_a = np.min(areas)
    shifted = areas - min_a
    soft_min_val = min_a - (1.0 / beta) * np.log(np.sum(np.exp(-beta * shifted)))
    score = soft_min_val / hull_area
    return -score


def _objective_hard_direct(p):
    points = p.reshape(N, 2)
    return -_score(points)


def _generate_initials():
    inits = []
    rng = np.random.RandomState(123)
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(reg)
    
    # 2. Perturbed regular
    for seed in range(5):
        rng2 = np.random.RandomState(42 + seed)
        noise = rng2.uniform(-0.1, 0.1, (N, 2))
        inits.append(reg + noise)
    
    # 3. Asymmetric convex via random edge lengths and angles
    for seed in range(4):
        rng3 = np.random.RandomState(100 + seed)
        edges = rng3.uniform(0.5, 1.5, N)
        angles_raw = rng3.uniform(0.8, 1.4, N - 1)  # internal angles in radians
        params = np.concatenate([edges, angles_raw])
        pts = _polygon_from_params(params)
        inits.append(pts)
    
    # 4. Two-ring-like (slightly non-convex initial, will be projected)
    inner_a = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    outer_a = np.linspace(0, 2 * np.pi, 9, endpoint=False) + np.pi / 9
    inner = 0.4 * np.column_stack([np.cos(inner_a), np.sin(inner_a)])
    outer = np.column_stack([np.cos(outer_a), np.sin(outer_a)])
    inits.append(np.vstack((inner, outer)))
    
    # 5. Random points in unit disk
    for seed in range(3):
        rng4 = np.random.RandomState(200 + seed)
        r = np.sqrt(rng4.rand(N))
        theta = rng4.rand(N) * 2 * np.pi
        inits.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
    
    return inits


def _convex_hull_projection(points):
    """If points are not in convex position, try to fix via hull."""
    try:
        hull = ConvexHull(points)
        return points[hull.vertices]
    except Exception:
        return points


def construct():
    start = time.time()
    inits = _generate_initials()
    
    best_params = inits[0].flatten().copy()
    best_score = _score(inits[0])
    
    # Phase 1: SLSQP on direct 28-dim representation with soft objective
    for init in inits:
        if _time_left() < 30:
            break
        try:
            res = minimize(
                _objective_soft_direct,
                init.flatten(),
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            cand_score = _score(res.x.reshape(N, 2))
            if cand_score > best_score:
                best_score = cand_score
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: SLSQP on polygon parameterization from best direct result
    if _time_left() > 40:
        # Convert best points to polygon params approximately
        pts = best_params.reshape(N, 2)
        try:
            hull = ConvexHull(pts)
            ordered = pts[hull.vertices]
        except Exception:
            ordered = pts
        # Estimate edge lengths
        edges = np.zeros(N)
        for i in range(N):
            edges[i] = np.linalg.norm(ordered[(i + 1) % N] - ordered[i])
        # Estimate internal angles
        angles_est = np.zeros(N - 1)
        for i in range(N - 1):
            p0 = ordered[i]
            p1 = ordered[(i + 1) % N]
            p2 = ordered[(i + 2) % N]
            v1 = p0 - p1
            v2 = p2 - p1
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-15)
            cos_a = np.clip(cos_a, -1.0, 1.0)
            angles_est[i] = np.arccos(cos_a)
        
        poly_params = np.concatenate([edges, angles_est])
        try:
            res_poly = minimize(
                _objective_soft,
                poly_params,
                method='SLSQP',
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            cand_pts = _params_to_points_flat(res_poly.x).reshape(N, 2)
            cand_score = _score(cand_pts)
            if cand_score > best_score:
                best_score = cand_score
                best_params = cand_pts.flatten().copy()
        except Exception:
            pass
    
    # Phase 3: Multi-start from polygon params
    if _time_left() > 25:
        rng = np.random.RandomState(777)
        for _ in range(3):
            if _time_left() < 10:
                break
            try:
                base = best_params.reshape(N, 2)
                try:
                    hull = ConvexHull(base)
                    ordered = base[hull.vertices]
                except Exception:
                    ordered = base
                edges = np.zeros(N)
                for i in range(N):
                    edges[i] = np.linalg.norm(ordered[(i + 1) % N] - ordered[i])
                angles_est = np.zeros(N - 1)
                for i in range(N - 1):
                    p0 = ordered[i]
                    p1 = ordered[(i + 1) % N]
                    p2 = ordered[(i + 2) % N]
                    v1 = p0 - p1
                    v2 = p2 - p1
                    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-15)
                    cos_a = np.clip(cos_a, -1.0, 1.0)
                    angles_est[i] = np.arccos(cos_a)
                
                pp = np.concatenate([edges, angles_est]) + rng.normal(0, 0.05, N + N - 1)
                pp[:N] = np.abs(pp[:N]) + 0.1
                pp[N:] = np.abs(pp[N:]) + 0.1
                
                res_pp = minimize(
                    _objective_soft,
                    pp,
                    method='SLSQP',
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                cand_pts = _params_to_points_flat(res_pp.x).reshape(N, 2)
                cand_score = _score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_params = cand_pts.flatten().copy()
            except Exception:
                pass
    
    # Phase 4: Nelder-Mead polish on hard objective (direct)
    if _time_left() > 15:
        try:
            res_nm = minimize(
                _objective_hard_direct,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            cand_score = _score(res_nm.x.reshape(N, 2))
            if cand_score > best_score:
                best_score = cand_score
                best_params = res_nm.x.copy()
        except Exception:
            pass
    
    # Phase 5: Powell polish
    if _time_left() > 10:
        try:
            res_pow = minimize(
                _objective_hard_direct,
                best_params,
                method='Powell',
                options={'maxiter': 400, 'xtol': 1e-12, 'ftol': 1e-14, 'adaptive': True}
            )
            cand_score = _score(res_pow.x.reshape(N, 2))
            if cand_score > best_score:
                best_score = cand_score
                best_params = res_pow.x.copy()
        except Exception:
            pass
    
    # Normalize to unit hull area
    points = best_params.reshape(N, 2)
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(points)
        hull_area = hull.volume
    
    if hull_area > 1e-12:
        scale = np.sqrt(1.0 / hull_area)
        points = points * scale
    
    return points
