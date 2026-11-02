"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 13
DEADLINE = 145.0  # safety margin before 150s kill

# Pre-compute all C(13,3)=286 triple indices for fast vectorized scoring
_triple_i = []
_triple_j = []
_triple_k = []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _triple_i.append(_i)
            _triple_j.append(_j)
            _triple_k.append(_k)

_triple_i = np.array(_triple_i, dtype=np.int32)
_triple_j = np.array(_triple_j, dtype=np.int32)
_triple_k = np.array(_triple_k, dtype=np.int32)
_num_triples = len(_triple_i)


def _triangle_areas(pts):
    """Vectorized computation of all triangle areas."""
    pi = pts[_triple_i]
    pj = pts[_triple_j]
    pk = pts[_triple_k]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return 0.5 * np.abs(cross)


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score_points(pts):
    """Compute min triangle area / convex hull area for given points."""
    n = len(pts)
    if n < 3:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    
    areas = _triangle_areas(pts)
    min_area = np.min(areas)
    
    if min_area < 1e-15:
        return 0.0
    
    return min_area / hull_area


def _symmetric_params_to_points(params):
    """
    Convert symmetric parameters to 13 points.
    
    params structure (13 values):
    - params[0]: center x
    - params[1]: center y
    - params[2:14]: radial distances for 12 boundary points (in order)
    
    The 12 boundary points are placed at fixed angular positions:
    angles[i] = 2*pi*i/12 + global_rotation
    where global_rotation is also a parameter... 
    
    Actually, let me use:
    - params[0]: center x
    - params[1]: center y  
    - params[2]: global rotation angle
    - params[3:15]: 12 radial distances
    
    That's 15 params. Let me keep it simpler:
    - params[0]: center x
    - params[1]: center y
    - params[2:14]: 12 radial distances (angles fixed at 2*pi*i/12)
    
    Total: 14 parameters
    """
    cx = params[0]
    cy = params[1]
    radii = params[2:14]  # 12 radial values
    
    # Fixed angles for 12 boundary points
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    
    # Build points: center + 12 boundary
    center = np.array([cx, cy])
    boundary_x = cx + radii * np.cos(angles)
    boundary_y = cy + radii * np.sin(angles)
    boundary = np.column_stack([boundary_x, boundary_y])
    
    pts = np.vstack([center, boundary])
    return pts


def _symmetric_score(params):
    """Score for symmetric parameterization."""
    pts = _symmetric_params_to_points(params)
    return _score_points(pts)


def _neg_symmetric_score(params):
    """Negative score for minimization."""
    return -_symmetric_score(params)


def _optimize_symmetric(t_start):
    """
    Optimize in the reduced 14-dimensional symmetric space.
    Uses differential evolution for global search, then local refinement.
    """
    t0 = t_start
    
    # Bounds: center in [-1, 1], radii in [0.1, 2.0]
    bounds = [(-1.0, 1.0),  # center x
              (-1.0, 1.0),  # center y
              ] + [(0.1, 2.0)] * 12  # 12 radii
    
    # Phase 1: Differential evolution in symmetric space
    best_params = None
    best_score = 0.0
    
    if (time.time() - t0) < (DEADLINE - 30.0):
        try:
            result = differential_evolution(
                _neg_symmetric_score,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                init='sobol',
                updating='deferred',
                workers=1
            )
            
            if result.x is not None:
                best_params = result.x.copy()
                best_score = _symmetric_score(best_params)
        except Exception:
            pass
    
    # Phase 2: Local refinement with L-BFGS-B
    if best_params is not None and (time.time() - t0) < (DEADLINE - 15.0):
        try:
            res = minimize(
                _neg_symmetric_score,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12}
            )
            candidate = res.x
            score = _symmetric_score(candidate)
            if score > best_score:
                best_score = score
                best_params = candidate
        except Exception:
            pass
    
    # Phase 3: Powell refinement
    if best_params is not None and (time.time() - t0) < (DEADLINE - 10.0):
        try:
            res = minimize(
                _neg_symmetric_score,
                best_params,
                method='Powell',
                options={'maxiter': 100, 'maxfun': 3000, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            candidate = res.x
            score = _symmetric_score(candidate)
            if score > best_score:
                best_score = score
                best_params = candidate
        except Exception:
            pass
    
    return best_params, best_score


def _optimize_full_space(pts_init, t_start):
    """
    Optimize in full 26-dimensional space from a given initialization.
    Uses Powell on the exact objective.
    """
    t0 = t_start
    best_params = pts_init.flatten().copy()
    best_score = _score_points(pts_init)
    
    # Powell on exact objective
    if (time.time() - t0) < (DEADLINE - 15.0):
        try:
            res = minimize(
                lambda p: -_score_points(p.reshape(N, 2)),
                best_params,
                method='Powell',
                options={'maxiter': 150, 'maxfun': 8000, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            candidate = res.x.reshape(N, 2)
            score = _score_points(candidate)
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            pass
    
    return best_params, best_score


def _regular_polygon(n_points, radius=1.0, center=(0.0, 0.0)):
    """Generate vertices of a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pts = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])
    return pts


def _make_symmetric_init(r_outer=1.0, r_inner=0.5, n_outer=9, n_inner=4):
    """Create a symmetric initialization with two rings."""
    # Center point
    center = np.array([0.0, 0.0])
    
    # Outer ring
    outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    outer = np.column_stack([
        r_outer * np.cos(outer_angles),
        r_outer * np.sin(outer_angles)
    ])
    
    # Inner ring, rotated to interleave
    inner_angles = outer_angles + np.pi / n_inner
    inner = np.column_stack([
        r_inner * np.cos(inner_angles),
        r_inner * np.sin(inner_angles)
    ])
    
    pts = np.vstack([center, outer, inner])
    return pts[:N]


def construct():
    t_start = time.time()
    
    best_params = None
    best_score = 0.0
    
    # Strategy 1: Symmetric parameterization (reduced 14-dim space)
    if (time.time() - t_start) < (DEADLINE - 40.0):
        try:
            sym_params, sym_score = _optimize_symmetric(t_start)
            if sym_params is not None and sym_score > best_score:
                best_score = sym_score
                # Convert to full 26-dim representation
                best_params = _symmetric_params_to_points(sym_params).flatten()
        except Exception:
            pass
    
    # Strategy 2: Full-space optimization from diverse initializations
    inits = []
    
    # Regular 13-gon
    poly13 = _regular_polygon(N, radius=1.0)
    inits.append(poly13)
    
    # Two-layer symmetric: 9 outer + 4 inner
    two_layer = _make_symmetric_init(r_outer=1.0, r_inner=0.5, n_outer=9, n_inner=4)
    inits.append(two_layer)
    
    # Two-layer: 8 outer + 5 inner
    two_layer2 = _make_symmetric_init(r_outer=1.0, r_inner=0.45, n_outer=8, n_inner=5)
    inits.append(two_layer2)
    
    # Random initialization (seeded)
    rng = np.random.RandomState(123)
    rand_pts = rng.uniform(-1, 1, size=(N, 2))
    inits.append(rand_pts)
    
    # Optimize from each initialization
    for idx, init in enumerate(inits):
        if (time.time() - t_start) > (DEADLINE - 25.0):
            break
        try:
            params, score = _optimize_full_space(init, t_start)
            if score > best_score:
                best_score = score
                best_params = params
        except Exception:
            pass
    
    # Final refinement if time permits
    if best_params is not None and (time.time() - t_start) < (DEADLINE - 8.0):
        try:
            res_final = minimize(
                lambda p: -_score_points(p.reshape(N, 2)),
                best_params,
                method='Powell',
                options={'maxiter': 50, 'maxfun': 2000, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            score_final = _score_points(res_final.x.reshape(N, 2))
            if score_final > best_score:
                best_score = score_final
                best_params = res_final.x.copy()
        except Exception:
            pass
    
    # Safety: if no valid result, return regular polygon
    if best_params is None:
        best_params = inits[0].flatten()
    
    return best_params.reshape(N, 2)
