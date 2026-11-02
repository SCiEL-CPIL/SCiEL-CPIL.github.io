"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize, minimize_scalar

N = 13
DEADLINE = 140.0  # safety margin before 150s kill

# Pre-compute all triples for fast vectorized scoring
# There are C(13, 3) = 286 triples
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

def _score_points(pts):
    """Compute min triangle area / convex hull area for given points."""
    n = len(pts)
    if n < 3:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    
    # Compute convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    # Vectorized computation of all triangle areas
    pi = pts[_triple_i]
    pj = pts[_triple_j]
    pk = pts[_triple_k]
    
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area < 1e-15:
        return 0.0
    
    return min_area / hull_area

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas vectorized."""
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

def _logsumexp_surrogate(pts, beta=50.0):
    """
    Smooth surrogate: -logsumexp(-beta * areas) approximates min(areas) for large beta.
    This gives a differentiable objective that we can optimize with gradient methods.
    """
    areas = _triangle_areas(pts)
    if np.any(areas < 1e-15):
        return 0.0
    # -logsumexp(-beta * areas) / beta approximates min(areas)
    # Use stable logsumexp
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:  # overflow guard
        return areas[np.argmin(areas)]
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -log_sum / beta

def _neg_score(params):
    """Negative score for minimization. params is 1D array of length 2*N."""
    pts = params.reshape(N, 2)
    return -_score_points(pts)

def _neg_surrogate(params, beta=50.0):
    """Negative log-sum-exp surrogate for minimization."""
    pts = params.reshape(N, 2)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1.0
    surrogate_val = _logsumexp_surrogate(pts, beta)
    return -surrogate_val / hull_a

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    """Generate a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])

def _make_two_layer(inner_n, outer_n, r_inner=0.5, r_outer=1.0):
    """
    Create a two-layer configuration.
    inner_n points on a circle of radius r_inner.
    outer_n points on a circle of radius r_outer.
    Total points must be N.
    """
    if inner_n + outer_n != N:
        raise ValueError("inner_n + outer_n must equal N")
    
    pts = []
    if inner_n > 0:
        angles_in = np.linspace(0, 2 * np.pi, inner_n, endpoint=False)
        x_in = r_inner * np.cos(angles_in)
        y_in = r_inner * np.sin(angles_in)
        pts.append(np.column_stack([x_in, y_in]))
        
    if outer_n > 0:
        angles_out = np.linspace(0, 2 * np.pi, outer_n, endpoint=False)
        x_out = r_outer * np.cos(angles_out)
        y_out = r_outer * np.sin(angles_out)
        pts.append(np.column_stack([x_out, y_out]))
        
    return np.vstack(pts)

def _make_three_layer(center_n, mid_n, outer_n, r_mid=0.5, r_outer=1.0):
    """
    Create a three-layer configuration.
    center_n points at center (or small ring).
    mid_n points on circle radius r_mid.
    outer_n points on circle radius r_outer.
    """
    if center_n + mid_n + outer_n != N:
        raise ValueError("center_n + mid_n + outer_n must equal N")
        
    pts = []
    if center_n == 1:
        pts.append(np.array([[0.0, 0.0]]))
    elif center_n > 1:
        angles_c = np.linspace(0, 2 * np.pi, center_n, endpoint=False)
        r_c = 0.1
        x_c = r_c * np.cos(angles_c)
        y_c = r_c * np.sin(angles_c)
        pts.append(np.column_stack([x_c, y_c]))
        
    if mid_n > 0:
        angles_m = np.linspace(0, 2 * np.pi, mid_n, endpoint=False)
        x_m = r_mid * np.cos(angles_m)
        y_m = r_mid * np.sin(angles_m)
        pts.append(np.column_stack([x_m, y_m]))
        
    if outer_n > 0:
        angles_o = np.linspace(0, 2 * np.pi, outer_n, endpoint=False)
        x_o = r_outer * np.cos(angles_o)
        y_o = r_outer * np.sin(angles_o)
        pts.append(np.column_stack([x_o, y_o]))
        
    return np.vstack(pts)

def _ring_params_to_points(r_center, r_outer, angle_offsets):
    """
    Convert ring parameters to 13 points: 1 center + 12 outer.
    angle_offsets: 12 angles (radians) for outer ring points.
    Returns (13, 2) array.
    """
    pts = np.zeros((N, 2))
    # Center point at origin
    pts[0] = [0.0, 0.0]
    # Outer ring
    for i in range(12):
        angle = angle_offsets[i]
        pts[i+1, 0] = r_outer * np.cos(angle)
        pts[i+1, 1] = r_outer * np.sin(angle)
    return pts

def _optimize_ring_1c12o(t_start, deadline_margin=10.0):
    """
    Optimize the 1-center + 12-outer ring configuration.
    Parameters: r_outer (radius), 12 angle offsets.
    Total: 13 parameters.
    """
    t_start = time.time()
    best_score = -np.inf
    best_params = None
    
    # Initial guess: regular 12-gon on outer ring
    base_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    
    def neg_obj_ring(params):
        """Objective for ring optimization: params = [r_outer, angle_0, ..., angle_11]"""
        r_outer = params[0]
        angles = params[1:]
        pts = _ring_params_to_points(0.0, r_outer, angles)
        hull_a = _hull_area(pts)
        if hull_a <= 1e-12:
            return 1.0
        areas = _triangle_areas(pts)
        min_a = np.min(areas)
        if min_a < 1e-15:
            return 1.0
        return -min_a / hull_a
    
    def neg_obj_ring_surrogate(params, beta=30.0):
        """Surrogate version for smoother optimization."""
        r_outer = params[0]
        angles = params[1:]
        pts = _ring_params_to_points(0.0, r_outer, angles)
        hull_a = _hull_area(pts)
        if hull_a <= 1e-12:
            return 1.0
        areas = _triangle_areas(pts)
        if np.any(areas < 1e-15):
            return 1.0
        # Log-sum-exp surrogate for min
        scaled = -beta * areas
        max_s = np.max(scaled)
        if max_s > 700:
            min_a = np.min(areas)
            return -min_a / hull_a
        log_sum = max_s + np.log(np.sum(np.exp(scaled - max_s)))
        surrogate = -log_sum / beta
        return -surrogate / hull_a
    
    # Bounds: r_outer in [0.3, 1.5], angles in [0, 2*pi]
    bounds_ring = [(0.3, 1.5)] + [(0.0, 2 * np.pi)] * 12
    
    # Phase 1: Differential evolution on ring parameters
    try:
        de_result = differential_evolution(
            neg_obj_ring,
            bounds_ring,
            seed=42,
            maxiter=80,
            popsize=25,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.8,
            polish=True,
            init='sobol',
            updating='deferred',
            workers=1
        )
        
        if de_result.fun < -best_score:
            candidate_params = de_result.x
            r_out = candidate_params[0]
            angles = candidate_params[1:]
            pts = _ring_params_to_points(0.0, r_out, angles)
            score = _score_points(pts)
            if score > best_score:
                best_score = score
                best_params = candidate_params.copy()
    except Exception:
        pass
    
    # Phase 2: Local refinement with L-BFGS-B from multiple starts
    if best_params is None:
        best_params = np.array([1.0] + list(base_angles))
    
    starts = [best_params.copy()]
    
    # Add some perturbed starts
    rng = np.random.RandomState(123)
    for _ in range(8):
        pert = rng.normal(0, 0.1, size=13)
        pert[0] = rng.uniform(-0.2, 0.2)  # smaller perturbation for radius
        start = best_params + pert
        start[0] = np.clip(start[0], 0.3, 1.5)
        start[1:] = np.mod(start[1:], 2 * np.pi)
        starts.append(start)
    
    # Also try regular polygon start
    starts.append(np.array([1.0] + list(base_angles)))
    
    for start in starts:
        if time.time() - t_start > deadline_margin:
            break
        try:
            res = minimize(
                neg_obj_ring,
                start,
                method='L-BFGS-B',
                bounds=bounds_ring,
                options={'maxiter': 500, 'maxfun': 10000}
            )
            candidate_params = res.x
            r_out = candidate_params[0]
            angles = candidate_params[1:]
            pts = _ring_params_to_points(0.0, r_out, angles)
            score = _score_points(pts)
            if score > best_score:
                best_score = score
                best_params = candidate_params.copy()
        except Exception:
            pass
    
    # Phase 3: Surrogate-based refinement for smoother landscape
    if best_params is not None and time.time() - t_start < deadline_margin:
        for beta in [10.0, 20.0, 50.0]:
            if time.time() - t_start > deadline_margin:
                break
            try:
                res_surr = minimize(
                    lambda p: neg_obj_ring_surrogate(p, beta=beta),
                    best_params,
                    method='L-BFGS-B',
                    bounds=bounds_ring,
                    options={'maxiter': 300, 'maxfun': 5000}
                )
                candidate_params = res_surr.x
                r_out = candidate_params[0]
                angles = candidate_params[1:]
                pts = _ring_params_to_points(0.0, r_out, angles)
                score = _score_points(pts)
                if score > best_score:
                    best_score = score
                    best_params = candidate_params.copy()
            except Exception:
                pass
    
    # Convert back to full 26-dim parameter vector
    if best_params is not None:
        r_out = best_params[0]
        angles = best_params[1:]
        pts = _ring_params_to_points(0.0, r_out, angles)
        return pts, best_score
    
    return None, best_score

def construct():
    t_start = time.time()
    
    best_score = -np.inf
    best_pts = None
    
    # === Phase 1: Symmetric ring optimization (1 center + 12 outer) ===
    # This is the primary structural approach, exploiting likely optimal geometry
    ring_pts, ring_score = _optimize_ring_1c12o(t_start, deadline_margin=DEADLINE - 30.0)
    
    if ring_pts is not None:
        best_pts = ring_pts
        best_score = ring_score
    
    # === Phase 2: General optimization from ring solution and other inits ===
    bounds = [(-1.0, 1.0)] * (2 * N)
    rng = np.random.RandomState(42)
    
    # Build initializations
    inits = []
    
    # Regular 13-gon
    inits.append(_regular_polygon(N))
    
    # 1 center + 12 outer (regular)
    inits.append(_make_three_layer(1, 0, 12))
    
    # 1 center + 6 mid + 6 outer
    inits.append(_make_three_layer(1, 6, 6))
    
    # 7 mid + 6 outer
    inits.append(_make_two_layer(7, 6))
    
    # Random
    inits.append(rng.uniform(-1, 1, size=(N, 2)))
    
    # Add the ring solution if we have one
    if best_pts is not None:
        inits.append(best_pts.copy())
    
    # Evaluate all inits
    for p in inits:
        s = _score_points(p)
        if s > best_score:
            best_score = s
            best_pts = p.copy()
    
    # Time check
    def time_remaining():
        return (DEADLINE - 5.0) - (time.time() - t_start)
    
    # Phase 2a: Short DE on full 26-dim space from best inits
    if time_remaining() > 30.0:
        try:
            de_result = differential_evolution(
                _neg_score,
                bounds,
                seed=42,
                maxiter=40,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=True,
                init='sobol',
                updating='deferred',
                workers=1
            )
            
            if de_result.fun < -best_score:
                candidate = de_result.x.reshape(N, 2)
                score = _score_points(candidate)
                if score > best_score:
                    best_score = score
                    best_pts = candidate
        except Exception:
            pass
    
    # Phase 2b: Local refinement from best solutions
    starts_to_refine = [best_pts.flatten()]
    for p in inits[:3]:  # Only refine top few inits to save time
        if p.flatten().shape == best_pts.flatten().shape:
            starts_to_refine.append(p.flatten())
    
    # Deduplicate
    unique_starts = []
    for s in starts_to_refine:
        is_unique = True
        for u in unique_starts:
            if np.allclose(s, u, atol=1e-3):
                is_unique = False
                break
        if is_unique:
            unique_starts.append(s)
    
    for start in unique_starts:
        if time_remaining() < 15.0:
            break
        try:
            res = minimize(
                _neg_score,
                start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 10000}
            )
            score = _score_points(res.x.reshape(N, 2))
            if score > best_score:
                best_score = score
                best_pts = res.x.reshape(N, 2)
        except Exception:
            pass
    
    # Phase 2c: Surrogate-based refinement if time allows
    if time_remaining() > 20.0:
        for beta in [15.0, 30.0]:
            if time_remaining() < 10.0:
                break
            try:
                def neg_surr_full(params):
                    pts = params.reshape(N, 2)
                    hull_a = _hull_area(pts)
                    if hull_a <= 1e-12:
                        return 1.0
                    areas = _triangle_areas(pts)
                    if np.any(areas < 1e-15):
                        return 1.0
                    scaled = -beta * areas
                    max_s = np.max(scaled)
                    if max_s > 700:
                        min_a = np.min(areas)
                        return -min_a / hull_a
                    log_sum = max_s + np.log(np.sum(np.exp(scaled - max_s)))
                    surrogate = -log_sum / beta
                    return -surrogate / hull_a
                
                res_surr = minimize(
                    neg_surr_full,
                    best_pts.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'maxfun': 5000}
                )
                score = _score_points(res_surr.x.reshape(N, 2))
                if score > best_score:
                    best_score = score
                    best_pts = res_surr.x.reshape(N, 2)
            except Exception:
                pass
    
    # Phase 2d: Final local polish
    if time_remaining() > 5.0:
        try:
            res_final = minimize(
                _neg_score,
                best_pts.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'maxfun': 4000}
            )
            score = _score_points(res_final.x.reshape(N, 2))
            if score > best_score:
                best_score = score
                best_pts = res_final.x.reshape(N, 2)
        except Exception:
            pass
    
    # Safety: ensure we return a valid point set
    if best_pts is None:
        best_pts = _regular_polygon(N)
    
    return best_pts
