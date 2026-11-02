"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import time

N = 13

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples, vectorized."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            if len(cross) > 0:
                local_min = 0.5 * np.min(cross)
                if local_min < min_area:
                    min_area = local_min
    return min_area if min_area != float('inf') else 0.0

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    return _min_triangle_area(pts) / hull_a

def _neg_score(params):
    pts = params.reshape(-1, 2)
    return -_score(pts)

def _all_triangle_areas(pts):
    """Compute all triangle areas as a flat array."""
    n = len(pts)
    if n < 3:
        return np.array([0.0])
    all_areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            if len(cross) > 0:
                all_areas.append(0.5 * cross)
    if not all_areas:
        return np.array([0.0])
    return np.concatenate(all_areas)

def _soft_min_triangle_area(pts, alpha=50.0):
    """Smooth approximation of the minimum triangle area via log-sum-exp."""
    n = len(pts)
    if n < 3:
        return 0.0
    areas = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    min_a = np.min(areas)
    shifted = -alpha * (areas - min_a)
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return min_a + log_sum / alpha

def _smooth_score(pts, alpha=50.0):
    """Smooth score surrogate."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    smooth_min = _soft_min_triangle_area(pts, alpha)
    return smooth_min / hull_a

def _neg_smooth_score(params, alpha=50.0):
    pts = params.reshape(-1, 2)
    s = _smooth_score(pts, alpha)
    if not np.isfinite(s) or s < 0:
        return 1e10
    return -s

def _make_ring12_params_to_points(params):
    """
    Convert 13-dim params to 13 points: center + 12 ring points.
    params = [cx, cy, r0, r1, ..., r11, a0, a1, ..., a11]
    Actually 2 + 12 + 12 = 26... that's too many.
    
    Let's use: center (2) + 12 radii (12) = 14 params, with fixed angles.
    Or: center (2) + 12 radii (12) + 12 angle offsets (12) = 26.
    
    For a reduced space, let's use:
    - Center: (cx, cy)
    - 12 radial distances: r0..r11
    - Angles fixed at k*2pi/12 + global_rotation
    Total: 2 + 12 + 1 = 15 params
    
    Or even simpler: assume center at origin, 12 radii, 1 global rotation = 13 params.
    """
    cx, cy = params[0], params[1]
    radii = params[2:14]
    rotation = params[14]
    
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False) + rotation
    ring_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    center = np.array([[cx, cy]])
    
    return np.vstack([center, ring_pts])

def _make_ring12_v2(params):
    """
    13 points: center + 12 ring with per-point angle jitter.
    params = [cx, cy, r0..r11, dtheta0..dtheta11]
    Total: 2 + 12 + 12 = 26 params
    """
    cx, cy = params[0], params[1]
    radii = params[2:14]
    dthetas = params[14:26]
    
    base_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    angles = base_angles + dthetas
    
    ring_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    center = np.array([[cx, cy]])
    
    return np.vstack([center, ring_pts])

def _neg_ring12_score(params):
    pts = _make_ring12_params_to_points(params)
    return -_score(pts)

def _neg_ring12_smooth_score(params, alpha=50.0):
    pts = _make_ring12_params_to_points(params)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1e10
    smooth_min = _soft_min_triangle_area(pts, alpha)
    val = smooth_min / hull_a
    if not np.isfinite(val) or val < 0:
        return 1e10
    return -val

def _neg_ring12_v2_score(params):
    pts = _make_ring12_v2(params)
    return -_score(pts)

def _neg_ring12_v2_smooth_score(params, alpha=50.0):
    pts = _make_ring12_v2(params)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1e10
    smooth_min = _soft_min_triangle_area(pts, alpha)
    val = smooth_min / hull_a
    if not np.isfinite(val) or val < 0:
        return 1e10
    return -val

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_score = 0.0
    best_pts = np.zeros((N, 2))
    rng = np.random.RandomState(42)
    
    # Initial guess: regular 12-gon + center
    angles12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    outer = np.column_stack([np.cos(angles12), np.sin(angles12)])
    center = np.array([[0.0, 0.0]])
    init_pts = np.vstack([center, outer])
    best_score = _score(init_pts)
    best_pts = init_pts.copy()
    
    # Strategy 1: Differential evolution on the 15-param ring12 space
    # params = [cx, cy, r0..r11, rotation]
    bounds15 = [(-0.5, 0.5), (-0.5, 0.5)] + [(0.3, 1.5)] * 12 + [(0, 2 * np.pi)]
    
    if time.time() < deadline - 40.0:
        try:
            def _de_objective(params):
                return _neg_ring12_score(params)
            
            de_result = differential_evolution(
                _de_objective,
                bounds=bounds15,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                polish=False,
                init='sobol',
                updating='immediate'
            )
            
            de_params = de_result.x
            de_pts = _make_ring12_params_to_points(de_params)
            de_score = _score(de_pts)
            
            if de_score > best_score:
                best_score = de_score
                best_pts = de_pts.copy()
        except Exception:
            pass
    
    # Strategy 2: Local refinement from DE result using Powell
    if time.time() < deadline - 30.0:
        try:
            # Get current best params in ring12 space
            # Extract params from best_pts
            cx, cy = best_pts[0]
            ring = best_pts[1:]
            radii = np.linalg.norm(ring, axis=1)
            angles = np.arctan2(ring[:, 1], ring[:, 0])
            base_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
            # Find rotation that best aligns
            from scipy.optimize import minimize_scalar
            def rot_err(rot):
                aligned = np.mod(angles - rot, 2 * np.pi)
                # Match to nearest base angle
                min_diff = np.min(np.abs(aligned[:, None] - base_angles[None, :]), axis=1)
                return np.sum(min_diff ** 2)
            
            rot_res = minimize_scalar(rot_err, bounds=(0, 2 * np.pi), method='bounded')
            rotation = rot_res.x
            
            current_params = np.concatenate([[cx, cy], radii, [rotation]])
            
            # Powell refinement
            res_powell = minimize(
                _neg_ring12_score,
                current_params,
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            powell_params = res_powell.x
            powell_pts = _make_ring12_params_to_points(powell_params)
            powell_score = _score(powell_pts)
            
            if powell_score > best_score:
                best_score = powell_score
                best_pts = powell_pts.copy()
        except Exception:
            pass
    
    # Strategy 3: Full 26-dim optimization with smooth surrogate + Nelder-Mead
    if time.time() < deadline - 25.0:
        try:
            # Use the best points as starting point for full optimization
            for alpha in [20.0, 50.0, 100.0, 200.0]:
                if time.time() > deadline - 20.0:
                    break
                res_smooth = minimize(
                    lambda p, a=alpha: _neg_smooth_score(p, a),
                    best_pts.flatten(),
                    method='Powell',
                    options={'maxiter': 150, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                smooth_pts = res_smooth.x.reshape(-1, 2)
                smooth_score = _score(smooth_pts)
                if smooth_score > best_score:
                    best_score = smooth_score
                    best_pts = smooth_pts.copy()
        except Exception:
            pass
    
    # Strategy 4: Nelder-Mead exact refinement
    if time.time() < deadline - 15.0:
        try:
            res_nm = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            nm_pts = res_nm.x.reshape(-1, 2)
            nm_score = _score(nm_pts)
            if nm_score > best_score:
                best_score = nm_score
                best_pts = nm_pts.copy()
        except Exception:
            pass
    
    # Strategy 5: Try ring12_v2 (26-param) with DE
    if time.time() < deadline - 20.0:
        try:
            bounds26 = [(-0.5, 0.5), (-0.5, 0.5)] + [(0.3, 1.5)] * 12 + [(-0.3, 0.3)] * 12
            
            def _de_obj26(params):
                return _neg_ring12_v2_score(params)
            
            de_result26 = differential_evolution(
                _de_obj26,
                bounds=bounds26,
                seed=123,
                maxiter=50,
                popsize=12,
                tol=1e-6,
                polish=False,
                init='sobol'
            )
            
            de_params26 = de_result26.x
            de_pts26 = _make_ring12_v2(de_params26)
            de_score26 = _score(de_pts26)
            
            if de_score26 > best_score:
                best_score = de_score26
                best_pts = de_pts26.copy()
        except Exception:
            pass
    
    # Strategy 6: Multi-start local optimization
    if time.time() < deadline - 10.0:
        # Generate diverse starts
        starts = [best_pts.copy()]
        
        # Perturbed versions
        for _ in range(5):
            perturbed = best_pts + 0.05 * rng.randn(N, 2)
            starts.append(perturbed)
        
        # Different ring radii patterns
        for r_pattern in [0.7, 0.8, 0.9]:
            angles12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
            ring = np.column_stack([r_pattern * np.cos(angles12), r_pattern * np.sin(angles12)])
            center = np.array([[0.0, 0.0]])
            starts.append(np.vstack([center, ring]))
        
        for s_pts in starts:
            if time.time() > deadline - 5.0:
                break
            try:
                res = minimize(
                    _neg_score,
                    s_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-13}
                )
                res_pts = res.x.reshape(-1, 2)
                res_score = _score(res_pts)
                if res_score > best_score:
                    best_score = res_score
                    best_pts = res_pts.copy()
            except Exception:
                pass
    
    # Final polish
    if time.time() < deadline - 3.0:
        try:
            res_final = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-13, 'fatol': 1e-15}
            )
            final_pts = res_final.x.reshape(-1, 2)
            final_score = _score(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_pts = final_pts.copy()
        except Exception:
            pass
    
    return best_pts
