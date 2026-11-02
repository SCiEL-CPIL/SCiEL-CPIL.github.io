"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
from itertools import combinations

DEADLINE = 145.0  # Safety margin before 150s kill
START_TIME = time.time()


def _time_left():
    return DEADLINE - (time.time() - START_TIME)


def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 0:
            return 0.0
    except Exception:
        return 0.0
    
    # Vectorized triangle area computation
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
                    if min_area < 1e-15:
                        return 0.0
    
    return min_area / hull_area


def _neg_score(pts):
    return -_score_points(pts)


def _param_to_points(params):
    """Convert symmetric parameters to point coordinates.
    
    params: array of shape (N,)
    params[0]: radius scale (0.5 to 1.5)
    params[1:]: angular offsets for each point
    
    Points are placed on a circle with slight radial perturbation.
    """
    n = len(params)
    scale = params[0]
    angles = params[1:]
    
    # Base angles: evenly spaced
    base_angles = np.arange(n) * 2 * np.pi / n
    # Add offsets
    total_offset = np.sum(angles)
    angles = base_angles + angles - total_offset / n  # Keep mean zero
    
    # Radial: slight perturbation to break perfect symmetry
    radii = scale * (1.0 + 0.1 * np.sin(angles * 2))
    
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts


def _param_to_points_v2(params):
    """Alternative parameterization: two concentric rings.
    
    params[0]: inner ring radius
    params[1]: outer ring radius
    params[2:]: angular offsets
    """
    n = len(params)
    r_inner = params[0]
    r_outer = params[1]
    angles = params[2:]
    
    # 6 points on inner ring, 7 on outer
    n_inner = 6
    n_outer = 7
    
    base_angles_inner = np.arange(n_inner) * 2 * np.pi / n_inner
    base_angles_outer = np.arange(n_outer) * 2 * np.pi / n_outer
    
    # Apply offsets
    total_offset = np.sum(angles)
    angles = angles - total_offset / n
    
    angles_inner = base_angles_inner + angles[:n_inner]
    angles_outer = base_angles_outer + angles[n_inner:]
    
    pts_inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    pts_outer = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
    
    pts = np.vstack([pts_inner, pts_outer])
    return pts


def _neg_score_v2(params):
    pts = _param_to_points_v2(params)
    return _neg_score(pts)


def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_score = 0.0
    best_pts = None
    
    # Strategy 1: Coarse grid over two-ring parameterization
    # Then Nelder-Mead refinement
    
    n_inner = 6
    n_outer = 7
    
    # Coarse grid: r_inner in [0.3, 0.6], r_outer in [0.8, 1.2]
    # Angular offsets: small perturbations
    
    r_inner_vals = np.linspace(0.3, 0.6, 5)
    r_outer_vals = np.linspace(0.8, 1.2, 5)
    
    # Small angular perturbations
    angle_pert_vals = np.linspace(-0.2, 0.2, 3)
    
    candidates = []
    
    for ri in r_inner_vals:
        for ro in r_outer_vals:
            for ap in angle_pert_vals:
                if _time_left() < 10:
                    break
                # Create parameter vector
                params = np.array([ri, ro])
                # Create 13 angular offsets: small perturbations
                angles = np.zeros(13)
                # Alternate perturbations
                for i in range(13):
                    angles[i] = ap * (1 if i % 2 == 0 else -1)
                params = np.concatenate([params, angles])
                candidates.append(params)
    
    # Also add some random candidates
    rng = np.random.RandomState(42)
    for _ in range(30):
        if _time_left() < 10:
            break
        ri = rng.uniform(0.3, 0.6)
        ro = rng.uniform(0.8, 1.2)
        angles = rng.uniform(-0.3, 0.3, 13)
        params = np.concatenate([[ri, ro], angles])
        candidates.append(params)
    
    # Score all candidates and keep top ones for refinement
    scored = []
    for params in candidates:
        if _time_left() < 15:
            break
        score = _score_points(_param_to_points_v2(params))
        scored.append((score, params))
    
    scored.sort(key=lambda x: -x[0])
    
    # Nelder-Mead refinement on top candidates
    top_k = min(8, len(scored))
    
    for i in range(top_k):
        if _time_left() < 20:
            break
        
        score, params = scored[i]
        
        # Nelder-Mead optimization
        result = minimize(
            _neg_score_v2,
            params,
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'maxfev': 1000,
                'xatol': 1e-6,
                'fatol': 1e-8
            }
        )
        
        if result.success or _time_left() < 5:
            refined_score = -result.fun
            if refined_score > best_score:
                best_score = refined_score
                best_pts = _param_to_points_v2(result.x)
    
    # Strategy 2: Single ring with radial perturbation
    if _time_left() > 30:
        # Coarse grid for single ring
        scale_vals = np.linspace(0.8, 1.2, 4)
        pert_vals = np.linspace(0.0, 0.2, 3)
        
        candidates2 = []
        for s in scale_vals:
            for p in pert_vals:
                if _time_left() < 15:
                    break
                params = np.array([s])
                angles = np.zeros(13)
                for i in range(13):
                    angles[i] = p * np.sin(i * 2 * np.pi / 13)
                params = np.concatenate([params, angles])
                candidates2.append(params)
        
        scored2 = []
        for params in candidates2:
            if _time_left() < 15:
                break
            score = _score_points(_param_to_points(params))
            scored2.append((score, params))
        
        scored2.sort(key=lambda x: -x[0])
        
        top_k2 = min(5, len(scored2))
        for i in range(top_k2):
            if _time_left() < 20:
                break
            
            score, params = scored2[i]
            
            result = minimize(
                _neg_score,
                params,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'maxfev': 1000,
                    'xatol': 1e-6,
                    'fatol': 1e-8
                }
            )
            
            if result.success or _time_left() < 5:
                refined_score = -result.fun
                if refined_score > best_score:
                    best_score = refined_score
                    best_pts = _param_to_points(result.x)
    
    # Strategy 3: Direct Nelder-Mead on full coordinate space from good starting points
    if _time_left() > 40 and best_pts is not None:
        # Start from best found and optimize directly in 2D coordinates
        x0 = best_pts.flatten()
        
        def _neg_score_coords(x):
            pts = x.reshape(-1, 2)
            return _neg_score(pts)
        
        result = minimize(
            _neg_score_coords,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 300,
                'maxfev': 800,
                'xatol': 1e-6,
                'fatol': 1e-8
            }
        )
        
        if result.success or _time_left() < 5:
            refined_score = -result.fun
            if refined_score > best_score:
                best_score = refined_score
                best_pts = result.x.reshape(-1, 2)
    
    # If we still have time, try a few random restarts with Nelder-Mead
    if _time_left() > 30:
        rng = np.random.RandomState(123)
        for _ in range(5):
            if _time_left() < 25:
                break
            
            # Random starting point: points in unit disk
            pts = rng.uniform(-1, 1, (13, 2))
            # Scale to unit area hull
            try:
                hull = ConvexHull(pts)
                area = hull.volume
                if area > 0:
                    pts = pts / np.sqrt(area)
            except:
                continue
            
            x0 = pts.flatten()
            
            result = minimize(
                _neg_score_coords,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'maxfev': 500,
                    'xatol': 1e-5,
                    'fatol': 1e-7
                }
            )
            
            if result.success or _time_left() < 5:
                refined_score = -result.fun
                if refined_score > best_score:
                    best_score = refined_score
                    best_pts = result.x.reshape(-1, 2)
    
    # Final check: ensure we have valid points
    if best_pts is None:
        # Fallback: regular 13-gon
        angles = np.arange(13) * 2 * np.pi / 13
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Final polish: one more Nelder-Mead step
    if _time_left() > 10:
        x0 = best_pts.flatten()
        result = minimize(
            _neg_score_coords,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 100,
                'maxfev': 300,
                'xatol': 1e-7,
                'fatol': 1e-9
            }
        )
        
        if result.success:
            refined_score = -result.fun
            if refined_score > best_score:
                best_score = refined_score
                best_pts = result.x.reshape(-1, 2)
    
    return best_pts
# EVOLVE-BLOCK-END

