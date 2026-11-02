"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

# Deadline guard
DEADLINE = 145.0
START_TIME = time.time()

def _time_left():
    return DEADLINE - (time.time() - START_TIME) - 5.0

# Pre-compute all triple indices for vectorized scoring
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

def _min_triangle_area_vec(pts):
    """Vectorized computation of minimum triangle area."""
    pi = pts[_triple_i]
    pj = pts[_triple_j]
    pk = pts[_triple_k]
    ax = pj[:, 0] - pi[:, 0]
    ay = pj[:, 1] - pi[:, 1]
    bx = pk[:, 0] - pi[:, 0]
    by = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(ax * by - ay * bx)
    return np.min(areas)

def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    min_area = _min_triangle_area_vec(pts)
    return min_area / hull_area

def _neg_score(params):
    """Objective: minimize negative score."""
    if _time_left() < 3.0:
        return 0.0
    pts = np.asarray(params).reshape((N, 2))
    if not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -5) or np.any(pts > 5):
        return 0.0
    score = _score_points(pts)
    return -score

def _regular_polygon(n, radius, center=(0, 0), phase=0):
    angles = 2 * np.pi * np.arange(n) / n + phase
    return np.column_stack([center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)])

def _make_two_layer(outer_n, outer_r, inner_n, inner_r, offset=0, center_r=0):
    """Create a configuration with points on two concentric circles."""
    outer_pts = _regular_polygon(outer_n, outer_r, phase=np.pi/2)
    inner_angles = 2 * np.pi * np.arange(inner_n) / inner_n + offset
    inner_pts = np.column_stack([inner_r * np.cos(inner_angles),
                                 inner_r * np.sin(inner_angles)])
    pts = np.vstack([outer_pts, inner_pts])
    return pts

def _make_three_layer(outer_n, outer_r, mid_n, mid_r, inner_n, inner_r, 
                      offset1=0, offset2=0):
    """Create a configuration with three concentric circles."""
    outer_pts = _regular_polygon(outer_n, outer_r, phase=np.pi/2)
    mid_angles = 2 * np.pi * np.arange(mid_n) / mid_n + offset1
    mid_pts = np.column_stack([mid_r * np.cos(mid_angles),
                               mid_r * np.sin(mid_angles)])
    inner_angles = 2 * np.pi * np.arange(inner_n) / inner_n + offset2
    inner_pts = np.column_stack([inner_r * np.cos(inner_angles),
                                 inner_r * np.sin(inner_angles)])
    return np.vstack([outer_pts, mid_pts, inner_pts])

def _center_plus_ring(n_ring, ring_r, phase=0):
    """One center point + n_ring points on a ring."""
    center = np.array([[0.0, 0.0]])
    ring_pts = _regular_polygon(n_ring, ring_r, phase=phase)
    return np.vstack([center, ring_pts])

def _center_plus_two_rings(n1, r1, n2, r2, phase1=0, phase2=0):
    """One center + two rings."""
    center = np.array([[0.0, 0.0]])
    ring1 = _regular_polygon(n1, r1, phase=phase1)
    ring2 = _regular_polygon(n2, r2, phase=phase2)
    return np.vstack([center, ring1, ring2])

def construct():
    """Construct 13 points to maximize the minimum triangle area over convex hull area."""
    global START_TIME
    START_TIME = time.time()
    
    best_score = 0.0
    best_params = None
    
    # Generate diverse symmetric initializations
    inits = []
    
    # Type A: 1 center + 12 on ring
    for r in [0.8, 1.0, 1.2, 1.4, 1.6]:
        for phase in [0, np.pi/12, np.pi/6]:
            pts = _center_plus_ring(12, r, phase)
            inits.append(pts)
    
    # Type B: 1 center + 6 inner + 6 outer
    for r1 in [0.3, 0.5, 0.7]:
        for r2 in [1.0, 1.2, 1.4]:
            for off in [0, np.pi/6, np.pi/3]:
                pts = _center_plus_two_rings(6, r1, 6, r2, phase1=off, phase2=0)
                inits.append(pts)
    
    # Type C: 7 outer + 6 inner (no center)
    for r_out in [1.0, 1.2, 1.4]:
        for r_in in [0.3, 0.5, 0.7]:
            for off in [0, np.pi/7, np.pi/14, np.pi/6]:
                pts = _make_two_layer(7, r_out, 6, r_in, offset=off)
                inits.append(pts)
    
    # Type D: 6 outer + 6 mid + 1 center
    for r_out in [1.0, 1.2]:
        for r_mid in [0.5, 0.7]:
            for off in [0, np.pi/6]:
                pts = _center_plus_two_rings(6, r_mid, 6, r_out, phase1=off, phase2=0)
                inits.append(pts)
    
    # Type E: 5 outer + 4 mid + 4 inner
    for r_out in [1.0, 1.3]:
        for r_mid in [0.4, 0.6]:
            for r_in in [0.2, 0.35]:
                pts = _make_three_layer(5, r_out, 4, r_mid, 4, r_in, 
                                        offset1=0, offset2=np.pi/4)
                inits.append(pts)
    
    # Score all initializations
    for pts in inits:
        if _time_left() < 5.0:
            break
        score = _score_points(pts)
        if score > best_score:
            best_score = score
            best_params = pts.flatten()
    
    # Local refinement from top initializations using Nelder-Mead
    # Sort initializations by score and refine the best ones
    scored_inits = []
    for pts in inits:
        s = _score_points(pts)
        scored_inits.append((s, pts))
    scored_inits.sort(key=lambda x: -x[0])
    
    # Refine top candidates
    num_refine = min(len(scored_inits), 15)
    for idx in range(num_refine):
        if _time_left() < 8.0:
            break
        s_init, pts_init = scored_inits[idx]
        x0 = pts_init.flatten()
        try:
            result = minimize(
                _neg_score,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11,
                         'adaptive': True}
            )
            if result.fun < 0:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_params = result.x
        except Exception:
            pass
    
    # If time remains, try a few more random perturbations around best
    if best_params is not None and _time_left() > 15.0:
        rng = np.random.RandomState(42)
        for trial in range(20):
            if _time_left() < 5.0:
                break
            noise = rng.normal(0, 0.05, (N, 2))
            x0 = best_params + noise.flatten()
            try:
                result = minimize(
                    _neg_score,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11,
                             'adaptive': True}
                )
                if result.fun < 0:
                    score = -result.fun
                    if score > best_score:
                        best_score = score
                        best_params = result.x
            except Exception:
                pass
    
    # Return the best configuration
    if best_params is not None:
        pts = np.array(best_params).reshape((N, 2))
        # Normalize to unit area convex hull
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area > 1e-12:
                scale = 1.0 / np.sqrt(hull_area)
                pts = pts * scale
        except Exception:
            pass
        return pts
    else:
        # Fallback: center + 12-gon
        return _center_plus_ring(12, 1.0)

# EVOLVE-BLOCK-END
