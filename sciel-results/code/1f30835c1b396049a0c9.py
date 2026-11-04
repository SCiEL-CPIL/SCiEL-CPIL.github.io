"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

# Precompute triangle indices
TRI_INDICES = list(combinations(range(N), 3))
NUM_TRIS = len(TRI_INDICES)  # 286

# Precompute index arrays for vectorized area computation
_I = np.array([t[0] for t in TRI_INDICES])
_J = np.array([t[1] for t in TRI_INDICES])
_K = np.array([t[2] for t in TRI_INDICES])

def _tri_areas(pts):
    """Vectorized triangle areas for all 286 triples."""
    pts = np.asarray(pts, dtype=np.float64)
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    return 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])

def _hull_area(pts):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return 0.0
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    min_area = np.min(_tri_areas(pts))
    return min_area / ha

def _soft_min_ratio(pts, temperature):
    """
    Smooth surrogate: soft-min of triangle areas / hull area.
    Uses log-sum-exp for numerical stability.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return -1e9
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return -1e9
    except Exception:
        return -1e9
    
    areas = _tri_areas(pts)
    areas = np.maximum(areas, 1e-15)
    
    # soft_min = -T * log(sum(exp(-a/T)))
    scaled = -areas / temperature
    scaled_max = np.max(scaled)
    log_sum = np.log(np.sum(np.exp(scaled - scaled_max))) + scaled_max
    soft_min_area = -log_sum * temperature
    
    return soft_min_area / hull_area

def _regular_polygon(n, radius=1.0, center=(0,0)):
    """Generate regular polygon vertices."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius*np.cos(angles), 
                            center[1] + radius*np.sin(angles)])

def _halton_sequence(n, base):
    """Generate n Halton sequence points in [0,1)."""
    result = []
    f = 1.0
    for i in range(1, n + 1):
        x = 0.0
        i_val = i
        while i_val > 0:
            i_val, digit = divmod(i_val, base)
            x += digit * f
            f /= base
        result.append(x)
    return np.array(result)

def _generate_halton_points(n, base1=2, base2=3):
    """Generate n points using 2D Halton sequence in unit square, mapped to disk."""
    x = _halton_sequence(n, base1)
    y = _halton_sequence(n, base2)
    r = np.sqrt(x)
    theta = 2 * np.pi * y
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return pts

def _concentric_rings(n, n_rings=3):
    """Generate points on concentric rings for diverse initialization."""
    pts = []
    pts.append([0.0, 0.0])
    remaining = n - 1
    for ring in range(n_rings):
        if remaining <= 0:
            break
        n_on_ring = min(remaining, max(3, (n-1)//n_rings))
        if ring == n_rings - 1:
            n_on_ring = remaining
        radius = (ring + 1) / n_rings
        angles = np.linspace(0, 2*np.pi, n_on_ring, endpoint=False) + ring * np.pi/6
        ring_pts = np.column_stack([radius*np.cos(angles), radius*np.sin(angles)])
        pts.append(ring_pts)
        remaining -= n_on_ring
    if remaining > 0:
        angles = np.linspace(0, 2*np.pi, remaining, endpoint=False)
        ring_pts = np.column_stack([1.0*np.cos(angles), 1.0*np.sin(angles)])
        pts.append(ring_pts)
    result = np.vstack(pts)
    return result[:n]

def _fixed_frame_transform(pts):
    """
    Transform a full point set into the reduced parameter space by fixing:
    - Point 0 at origin
    - Point 1 on positive x-axis at distance 1
    - Point 2 with y > 0 (breaks reflection symmetry)
    
    Returns the 20-dim parameter vector for points 2..12.
    """
    pts = np.asarray(pts, dtype=np.float64)
    # Translate: p0 -> origin
    pts = pts - pts[0]
    # Rotate: p1 -> positive x-axis
    p1 = pts[1]
    angle = np.arctan2(p1[1], p1[0])
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s], [s, c]])
    pts = (R @ pts.T).T
    # Scale: |p1| = 1
    scale = np.linalg.norm(pts[1])
    if scale < 1e-12:
        return None
    pts = pts / scale
    # Ensure p2 has y >= 0 (flip if needed)
    if pts[2, 1] < 0:
        pts[1:, 1] = -pts[1:, 1]
    # Return parameters for points 2 through 12 (20 values)
    return pts[2:].flatten()

def _reconstruct_from_params(params, p0=(0.0, 0.0), p1=(1.0, 0.0)):
    """
    Reconstruct full 13-point set from 20-dim parameter vector.
    p0 and p1 are fixed by the normalization.
    """
    params = np.asarray(params, dtype=np.float64)
    pts = np.zeros((N, 2))
    pts[0] = p0
    pts[1] = p1
    pts[2:] = params.reshape(N - 2, 2)
    return pts

def _objective_reduced(x, temperature):
    """
    Objective function in reduced parameter space.
    x is 20-dim vector for points 2..12.
    """
    pts = _reconstruct_from_params(x)
    val = _soft_min_ratio(pts, temperature)
    if val < -1e8:
        return 1e8
    return -val

def _optimize_reduced(x0, time_budget):
    """
    Optimize in the reduced 20-dim parameter space using multi-stage L-BFGS.
    """
    start_time = time.time()
    best_params = x0.copy()
    best_score = _score(_reconstruct_from_params(x0))
    
    # Multi-stage temperature annealing
    temperatures = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for temp in temperatures:
        if time.time() - start_time > time_budget - 3:
            break
        
        try:
            result = minimize(
                _objective_reduced, best_params, args=(temp,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9, 'maxls': 50}
            )
            cand_params = result.x
            cand_pts = _reconstruct_from_params(cand_params)
            cand_score = _score(cand_pts)
            if cand_score > best_score:
                best_score = cand_score
                best_params = cand_params.copy()
        except Exception:
            pass
    
    # Nelder-Mead polish on exact objective
    if time.time() - start_time < time_budget - 2:
        def exact_obj(x):
            pts = _reconstruct_from_params(x)
            return -_score(pts)
        
        try:
            result = minimize(
                exact_obj, best_params, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            cand_params = result.x
            cand_pts = _reconstruct_from_params(cand_params)
            cand_score = _score(cand_pts)
            if cand_score > best_score:
                best_score = cand_score
                best_params = cand_params.copy()
        except Exception:
            pass
    
    # Final tight L-BFGS
    if time.time() - start_time < time_budget - 1:
        try:
            result = minimize(
                _objective_reduced, best_params, args=(0.0001,),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            cand_params = result.x
            cand_pts = _reconstruct_from_params(cand_params)
            cand_score = _score(cand_pts)
            if cand_score > best_score:
                best_score = cand_score
                best_params = cand_params.copy()
        except Exception:
            pass
    
    return _reconstruct_from_params(best_params), best_score

def _generate_reduced_starts(rng):
    """Generate diverse starting points in reduced parameter space."""
    starts = []
    
    # 1. Regular 13-gon transformed
    poly = _regular_polygon(N, radius=1.0)
    params = _fixed_frame_transform(poly)
    if params is not None:
        starts.append(params)
    
    # 2-4: Halton sequences
    for b1, b2 in [(2, 3), (3, 5), (5, 7)]:
        halton = _generate_halton_points(N, base1=b1, base2=b2)
        params = _fixed_frame_transform(halton)
        if params is not None:
            starts.append(params)
    
    # 5. Concentric rings
    rings = _concentric_rings(N, n_rings=3)
    params = _fixed_frame_transform(rings)
    if params is not None:
        starts.append(params)
    
    # 6-8: Jittered regular polygon
    for jitter_std in [0.05, 0.1, 0.2]:
        poly = _regular_polygon(N, radius=1.0)
        jitter = rng.normal(0, jitter_std, (N, 2))
        pts = poly + jitter
        params = _fixed_frame_transform(pts)
        if params is not None:
            starts.append(params)
    
    # 9-11: Random points in disk
    for _ in range(3):
        r = np.sqrt(rng.uniform(0.1, 1.0, N))
        theta = rng.uniform(0, 2*np.pi, N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        params = _fixed_frame_transform(pts)
        if params is not None:
            starts.append(params)
    
    # 12-14: Jittered Halton
    for jitter_std in [0.05, 0.1, 0.15]:
        halton = _generate_halton_points(N, base1=2, base2=3)
        jitter = rng.normal(0, jitter_std, (N, 2))
        pts = halton + jitter
        params = _fixed_frame_transform(pts)
        if params is not None:
            starts.append(params)
    
    return starts

def construct():
    """Main entry point: find optimal 13-point configuration."""
    start_time = time.time()
    total_budget = 145.0  # Safety margin for 150s limit
    
    rng = np.random.RandomState(12345)
    
    best_points = None
    best_ratio = -1e9
    
    # Generate diverse starting points in reduced space
    starts = _generate_reduced_starts(rng)
    
    if not starts:
        # Fallback
        best_points = _regular_polygon(N, radius=1.0)
        return best_points
    
    # Evaluate and sort by initial score
    scored_starts = []
    for params in starts:
        try:
            pts = _reconstruct_from_params(params)
            s = _score(pts)
            scored_starts.append((s, params))
        except Exception:
            scored_starts.append((0.0, params))
    scored_starts.sort(key=lambda x: -x[0])
    
    # Optimize from each starting point
    n_starts = len(scored_starts)
    base_time = (total_budget - 5.0) / n_starts
    
    for i, (init_score, params) in enumerate(scored_starts):
        if time.time() - start_time > total_budget - 2:
            break
        
        remaining_time = total_budget - (time.time() - start_time)
        # Allocate more time to better starting points
        quality_factor = 1.0 + 0.5 * (init_score / max(best_ratio, 1e-9)) if best_ratio > 0 else 1.0
        alloc_time = min(base_time * quality_factor, remaining_time - 1.0)
        if alloc_time < 3.0:
            continue
        
        try:
            opt_pts, opt_score = _optimize_reduced(params, alloc_time)
            if opt_score > best_ratio:
                best_ratio = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Final refinement on the best found
    if best_points is not None and time.time() - start_time < total_budget - 3:
        final_time = total_budget - (time.time() - start_time) - 1.0
        if final_time > 10.0:
            try:
                final_params = _fixed_frame_transform(best_points)
                if final_params is not None:
                    final_pts, final_score = _optimize_reduced(final_params, final_time)
                    if final_score > best_ratio:
                        best_ratio = final_score
                        best_points = final_pts.copy()
            except Exception:
                pass
    
    if best_points is None:
        best_points = _regular_polygon(N, radius=1.0)
    
    return best_points
# EVOLVE-BLOCK-END
