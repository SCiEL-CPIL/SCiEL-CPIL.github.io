"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

# Precompute triangle indices and index arrays for vectorized area computation
TRI_INDICES = list(combinations(range(N), 3))
NUM_TRIS = len(TRI_INDICES)  # 286

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
    # Map from [0,1]^2 to unit disk using polar coordinates
    r = np.sqrt(x)
    theta = 2 * np.pi * y
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return pts

def _regular_polygon(n, radius=1.0, center=(0,0)):
    """Generate regular polygon vertices."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius*np.cos(angles), 
                            center[1] + radius*np.sin(angles)])

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

def _structured_grid(n):
    """Generate points on a structured grid pattern."""
    xs = np.linspace(-0.9, 0.9, 4)
    ys = np.linspace(-0.9, 0.9, 4)
    grid_pts = np.array([[x, y] for x in xs for y in ys])
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(grid_pts))[:n]
    return grid_pts[idx]

def _refine_cartesian(pts, time_budget):
    """
    Refine a Cartesian point set using L-BFGS on a smooth surrogate.
    This acts as the local polish step in Cartesian space.
    """
    t0 = time.time()
    best = pts.copy()
    best_s = _score(pts)
    
    # Use a moderately small temperature for the soft-min to guide L-BFGS
    # towards the exact min without being too noisy.
    temp = 0.005
    
    def obj(x):
        p = x.reshape(N, 2)
        val = _soft_min_ratio(p, temp)
        if val < -1e8:
            return 1e8
        return -val
    
    try:
        res = minimize(obj, best.flatten(), method='L-BFGS-B',
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        cand = res.x.reshape(N, 2)
        s = _score(cand)
        if s > best_s:
            best_s = s
            best = cand.copy()
    except Exception:
        pass
        
    # Second pass with tighter temperature if time permits
    if time.time() - t0 < time_budget - 1:
        temp2 = 0.001
        def obj2(x):
            p = x.reshape(N, 2)
            val = _soft_min_ratio(p, temp2)
            if val < -1e8:
                return 1e8
            return -val
        try:
            res = minimize(obj2, best.flatten(), method='L-BFGS-B',
                           options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    # Final Nelder-Mead on exact objective
    if time.time() - t0 < time_budget - 0.5:
        def exact_obj(x):
            p = x.reshape(N, 2)
            r = _score(p)
            return -r
        
        try:
            result = minimize(
                exact_obj, best.flatten(), method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11,
                         'initial_simplex': 1e-3}
            )
            cand = result.x
            cand_score = _score(cand.reshape(N, 2))
            if cand_score > best_s:
                best_s = cand_score
                best = cand.reshape(N, 2).copy()
        except Exception:
            pass
    
    return best, best_s

def _optimize_from_start(initial_pts, time_budget, rng):
    """
    Optimize from a starting point using multi-stage approach:
    1. L-BFGS with smooth surrogate in Cartesian space (coarse to fine)
    2. Nelder-Mead on exact objective
    """
    start_time = time.time()
    best_pts = initial_pts.copy()
    best_score = _score(initial_pts)
    
    # Stage 1: L-BFGS with decreasing temperature in Cartesian space
    temperatures = [0.05, 0.02, 0.005, 0.001]
    
    for temp in temperatures:
        if time.time() - start_time > time_budget - 5:
            break
        
        def obj(x):
            p = x.reshape(N, 2)
            val = _soft_min_ratio(p, temp)
            if val < -1e8:
                return 1e8
            return -val
        
        try:
            result = minimize(
                obj, best_pts.flatten(), method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30}
            )
            cand = result.x.reshape(N, 2)
            cand_score = _score(cand)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = cand.copy()
        except Exception:
            pass
    
    # Stage 2: Nelder-Mead polish on exact objective in Cartesian space
    if time.time() - start_time < time_budget - 2:
        def exact_obj(x):
            pts = x.reshape(N, 2)
            r = _score(pts)
            return -r
        
        try:
            result = minimize(
                exact_obj, best_pts.flatten(), method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11,
                         'initial_simplex': 1e-3}
            )
            cand = result.x
            cand_score = _score(cand.reshape(N, 2))
            if cand_score > best_score:
                best_score = cand_score
                best_pts = cand.reshape(N, 2).copy()
        except Exception:
            pass
    
    return best_pts, best_score

def construct():
    """Main entry point: find optimal 13-point configuration."""
    start_time = time.time()
    total_budget = 145.0  # Safety margin for 150s limit
    
    rng = np.random.RandomState(12345)
    
    best_points = None
    best_ratio = -1e9
    
    # Generate diverse starting points
    starts = []
    
    # 1. Regular 13-gon
    starts.append(_regular_polygon(N, radius=1.0))
    
    # 2-4: Halton sequences with different bases
    starts.append(_generate_halton_points(N, base1=2, base2=3))
    starts.append(_generate_halton_points(N, base1=3, base2=5))
    starts.append(_generate_halton_points(N, base1=5, base2=7))
    
    # 5. Concentric rings
    starts.append(_concentric_rings(N, n_rings=3))
    
    # 6. Structured grid
    starts.append(_structured_grid(N))
    
    # 7-9: Jittered Halton
    for seed_offset in [1, 2, 3]:
        halton_pts = _generate_halton_points(N, base1=2, base2=3)
        jitter = rng.normal(0, 0.08, (N, 2))
        starts.append(halton_pts + jitter)
    
    # 10-12: Random points on unit disk
    for _ in range(3):
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2*np.pi, N)
        starts.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
    
    # 13-15: More diverse random initializations
    for _ in range(3):
        r = np.sqrt(rng.uniform(0.2, 1.0, N))
        theta = rng.uniform(0, 2*np.pi, N)
        starts.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
    
    # Evaluate initial candidates
    for cand in starts:
        if time.time() - start_time > total_budget:
            break
        try:
            s = _score(cand)
            if s > best_ratio:
                best_ratio = s
                best_points = cand.copy()
        except Exception:
            pass
    
    # Sort starts by initial score (best first)
    scored_starts = []
    for start in starts:
        try:
            s = _score(start)
            scored_starts.append((s, start))
        except Exception:
            scored_starts.append((0.0, start))
    scored_starts.sort(key=lambda x: -x[0])
    
    # Optimize from each starting point
    n_starts = len(starts)
    base_time = (total_budget - 5.0) / n_starts
    
    for i, (init_score, start) in enumerate(scored_starts):
        if time.time() - start_time > total_budget - 2:
            break
        
        remaining_time = total_budget - (time.time() - start_time)
        quality_factor = 1.0 + 0.5 * (init_score / max(best_ratio, 1e-9))
        alloc_time = min(base_time * quality_factor, remaining_time - 1.0)
        if alloc_time < 2.0:
            continue
        
        try:
            opt_pts, opt_score = _optimize_from_start(start, alloc_time, rng)
            if opt_score > best_ratio:
                best_ratio = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Final refinement on the best found
    if best_points is not None and time.time() - start_time < total_budget - 3:
        final_time = total_budget - (time.time() - start_time) - 1.0
        if final_time > 8.0:
            try:
                final_pts, final_score = _refine_cartesian(best_points, final_time)
                if final_score > best_ratio:
                    best_ratio = final_score
                    best_points = final_pts.copy()
            except Exception:
                pass
    
    if best_points is None:
        # Fallback: regular polygon
        best_points = _regular_polygon(N, radius=1.0)
    
    return best_points
# EVOLVE-BLOCK-END
