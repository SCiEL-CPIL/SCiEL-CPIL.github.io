"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations
from math import comb

# Precompute triangle indices
TRI_INDICES = list(combinations(range(N), 3))
NUM_TRIS = len(TRI_INDICES)  # 286

def _all_tri_areas(pts):
    """Compute all 286 triangle areas for a point set."""
    pts = np.asarray(pts, dtype=np.float64)
    areas = np.empty(NUM_TRIS)
    for idx, (i, j, k) in enumerate(TRI_INDICES):
        v1 = pts[j] - pts[i]
        v2 = pts[k] - pts[i]
        areas[idx] = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
    return areas

def _convex_hull_area(pts):
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
    areas = _all_tri_areas(pts)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    return min_area / hull_area

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
    
    areas = _all_tri_areas(pts)
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
    """Generate the first n terms of the Halton sequence in the given base."""
    seq = []
    f = 1.0
    for i in range(1, n+1):
        num = i
        res = 0.0
        while num > 0:
            f /= base
            num, rem = divmod(num, base)
            res += f * rem
        seq.append(res)
    return np.array(seq)

def _generate_halton_points(n_points, center=(0,0), radius=1.0):
    """Generate points using Halton sequence mapped to a disk."""
    # Use bases 2 and 3
    x = _halton_sequence(n_points, 2)
    y = _halton_sequence(n_points, 3)
    
    # Map to disk using inverse CDF of radius for uniform distribution
    # r = sqrt(U) for uniform area distribution
    r = np.sqrt(x) * radius
    theta = 2 * np.pi * y
    
    pts = np.column_stack([center[0] + r * np.cos(theta), 
                           center[1] + r * np.sin(theta)])
    return pts

def _concentric_rings(n_points, num_rings=3, center=(0,0), max_radius=1.0):
    """Generate points on concentric rings."""
    pts = []
    # Distribute points among rings
    # For 13 points: 1 center, 5 inner, 7 outer? Or 1, 4, 8?
    # Let's try 1 center, 4 inner, 8 outer
    ring_counts = [1, 4, 8]
    ring_radii = [0.0, 0.5, 1.0]
    
    idx = 0
    for count, rad in zip(ring_counts, ring_radii):
        if count == 1:
            pts.append(np.array([center[0], center[1]]))
        else:
            angles = np.linspace(0, 2*np.pi, count, endpoint=False)
            ring_pts = np.column_stack([center[0] + rad * np.cos(angles), 
                                        center[1] + rad * np.sin(angles)])
            pts.append(ring_pts)
    pts = np.vstack(pts)
    return pts[:n_points]

def _structured_grid(n_points, center=(0,0), size=1.0):
    """Generate points on a structured grid (e.g., square or hexagonal)."""
    # For 13 points, a 4x4 grid minus 3 corners? Or a hexagonal lattice?
    # Let's try a hexagonal lattice pattern
    # Center + 6 inner + 6 outer?
    # Hex lattice:
    # Row 0: 3 points
    # Row 1: 4 points
    # Row 2: 3 points
    # Row 3: 3 points? No.
    # Let's just use a jittered grid for simplicity or Halton.
    # Actually, let's stick to Halton and Rings for variety.
    # This function is a placeholder for other structured starts.
    return _regular_polygon(n_points, radius=size)

def _angular_to_cartesian(angles, radii, center=(0,0)):
    """Convert angles and radii to Cartesian coordinates."""
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    return np.column_stack([x, y])

def _cartesian_to_angular(pts, center=(0,0)):
    """Convert Cartesian points to angles and radii."""
    dx = pts[:, 0] - center[0]
    dy = pts[:, 1] - center[1]
    radii = np.sqrt(dx**2 + dy**2)
    angles = np.arctan2(dy, dx)
    return angles, radii

def _smooth_max_min_ratio(pts, temperature):
    """
    Objective function for optimization: maximize min triangle area / hull area.
    Returns negative for minimization.
    """
    if pts is None:
        return 1e9
    pts = np.asarray(pts).reshape(N, 2)
    # Check for degeneracy
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return 1e9
    
    ratio = _soft_min_ratio(pts, temperature)
    if ratio < -1e8:
        return 1e8
    return -ratio

def _neg_exact_ratio(pts_flat):
    """Negative exact score for minimization."""
    pts = pts_flat.reshape(N, 2)
    return -_score(pts)

def _optimize_from_start(start_pts, time_budget, rng):
    """
    Optimize from a starting point using differential evolution in angular/radial space,
    followed by local polish.
    """
    start_time = time.time()
    best_pts = start_pts.copy()
    best_score = _score(best_pts)
    
    # Convert to angular/radial representation for better global search
    # Sort points by angle to maintain continuity? No, just use indices.
    angles, radii = _cartesian_to_angular(best_pts)
    
    # Define bounds for angles and radii
    # Angles in [0, 2*pi), Radii in [0.1, 1.5] (allowing expansion/contraction)
    # To avoid wrap-around issues in DE, we might need to be careful.
    # Alternatively, optimize Cartesian directly with DE?
    # DE in Cartesian is 26D. DE in Angular/Radial is 26D.
    # Angular/Radial might be better conditioned for convex sets.
    
    # Let's try optimizing in Cartesian space with DE first, as it's simpler.
    # But the inspirations suggest angular/radial might work better.
    # Let's use a hybrid: DE in Cartesian.
    
    def objective(x):
        pts = x.reshape(N, 2)
        # Penalty if points go out of bounds or are too close
        # We want to maximize min area / hull area.
        # Soft min ratio is a good surrogate.
        return _smooth_max_min_ratio(pts, 0.01)
    
    bounds = [(-1.5, 1.5)] * (2 * N)
    
    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=rng.randint(0, 2**31),
            maxiter=100,
            tol=1e-8,
            polish=False,
            init='sobol',
            update_interval=10,
            popsize=15,
            mutation=(0.3, 1.0),
            recombination=0.9,
            seed=rng.randint(0, 2**31)
        )
        
        if result.fun < 1e8:
            cand_pts = result.x.reshape(N, 2)
            cand_score = _score(cand_pts)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = cand_pts.copy()
    except Exception:
        pass
    
    # Local polish with Nelder-Mead
    if time.time() - start_time < time_budget - 2:
        def exact_obj(x):
            return _neg_exact_ratio(x)
        
        try:
            result = minimize(
                exact_obj, best_pts.flatten(), method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11,
                         'initial_simplex': 1e-3}
            )
            cand_pts = result.x.reshape(N, 2)
            cand_score = _score(cand_pts)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = cand_pts.copy()
        except Exception:
            pass
    
    return best_pts, best_score

def construct():
    """Main entry point: find optimal 13-point configuration."""
    start_time = time.time()
    total_budget = 145.0  # Safety margin for 150s limit
    
    rng = np.random.RandomState(42)
    
    best_points = None
    best_ratio = -1e9
    
    # Generate diverse starting points
    starts = []
    
    # 1. Regular polygon
    starts.append(_regular_polygon(N, radius=1.0))
    
    # 2. Halton sequence
    starts.append(_generate_halton_points(N, radius=1.0))
    
    # 3. Concentric rings
    starts.append(_concentric_rings(N))
    
    # 4. Random jittered regular polygon
    for seed in [1, 2, 3]:
        rng_local = np.random.RandomState(seed)
        base = _regular_polygon(N, radius=1.0)
        jitter = rng_local.normal(0, 0.1, (N, 2))
        starts.append(base + jitter)
    
    # 5. Random points in disk
    for seed in [4, 5]:
        rng_local = np.random.RandomState(seed)
        angles = rng_local.uniform(0, 2*np.pi, N)
        radii = np.sqrt(rng_local.uniform(0, 1, N))
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        starts.append(pts)
    
    # Evaluate initial candidates
    for pts in starts:
        if time.time() - start_time > total_budget:
            break
        try:
            s = _score(pts)
            if s > best_ratio:
                best_ratio = s
                best_points = pts.copy()
        except Exception:
            pass
    
    # Sort starts by initial score (best first)
    scored_starts = []
    for pts in starts:
        try:
            s = _score(pts)
            scored_starts.append((s, pts))
        except Exception:
            scored_starts.append((0.0, pts))
    scored_starts.sort(key=lambda x: -x[0])
    
    n_starts = len(scored_starts)
    base_time = (total_budget - 5.0) / n_starts
    
    # Optimize from each starting point
    for i, (init_score, pts) in enumerate(scored_starts):
        if time.time() - start_time > total_budget - 2:
            break
        
        remaining_time = total_budget - (time.time() - start_time)
        # Give more time to better initial points
        quality_factor = 1.0 + 0.5 * (init_score / max(best_ratio, 1e-9))
        alloc_time = min(base_time * quality_factor, remaining_time - 1.0)
        if alloc_time < 2.0:
            continue
        
        try:
            opt_pts, opt_score = _optimize_from_start(pts, alloc_time, rng)
            if opt_score > best_ratio:
                best_ratio = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Final refinement on the best found
    if best_points is not None and time.time() - start_time < total_budget - 3:
        final_time = total_budget - (time.time() - start_time) - 1.0
        if final_time > 8.0:
            def exact_obj(x):
                return _neg_exact_ratio(x)
            
            try:
                result = minimize(
                    exact_obj, best_points.flatten(), method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12,
                             'initial_simplex': 1e-4}
                )
                cand_pts = result.x.reshape(N, 2)
                cand_score = _score(cand_pts)
                if cand_score > best_ratio:
                    best_ratio = cand_score
                    best_points = cand_pts.copy()
            except Exception:
                pass
    
    if best_points is None:
        # Fallback: regular polygon
        best_points = _regular_polygon(N, radius=1.0)
    
    return best_points
# EVOLVE-BLOCK-END
