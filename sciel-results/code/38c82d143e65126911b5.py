import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

def _precompute_triangles(n=13):
    """Precompute triangle index combinations and area coefficient matrices."""
    combos = list(itertools.combinations(range(n), 3))
    tri_idx = np.array(combos, dtype=np.int32)  # shape (286, 3)
    return tri_idx

def _triangle_areas(points, tri_idx):
    """Compute areas of all triangles defined by tri_idx for given points."""
    x1 = points[tri_idx[:, 0], 0]
    y1 = points[tri_idx[:, 0], 1]
    x2 = points[tri_idx[:, 1], 0]
    y2 = points[tri_idx[:, 1], 1]
    x3 = points[tri_idx[:, 2], 0]
    y3 = points[tri_idx[:, 2], 1]
    areas = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    return areas

def _hull_area(points):
    """Compute convex hull area, return 0 if degenerate."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points, tri_idx):
    """Compute A_min / A_hull score."""
    n = points.shape[0]
    if n < 3 or not np.all(np.isfinite(points)):
        return 0.0
    A_hull = _hull_area(points)
    if A_hull <= 1e-12:
        return 0.0
    areas = _triangle_areas(points, tri_idx)
    A_min = np.min(areas)
    if A_min <= 0:
        return 0.0
    return A_min / A_hull

def _soft_min(areas, temperature):
    """Smooth approximation of min using log-sum-exp (negative for minimization)."""
    # Use a soft min: -log(sum(exp(-areas/T))) * T  approximates min(areas)
    # For numerical stability
    if temperature <= 0:
        return np.min(areas)
    # Scale to avoid overflow
    min_val = np.min(areas)
    exp_terms = np.exp(-(areas - min_val) / temperature)
    soft = min_val + temperature * np.log(np.sum(exp_terms))
    return soft

def _negative_soft_score(points, tri_idx, temperature=0.01):
    """Objective: minimize -soft_min(areas)/A_hull."""
    A_hull = _hull_area(points)
    if A_hull <= 1e-12:
        return 1e10
    areas = _triangle_areas(points, tri_idx)
    soft_min_val = _soft_min(areas, temperature)
    if not np.isfinite(soft_min_val):
        return 1e10
    return -(soft_min_val / A_hull)

def _lattice_search(tri_idx, time_budget=120.0):
    """
    Structured lattice-based search over a grid of point configurations.
    Uses a regular polygon base with radial and angular perturbations on a grid.
    """
    start_time = time.time()
    n = 13
    best_points = None
    best_score = -np.inf
    
    # Define a parameterized family: regular n-gon with per-point radial scaling
    # and angular offset. We'll search over a coarse grid.
    
    # Radial grid: 3 levels
    radial_levels = [0.7, 0.85, 1.0, 1.15, 1.3]
    # Angular shift: a few values
    angle_shifts = [0.0, 0.1, 0.2, 0.3]
    # Per-point radial multipliers: use a few symmetric patterns
    # Pattern 0: all equal (regular)
    # Pattern 1: alternating high/low
    # Pattern 2: one point pulled in
    # Pattern 3: two adjacent pulled out
    
    patterns = [np.ones(n)]
    # Alternating
    alt = np.ones(n)
    alt[::2] = 1.15
    alt[1::2] = 0.85
    patterns.append(alt)
    # Single dip
    for dip_i in range(0, n, 3):
        p = np.ones(n)
        p[dip_i] = 0.7
        patterns.append(p)
    # Single bump
    for bump_i in range(0, n, 3):
        p = np.ones(n)
        p[bump_i] = 1.3
        patterns.append(p)
    
    best_points = None
    best_score = -np.inf
    
    for shift in angle_shifts:
        for pat in patterns:
            if time.time() - start_time > time_budget - 10:
                break
            base_angles = 2 * np.pi * np.arange(n) / n + shift
            for r_scale in radial_levels:
                radii = pat * r_scale
                points = np.column_stack((radii * np.cos(base_angles), radii * np.sin(base_angles)))
                s = _score(points, tri_idx)
                if s > best_score:
                    best_score = s
                    best_points = points.copy()
    
    return best_points, best_score

def _local_refine(points, tri_idx, time_budget=60.0):
    """
    Local refinement using L-BFGS-B on the 2D coordinates directly,
    with a soft-min objective for smoothness.
    """
    start_time = time.time()
    best_points = points.copy()
    best_score = _score(points, tri_idx)
    
    # Optimize in 2D directly (26 parameters)
    x0 = points.flatten()
    
    bounds = [(-2.0, 2.0)] * (2 * points.shape[0])
    
    for temp in [0.05, 0.02, 0.01, 0.005]:
        if time.time() - start_time > time_budget - 10:
            break
        
        def obj(x):
            pts = x.reshape(-1, 2)
            return _negative_soft_score(pts, tri_idx, temperature=temp)
        
        try:
            res = minimize(
                obj, x0, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            pts_opt = res.x.reshape(-1, 2)
            s = _score(pts_opt, tri_idx)
            if s > best_score:
                best_score = s
                best_points = pts_opt.copy()
                x0 = res.x
        except Exception:
            pass
    
    return best_points, best_score

def _hill_climb(points, tri_idx, time_budget=30.0):
    """
    Simple stochastic hill climbing with decreasing step sizes.
    Tries random perturbations and keeps improvements.
    """
    start_time = time.time()
    current = points.copy()
    best = points.copy()
    best_score = _score(points, tri_idx)
    current_score = best_score
    
    n = points.shape[0]
    step_size = 0.05
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < time_budget - 5:
        # Generate a perturbation
        perturbation = rng.normal(0, step_size, size=(n, 2))
        candidate = current + perturbation
        
        cand_score = _score(candidate, tri_idx)
        
        if cand_score > current_score:
            current = candidate
            current_score = cand_score
            if cand_score > best_score:
                best = candidate.copy()
                best_score = cand_score
        else:
            # Occasionally accept worse moves (simulated annealing-lite)
            if rng.random() < 0.05:
                current = candidate
                current_score = cand_score
        
        # Decrease step size gradually
        step_size *= 0.99
        
        # If stuck, try a larger jump
        if step_size < 0.001:
            step_size = 0.05
            perturbation = rng.normal(0, 0.1, size=(n, 2))
            current = current + perturbation
            current_score = _score(current, tri_idx)
    
    return best, best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the area of the
    smallest triangle formed by these points.
    """
    total_start = time.time()
    max_time = 600.0
    
    n = 13
    tri_idx = _precompute_triangles(n)
    
    # Phase 1: Lattice/structured search (find good basins)
    lattice_time = min(120.0, max_time * 0.2)
    best_points, best_score = _lattice_search(tri_idx, time_budget=lattice_time)
    
    if best_points is None:
        # Fallback: regular polygon
        angles = 2 * np.pi * np.arange(n) / n
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _score(best_points, tri_idx)
    
    # Phase 2: Local refinement with gradient-based optimizer
    elapsed = time.time() - total_start
    local_time = max(60.0, max_time * 0.25 - elapsed)
    if local_time > 20:
        refined_points, refined_score = _local_refine(best_points, tri_idx, time_budget=local_time)
        if refined_score > best_score:
            best_points = refined_points
            best_score = refined_score
    
    # Phase 3: Hill climbing for fine-tuning
    elapsed = time.time() - total_start
    hc_time = max(30.0, max_time * 0.2 - elapsed)
    if hc_time > 15:
        hc_points, hc_score = _hill_climb(best_points, tri_idx, time_budget=hc_time)
        if hc_score > best_score:
            best_points = hc_points
            best_score = hc_score
    
    # Final safety check
    elapsed = time.time() - total_start
    if elapsed > max_time - 10:
        pass  # Just return what we have
    
    # Validate
    final_score = _score(best_points, tri_idx)
    if final_score <= 0:
        angles = 2 * np.pi * np.arange(n) / n
        return np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_points

if __name__ == "__main__":
    pts = heilbronn_convex13()
    print(f"Score: {_score(pts, _precompute_triangles(13))}")
    print(pts)
