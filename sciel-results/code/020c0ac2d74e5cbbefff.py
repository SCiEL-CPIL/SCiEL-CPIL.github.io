# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time


def _triangle_areas(pts):
    """Compute all triangle areas for 13 points (286 triangles)."""
    n = len(pts)
    areas = np.empty((n * (n - 1) * (n - 2)) // 6)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                areas[idx] = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
                idx += 1
    return areas

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return np.inf

def _score(pts):
    """Compute A_min / A_hull."""
    areas = _triangle_areas(pts)
    a_min = np.min(areas)
    if a_min <= 0:
        return 0.0
    a_hull = _hull_area(pts)
    if not np.isfinite(a_hull) or a_hull <= 0:
        return 0.0
    return a_min / a_hull

def _compute_min_triangle_and_indices(pts):
    """
    Compute min area and the indices of triangles that are 'critical' (within epsilon of min).
    Returns min_area, list of critical triangle indices (i, j, k).
    """
    n = len(pts)
    areas = np.empty((n * (n - 1) * (n - 2)) // 6)
    indices = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
                areas[idx] = area
                indices.append((i, j, k))
                idx += 1
    min_area = np.min(areas)
    # Find critical triangles: those with area close to the minimum
    # Using a small tolerance to group nearly-minimum triangles
    epsilon = 1e-6 * min_area if min_area > 0 else 1e-6
    critical_indices = []
    for idx, (i, j, k) in enumerate(indices):
        if abs(areas[idx] - min_area) < epsilon:
            critical_indices.append((i, j, k))
    return min_area, critical_indices

def _directed_local_search(pts, max_iters=1000, deadline=None):
    """
    Directed local search:
    1. Find critical triangles.
    2. For each point, calculate a direction that increases the area of critical triangles involving that point.
    3. Move points in this direction with a line search.
    """
    n = len(pts)
    pts = pts.copy()
    best_score = _score(pts)
    step_size = 0.01
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        min_area, critical_indices = _compute_min_triangle_and_indices(pts)
        if len(critical_indices) == 0:
            break
            
        # Accumulate force for each point
        forces = np.zeros_like(pts)
        
        for (i, j, k) in critical_indices:
            p1, p2, p3 = pts[i], pts[j], pts[k]
            # Area = 0.5 * | (p2-p1) x (p3-p1) |
            # To increase area, we want to increase the cross product magnitude.
            # The gradient of the area with respect to p1, p2, p3 can be derived.
            # For a triangle (A, B, C), Area = 0.5 * | (B-A) x (C-A) |.
            # Let v1 = B-A, v2 = C-A. Cross = v1x*v2y - v1y*v2x.
            # If Cross > 0, grad_A = -0.5*(C-B), grad_B = 0.5*(C-A), grad_C = -0.5*(B-A).
            # If Cross < 0, signs flip.
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            
            if cross > 0:
                sign = 1
            else:
                sign = -1
                
            # The direction to increase area is to move vertices "outwards" relative to the triangle orientation.
            # A simple heuristic: move each vertex away from the centroid of the other two?
            # Or use the gradient of the absolute area.
            # Let's use the gradient of the signed area, then flip if needed to increase abs area.
            
            # Grad of signed area w.r.t A: -0.5 * (C - B)
            # Grad of signed area w.r.t B:  0.5 * (C - A)
            # Grad of signed area w.r.t C: -0.5 * (B - A)
            
            gA = -0.5 * (p3 - p2)
            gB =  0.5 * (p3 - p1)
            gC = -0.5 * (p2 - p1)
            
            # We want to increase |Area|. If cross > 0, we want to increase signed area (add grads).
            # If cross < 0, we want to decrease signed area (subtract grads).
            multiplier = sign
            
            forces[i] += multiplier * gA
            forces[j] += multiplier * gB
            forces[k] += multiplier * gC
            
        # Normalize forces to prevent huge jumps
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        # Avoid division by zero
        force_norms[force_norms == 0] = 1.0
        forces = forces / force_norms
        
        # Line search: try moving in the direction of forces
        moved = False
        for step_mult in [1.0, 0.5, 0.1, 0.01]:
            new_pts = pts + forces * step_size * step_mult
            new_score = _score(new_pts)
            if new_score > best_score:
                pts = new_pts
                best_score = new_score
                moved = True
                break
            else:
                # If it got worse, try smaller step
                continue
        
        if not moved:
            # Try random perturbation if stuck
            rng = np.random.default_rng(it)
            new_pts = pts + rng.normal(0, step_size * 0.1, size=pts.shape)
            new_score = _score(new_pts)
            if new_score > best_score:
                pts = new_pts
                best_score = new_score
            else:
                # Decay step size
                step_size *= 0.8
                if step_size < 1e-6:
                    break
        else:
            # Slightly increase step size if successful, up to a limit
            step_size = min(step_size * 1.1, 0.1)

    return pts, best_score

def _halton_sequence(count, base):
    """Generate Halton sequence in [0,1)."""
    result = np.zeros(count)
    for i in range(1, count + 1):
        f = 1.0
        r = 0.0
        n = i
        while n > 0:
            f /= base
            r += f * (n % base)
            n //= base
        result[i - 1] = r
    return result

def _generate_seeds():
    """Generate diverse initial seeds."""
    seeds = []
    
    # 1. Triangular lattice subset
    pts = []
    pts.append([0.0, 0.0])
    coords_1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    for i, j in coords_1:
        x = i + 0.5 * j
        y = (np.sqrt(3) / 2.0) * j
        pts.append([x, y])
    coords_3 = [(1, 1), (-1, 2), (2, -1), (-2, 1), (1, -2), (-1, -2)]
    for i, j in coords_3:
        x = i + 0.5 * j
        y = (np.sqrt(3) / 2.0) * j
        pts.append([x, y])
    seed1 = np.array(pts)
    seed1 = seed1 - np.mean(seed1, axis=0)
    scale = np.max(np.abs(seed1))
    if scale > 0:
        seed1 = seed1 / scale * 0.5
    seeds.append(seed1)
    
    # 2. Halton sequence points
    h1 = _halton_sequence(13, 2)
    h2 = _halton_sequence(13, 3)
    seed2 = np.column_stack((h1, h2))
    seed2 = seed2 - np.mean(seed2, axis=0)
    scale = np.max(np.abs(seed2))
    if scale > 0:
        seed2 = seed2 / scale * 0.5
    seeds.append(seed2)
    
    # 3. Random points with different seeds
    for s in [1, 2, 3]:
        rng = np.random.default_rng(s)
        seed = rng.random((13, 2))
        seed = seed - np.mean(seed, axis=0)
        scale = np.max(np.abs(seed))
        if scale > 0:
            seed = seed / scale * 0.5
        seeds.append(seed)
        
    return seeds

def heilbronn_convex13() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    
    seeds = _generate_seeds()
    
    best_pts = None
    best_score = 0.0
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        # Run directed local search
        pts, score = _directed_local_search(seed, max_iters=500, deadline=deadline)
        
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
            
    if best_pts is None:
        # Fallback
        best_pts = seeds[0]
        
    # Final polish with smaller steps if time remains
    if time.time() < deadline - 10:
        best_pts, best_score = _directed_local_search(best_pts, max_iters=200, deadline=deadline)
        
    return best_pts


# EVOLVE-BLOCK-END
