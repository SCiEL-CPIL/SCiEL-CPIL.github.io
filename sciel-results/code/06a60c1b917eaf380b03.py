import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize, dual_annealing

N = 11
H = np.sqrt(3) / 2
# Vertices of the unit equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) = 165 triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)
_n_tris = len(_idx_i)

# Precompute point masks: which triangles involve each point
_point_masks = []
for p in range(N):
    mask = ((_idx_i == p) | (_idx_j == p) | (_idx_k == p))
    _point_masks.append(mask)

def _clamp_to_triangle(points):
    """Clamp points to be inside the triangle."""
    points = points.copy()
    for i in range(N):
        x, y = points[i]
        # y is bounded by [0, H]
        y = max(0.0, min(H, y))
        # x is bounded by the slanted sides
        # Left side: y = sqrt(3) * x  => x = y / sqrt(3)
        # Right side: y = sqrt(3) * (1 - x) => x = 1 - y / sqrt(3)
        x_min = y / np.sqrt(3)
        x_max = 1.0 - y / np.sqrt(3)
        if x_min > x_max:
            x = 0.5
            y = 0.0
        else:
            x = max(x_min, min(x_max, x))
        points[i] = [x, y]
    return points

def _min_area_fast(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    dx1 = p2[:, 0] - p1[:, 0]
    dy1 = p2[:, 1] - p1[:, 1]
    dx2 = p3[:, 0] - p1[:, 0]
    dy2 = p3[:, 1] - p1[:, 1]
    cross = np.abs(dx1 * dy2 - dy1 * dx2)
    return 0.5 * np.min(cross)

def _min_area_and_idx(points):
    """Return (min_area, index_of_worst_triangle)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    dx1 = p2[:, 0] - p1[:, 0]
    dy1 = p2[:, 1] - p1[:, 1]
    dx2 = p3[:, 0] - p1[:, 0]
    dy2 = p3[:, 1] - p1[:, 1]
    cross = np.abs(dx1 * dy2 - dy1 * dx2)
    areas = 0.5 * cross
    idx = int(np.argmin(areas))
    return areas[idx], idx

def _inside_penalty(points, margin=0.001):
    """Smooth penalty for points outside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    # Line 1: y >= 0
    viol1 = np.maximum(0.0, -y + margin)
    # Line 2: y <= sqrt(3) * x  =>  y - sqrt(3)*x <= 0
    viol2 = np.maximum(0.0, y - np.sqrt(3) * x + margin)
    # Line 3: y <= sqrt(3) * (1 - x) => y - sqrt(3)*(1-x) <= 0
    viol3 = np.maximum(0.0, y - np.sqrt(3) * (1.0 - x) + margin)
    return np.sum(viol1**2) + np.sum(viol2**2) + np.sum(viol3**2)

def _is_inside(points, tol=1e-8):
    """Check if all points are inside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    return (np.all(y >= -tol) and
            np.all(y <= np.sqrt(3) * x + tol) and
            np.all(y <= np.sqrt(3) * (1.0 - x) + tol))

def _objective(params):
    """Objective: negative min area + boundary penalty."""
    points = params.reshape(N, 2)
    pen = _inside_penalty(points)
    if pen > 1e-4:
        return -_min_area_fast(points) + 1000.0 * pen
    return -_min_area_fast(points)

def _local_refine_worst_triangle(points, time_limit):
    """
    Iteratively refine by optimizing the worst triangle's points.
    This is a block coordinate descent method.
    """
    start = time.time()
    best_points = points.copy()
    best_score = _min_area_fast(points)
    
    # We will iterate, finding the worst triangle and trying to improve it.
    # To avoid getting stuck in a cycle, we keep a history or just rely on the score improvement check.
    
    for iteration in range(1000):
        if time.time() > time_limit:
            break
        
        min_area, worst_idx = _min_area_and_idx(best_points)
        
        if min_area > best_score + 1e-12:
            best_score = min_area
            best_points = points.copy() # This line is redundant if we update best_points below, but safe.
        
        i, j, k = _idx_i[worst_idx], _idx_j[worst_idx], _idx_k[worst_idx]
        tri_points = best_points[[i, j, k]].copy()
        
        # Mask for all triangles involving i, j, or k
        combined_mask = _point_masks[i] | _point_masks[j] | _point_masks[k]
        
        def _tri_obj(tri_params):
            tp = tri_params.reshape(3, 2)
            # Check bounds loosely to avoid expensive penalty calc if way out
            x, y = tp[:, 0], tp[:, 1]
            if np.any(y < -0.1) or np.any(y > H + 0.1):
                return 1000.0
            if np.any(y > np.sqrt(3)*x + 0.1) or np.any(y > np.sqrt(3)*(1-x) + 0.1):
                return 1000.0
            
            full = best_points.copy()
            full[i], full[j], full[k] = tp[0], tp[1], tp[2]
            
            # Compute areas for all triangles involving i, j, or k
            p1 = full[_idx_i]
            p2 = full[_idx_j]
            p3 = full[_idx_k]
            dx1 = p2[:, 0] - p1[:, 0]
            dy1 = p2[:, 1] - p1[:, 1]
            dx2 = p3[:, 0] - p1[:, 0]
            dy2 = p3[:, 1] - p1[:, 1]
            cross = np.abs(dx1 * dy2 - dy1 * dx2)
            areas = 0.5 * cross
            
            # We want to maximize the minimum area among the affected triangles
            # But we also need to ensure we don't break other triangles too badly?
            # Actually, since we only move i,j,k, only triangles involving them change.
            # So minimizing the min of affected areas is sufficient for local improvement.
            return -np.min(areas[combined_mask])
        
        try:
            # Use Nelder-Mead for this 6D subproblem
            res = minimize(_tri_obj, tri_points.flatten(), method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            
            # Check if the result is better
            new_tri = res.x.reshape(3, 2)
            # Clamp to be safe
            new_tri = _clamp_to_triangle(new_tri)
            
            # Evaluate new global score
            test_points = best_points.copy()
            test_points[i], test_points[j], test_points[k] = new_tri[0], new_tri[1], new_tri[2]
            
            if _is_inside(test_points, tol=1e-8):
                new_score = _min_area_fast(test_points)
                if new_score > best_score:
                    best_points = test_points
                    best_score = new_score
                else:
                    # If no improvement, we might be stuck. 
                    # We could try a random perturbation or break.
                    # For now, just continue.
                    pass
        except Exception:
            pass

    return best_points, best_score

def _global_optimize(points, time_limit, seed=42):
    """Run differential evolution from current best."""
    start = time.time()
    bounds = [(0, 1), (0, H)] * N
    
    try:
        # DE is slow, so we limit iterations
        # popsize=15 is a good balance
        result = differential_evolution(
            _objective,
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            seed=seed,
            polish=False,
            init='latinhypercube',
            mutation=(0.5, 0.9),
            recombination=0.7
        )
        de_points = result.x.reshape(N, 2)
        de_points = _clamp_to_triangle(de_points)
        return de_points
    except Exception:
        return points

def _global_optimize_da(points, time_limit, seed=42):
    """Run dual annealing."""
    start = time.time()
    bounds = [(0, 1), (0, H)] * N
    try:
        da_result = dual_annealing(
            _objective,
            bounds=bounds,
            seed=seed,
            maxiter=50,
            restarts=2,
            initial_temp=5230.0,
            visit=[1.1, 1.3],
            clip=[-1, 1]
        )
        da_points = da_result.x.reshape(N, 2)
        da_points = _clamp_to_triangle(da_points)
        return da_points
    except Exception:
        return points

def _init_symmetric():
    """Generate a good symmetric starting configuration."""
    points = np.zeros((N, 2))
    # Heuristic layout for 11 points
    # Row 0 (y=0): 4 points? Or 3?
    # Let's try a standard grid-like approach
    # y=0: 0.1, 0.3, 0.7, 0.9 (4 pts)
    # y=H/3: 0.25, 0.5, 0.75 (3 pts)
    # y=2H/3: 0.4, 0.6 (2 pts)
    # y=H: 0.5 (1 pt)
    # Total 10. Need 1. Add one more?
    # Let's just use random perturbation of a lattice
    
    # Better: Use a known good configuration if available, or generate via lattice
    # Lattice points in triangle
    candidates = []
    for i in range(5):
        for j in range(5):
            # Barycentric coords
            u = i / 4.0
            v = j / 4.0
            w = 1.0 - u - v
            if w >= -1e-9:
                w = max(0.0, w)
                x = v * 1.0 + w * 0.5
                y = w * H
                if x >= -1e-9 and x <= 1 + 1e-9 and y >= -1e-9 and y <= H + 1e-9:
                    candidates.append([x, y])
    
    candidates = np.array(candidates)
    if len(candidates) < N:
        # Add random
        while len(candidates) < N:
            u = np.random.random()
            v = np.random.random()
            if u + v <= 1:
                x = v + 0.5 * (1 - u - v)
                y = H * (1 - u - v)
                candidates = np.vstack([candidates, [[x, y]]])
    
    # Select N points that maximize min distance (greedy)
    selected = [candidates[0]]
    for _ in range(N - 1):
        dists = np.min([np.linalg.norm(candidates - s, axis=1) for s in selected], axis=0)
        next_idx = np.argmax(dists)
        selected.append(candidates[next_idx])
        
    pts = np.array(selected)
    # Add small noise
    pts += np.random.normal(0, 0.01, size=pts.shape)
    return _clamp_to_triangle(pts)

def construct():
    start_time = time.time()
    # Safety margin: 120s limit, use 115s for safety
    deadline = start_time + 115
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Initializations
    inits = [
        _init_symmetric(),
    ]
    
    # Add a few random inits
    for _ in range(3):
        pts = np.random.random((N, 2))
        pts[:, 1] *= H
        # Project to triangle
        pts = _clamp_to_triangle(pts)
        inits.append(pts)
        
    for init in inits:
        score = _min_area_fast(init)
        if score > best_score:
            best_score = score
            best_points = init.copy()
            
    # Phase 2: Global Optimization + Local Refinement Loop
    # We will run a few rounds of DE/DA, and refine the best result from each.
    
    # Round 1: DE
    if time.time() < deadline - 60:
        t_limit = time.time() + 25
        de_points = _global_optimize(best_points, t_limit, seed=1)
        de_score = _min_area_fast(de_points)
        if de_score > best_score:
            best_score = de_score
            best_points = de_points.copy()
            
        # Refine DE result
        if time.time() < deadline - 30:
            t_limit = time.time() + 15
            refined_pts, refined_score = _local_refine_worst_triangle(best_points, t_limit)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_pts.copy()

    # Round 2: DA
    if time.time() < deadline - 40:
        t_limit = time.time() + 20
        da_points = _global_optimize_da(best_points, t_limit, seed=2)
        da_score = _min_area_fast(da_points)
        if da_score > best_score:
            best_score = da_score
            best_points = da_points.copy()
            
        # Refine DA result
        if time.time() < deadline - 20:
            t_limit = time.time() + 15
            refined_pts, refined_score = _local_refine_worst_triangle(best_points, t_limit)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_pts.copy()

    # Round 3: Another DE with different seed, starting from best
    if time.time() < deadline - 25:
        t_limit = time.time() + 15
        de_points = _global_optimize(best_points, t_limit, seed=3)
        de_score = _min_area_fast(de_points)
        if de_score > best_score:
            best_score = de_score
            best_points = de_points.copy()
            
        # Refine
        if time.time() < deadline - 10:
            t_limit = time.time() + 8
            refined_pts, refined_score = _local_refine_worst_triangle(best_points, t_limit)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_pts.copy()

    # Final Polish: Local Refine on the very best
    if time.time() < deadline - 5:
        t_limit = time.time() + 4
        refined_pts, refined_score = _local_refine_worst_triangle(best_points, t_limit)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts.copy()

    if best_points is None:
        best_points = _init_symmetric()
        
    return _clamp_to_triangle(best_points)
