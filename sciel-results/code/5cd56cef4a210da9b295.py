# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE = 140

_TRIPLETS = list(itertools.combinations(range(11), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

def _min_triangle_area(pts):
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)

def _cartesian_to_barycentric(pts):
    n = len(pts)
    abc = np.zeros((n, 3))
    c = pts[:, 1] / S3H
    b = pts[:, 0] - 0.5 * c
    a = 1.0 - b - c
    abc[:, 0] = a
    abc[:, 1] = b
    abc[:, 2] = c
    return abc

def _barycentric_to_cartesian(abc):
    n = len(abc)
    pts = np.zeros((n, 2))
    pts[:, 0] = abc[:, 1] + 0.5 * abc[:, 2]
    pts[:, 1] = abc[:, 2] * S3H
    return pts

def _project_to_triangle(pts):
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts

def _normalize_barycentric(abc):
    abc = np.clip(abc, 0.0, 1.0)
    sums = abc.sum(axis=1, keepdims=True)
    sums = np.where(sums > 1e-12, sums, 1.0)
    abc = abc / sums
    return abc

def _objective_bary_hard(flat):
    abc = flat.reshape(-1, 3)
    pts = _barycentric_to_cartesian(abc)
    ma = _min_triangle_area(pts)
    # Penalty for going out of bounds
    neg = np.minimum(abc, 0.0)
    pos = np.maximum(abc - 1.0, 0.0)
    pen = 100.0 * (np.sum(neg ** 2) + np.sum(pos ** 2))
    return -ma + pen

def construct():
    start_time = time.time()
    rng = np.random.default_rng(123)
    
    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts.copy())
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # 1. Generate a discrete grid of candidate points
    # Grid size 15x15 in barycentric coordinates
    grid_size = 15
    grid_pts = []
    for i in range(grid_size):
        for j in range(grid_size):
            b = i / (grid_size - 1)
            c = j / (grid_size - 1)
            a = 1.0 - b - c
            if a >= -1e-6 and b >= -1e-6 and c >= -1e-6:
                # Clip to valid range
                a = max(0, a)
                b = max(0, b)
                c = max(0, c)
                # Normalize to ensure sum=1 (handles clipping)
                s = a + b + c
                if s > 0:
                    a /= s; b /= s; c /= s
                pts = _barycentric_to_cartesian(np.array([[a, b, c]]))
                grid_pts.append(pts[0])
    grid_pts = np.array(grid_pts)
    
    # 2. Branch and Bound / Backtracking search on the grid
    # We want to place 11 points from grid_pts such that min area is maximized.
    # Since grid is small (~100 points), we can try a greedy randomized approach 
    # or a limited backtracking.
    # Full backtracking is too slow. We will use a "Greedy with Random Restarts" 
    # which is a form of discrete search.
    
    # Initial guess from continuous optimization (keep some of the old logic for robustness if grid fails)
    init = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, S3H], [0.5, 0.0], [0.75, S3H / 2],
        [0.25, S3H / 2], [1/3, S3H/3], [0.2, S3H/6], [0.8, S3H/6],
        [0.5, 2*S3H/3], [0.35, S3H/2.5]
    ])
    _update_best(init)

    # Discrete Search: Greedy Randomized Construction
    # We pick points one by one. At each step, we choose a point from the grid 
    # that maximizes the minimum area with the already chosen points.
    # To avoid getting stuck, we randomize the choice among the top K candidates.
    
    num_restarts = 50
    for r in range(num_restarts):
        if not _elapsed_ok(60):
            break
            
        chosen_indices = []
        chosen_pts = []
        available = list(range(len(grid_pts)))
        
        # Sort available indices initially? No, we evaluate on the fly.
        
        for k in range(11):
            if not available:
                break
            
            # Calculate min area with current chosen points for each candidate
            # If k==0, area is inf (or 0, doesn't matter, we just pick one)
            # If k>0, we calculate min area of triangles formed by candidate and any pair of chosen points?
            # Actually, the "min area" of the set is the min over ALL triplets.
            # When adding a new point P to set S, the new min area is 
            # min( old_min_area(S), min_{i,j in S} Area(P, Si, Sj) )
            
            if len(chosen_pts) < 2:
                # Not enough points to form a triangle, pick randomly or based on spread
                # Let's just pick randomly to diversify
                idx = rng.choice(len(available))
                chosen_indices.append(available[idx])
                chosen_pts.append(grid_pts[available[idx]])
                available.pop(idx)
            else:
                # Evaluate all available candidates
                # We need min_{i,j in chosen} Area(Cand, chosen_i, chosen_j)
                # Vectorized: 
                # Cand is (1, 2). Chosen is (k, 2).
                # We can compute areas for all pairs in chosen against Cand.
                
                best_idx = -1
                best_val = -1
                candidates_vals = []
                
                for ai in available:
                    cand = grid_pts[ai]
                    # Compute areas with all pairs in chosen_pts
                    # chosen_pts is a list of arrays
                    cp = np.array(chosen_pts)
                    # Pairs in cp
                    # Use precomputed indices for chosen? 
                    # Simpler: loop over pairs in chosen (k is small, max 10)
                    min_area_with_chosen = np.inf
                    for i in range(len(cp)):
                        for j in range(i+1, len(cp)):
                            p1 = cp[i]
                            p2 = cp[j]
                            # Area of (cand, p1, p2)
                            dx1 = p1[0] - cand[0]
                            dy1 = p1[1] - cand[1]
                            dx2 = p2[0] - cand[0]
                            dy2 = p2[1] - cand[1]
                            area = 0.5 * abs(dx1*dy2 - dy1*dx2)
                            if area < min_area_with_chosen:
                                min_area_with_chosen = area
                                if min_area_with_chosen <= best_val:
                                    break
                        if min_area_with_chosen <= best_val:
                            break
                    
                    # Also, the global min area is min(current_global_min, min_area_with_chosen)
                    # We don't have current_global_min easily accessible without recalculating, 
                    # but since we are building up, the constraint is that the NEW triangles must be large.
                    # The old triangles are already fixed. 
                    # So the new global min is min(old_global_min, min_area_with_chosen).
                    # We want to maximize this.
                    # Since old_global_min is constant for this step, maximizing min_area_with_chosen is sufficient.
                    
                    candidates_vals.append((min_area_with_chosen, ai))
                
                if not candidates_vals:
                    break
                    
                # Sort by value descending
                candidates_vals.sort(key=lambda x: x[0], reverse=True)
                
                # Pick from top 5 (randomized greedy)
                top_k = min(5, len(candidates_vals))
                selected = rng.choice(top_k)
                best_idx = candidates_vals[selected][1]
                
                chosen_indices.append(best_idx)
                chosen_pts.append(grid_pts[best_idx])
                available.remove(best_idx)
        
        if len(chosen_pts) == 11:
            pts = np.array(chosen_pts)
            _update_best(pts)

    # 3. Polish the best discrete solution with Continuous Local Search
    if best_pts is not None and _elapsed_ok(30):
        # Try polishing from the best discrete point
        # Also try polishing from the initial continuous point if it was better
        candidates = [best_pts]
        
        # Add a few random perturbations of the best discrete point
        for _ in range(5):
            if not _elapsed_ok(10):
                break
            perturbed = best_pts + rng.normal(0, 0.01, size=(11, 2))
            candidates.append(perturbed)
        
        for cand in candidates:
            if not _elapsed_ok(5):
                break
            try:
                abc = _cartesian_to_barycentric(cand)
                abc = _normalize_barycentric(abc)
                res = minimize(
                    _objective_bary_hard,
                    abc.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                abc_res = _normalize_barycentric(res.x.reshape(-1, 3))
                p = _barycentric_to_cartesian(abc_res)
                p = _project_to_triangle(p)
                _update_best(p)
            except Exception:
                pass

    # 4. If we have time, do a bit of continuous DE or SA? 
    # The directive says "structural change... discrete constraint satisfaction". 
    # I have done the discrete part. 
    # Let's add a small amount of continuous refinement if time permits, 
    # but the core is the grid search.
    
    if best_pts is None:
        pts = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, S3H], [0.5, 0.0], [0.75, S3H / 2],
            [0.25, S3H / 2], [1/3, S3H/3], [0.2, S3H/6], [0.8, S3H/6],
            [0.5, 2*S3H/3], [0.35, S3H/2.5]
        ])
        return pts

    best_pts = _project_to_triangle(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
