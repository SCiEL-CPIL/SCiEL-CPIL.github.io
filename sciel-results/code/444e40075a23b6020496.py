# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools
from scipy.optimize import minimize

N = 13
# Precompute all 286 triangle index triples
TRI_IDX = np.array(list(itertools.combinations(range(N), 3)))

def compute_score(points):
    """Hard Heilbronn score: A_min / A_hull."""
    if not np.all(np.isfinite(points)):
        return 0.0
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    if A_hull <= 1e-12:
        return 0.0
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    return np.min(areas) / A_hull

def compute_all_areas(pts):
    """Vectorized areas of all 286 triangles."""
    p1 = pts[TRI_IDX[:, 0]]
    p2 = pts[TRI_IDX[:, 1]]
    p3 = pts[TRI_IDX[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return np.abs(cross) * 0.5

def soft_min_score(points, temperature=0.01):
    """
    Smooth surrogate: soft-min of triangle areas / hull area.
    We maximize this. For minimization, we return -soft_min_score.
    """
    if not np.all(np.isfinite(points)):
        return -1e10
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return -1e10
    if A_hull <= 1e-12:
        return -1e10
    
    areas = compute_all_areas(points)
    if not np.all(np.isfinite(areas)):
        return -1e10
        
    # Soft min: -1/T * log(sum(exp(-A_i/T)))
    scaled = -areas / temperature
    m = np.max(scaled)
    log_sum = m + np.log(np.sum(np.exp(scaled - m)))
    soft_min_val = -log_sum * temperature
    return soft_min_val / A_hull

def local_search(points, time_limit, rng, step_size_init=0.1):
    """
    Fast coordinate ascent: only recompute the 26 triangles affected by a moved point.
    """
    pts = points.copy()
    current_score = compute_score(pts)
    best_score = current_score
    best_pts = pts.copy()
    step_size = step_size_init
    
    # Precompute: for each point, which triangle indices contain it
    tri_masks = []
    for p in range(N):
        mask = (TRI_IDX[:, 0] == p) | (TRI_IDX[:, 1] == p) | (TRI_IDX[:, 2] == p)
        tri_masks.append(mask)
        
    # Precompute indices for each point's triangles
    tri_indices = [np.where(mask)[0] for mask in tri_masks]
    
    areas = compute_all_areas(pts)
    try:
        A_hull = ConvexHull(pts).volume
    except Exception:
        return best_pts, best_score

    def fast_min_area(p, changed_pt, prev_areas):
        """Recompute only triangles involving changed_pt, return new min area."""
        idxs = tri_indices[changed_pt]
        p1 = p[TRI_IDX[idxs, 0]]
        p2 = p[TRI_IDX[idxs, 1]]
        p3 = p[TRI_IDX[idxs, 2]]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        new_areas = np.abs(cross) * 0.5
        prev_areas[idxs] = new_areas
        return np.min(prev_areas)
    
    while time.time() < time_limit and step_size > 1e-8:
        improved = False
        order = rng.permutation(N)
        for idx in order:
            for _ in range(3):
                theta = rng.uniform(0, 2 * np.pi)
                mag = step_size * rng.uniform(0.4, 1.0)
                dx = mag * np.cos(theta)
                dy = mag * np.sin(theta)
                
                old_x, old_y = pts[idx, 0], pts[idx, 1]
                pts[idx, 0] += dx
                pts[idx, 1] += dy
                
                try:
                    A_hull_new = ConvexHull(pts).volume
                except Exception:
                    pts[idx, 0] = old_x
                    pts[idx, 1] = old_y
                    continue
                
                new_min = fast_min_area(pts, idx, areas)
                new_score = new_min / A_hull_new
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    A_hull = A_hull_new
                    if current_score > best_score:
                        best_score = current_score
                        best_pts = pts.copy()
                else:
                    pts[idx, 0] = old_x
                    pts[idx, 1] = old_y
                    # Restore areas for this point
                    idxs = tri_indices[idx]
                    p1 = pts[TRI_IDX[idxs, 0]]
                    p2 = pts[TRI_IDX[idxs, 1]]
                    p3 = pts[TRI_IDX[idxs, 2]]
                    v1 = p2 - p1
                    v2 = p3 - p1
                    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
                    areas[idxs] = np.abs(cross) * 0.5
        
        if not improved:
            step_size *= 0.7
        else:
            step_size *= 0.95
    
    return best_pts, best_score

def heilbronn_convex13() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -1.0
    
    candidates = []
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    candidates.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. 7 outer, 6 inner
    ang7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    p7 = np.column_stack([np.cos(ang7), np.sin(ang7)])
    ang6 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    p6 = np.column_stack([0.55 * np.cos(ang6), 0.55 * np.sin(ang6)])
    candidates.append(np.vstack([p7, p6]))
    
    # 3. Fibonacci spiral
    golden = (1 + np.sqrt(5)) / 2
    fib_angles = 2 * np.pi * np.arange(N) / (golden ** 2)
    fib_radii = np.sqrt(np.arange(N) / N)
    candidates.append(np.column_stack([fib_radii * np.cos(fib_angles),
                                       fib_radii * np.sin(fib_angles)]))
    
    # 4. Random
    for _ in range(5):
        candidates.append(rng.random((N, 2)) * 2 - 1)
    
    # 5. Perturbed structured
    for base_idx in [0, 1, 2]:
        if base_idx < len(candidates):
            candidates.append(candidates[base_idx] + rng.normal(0, 0.03, size=(N, 2)))
            
    n_cand = len(candidates)
    
    for i, init_pts in enumerate(candidates):
        if time.time() >= deadline - 5:
            break
            
        init_score = compute_score(init_pts)
        if init_score > best_score:
            best_score = init_score
            best_points = init_pts.copy()
            
        elapsed = time.time() - start_time
        remaining = deadline - 5 - elapsed
        remaining_cands = n_cand - i
        time_per_cand = min(remaining / max(remaining_cands, 1), 60.0)
        cand_deadline = time.time() + time_per_cand
        
        cur = init_pts.copy()
        
        # Phase 1: Nelder-Mead on soft-min
        if time.time() < cand_deadline - 10:
            def objective(x):
                pts = x.reshape(N, 2)
                # Add a penalty if points go too far out (optional, but helps stability)
                return -soft_min_score(pts, temperature=0.05)
            
            try:
                res = minimize(objective, cur.flatten(), method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False})
                if res.success or res.fun < 0: # Check if it improved
                    new_pts = res.x.reshape(N, 2)
                    new_score = compute_score(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_pts.copy()
                    cur = new_pts
            except Exception:
                pass
        
        # Phase 2: Local Search
        if time.time() < cand_deadline - 5:
            ls_pts, ls_score = local_search(cur, cand_deadline - 5, rng, step_size_init=0.05)
            if ls_score > best_score:
                best_score = ls_score
                best_points = ls_pts.copy()
            cur = ls_pts
            
    # Final Polish
    if best_points is not None and time.time() < deadline - 5:
        # High precision local search
        fs_pts, fs_score = local_search(best_points, deadline - 2, rng, step_size_init=0.01)
        if fs_score > best_score:
            best_score = fs_score
            best_points = fs_pts.copy()
            
    if best_points is None or not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
