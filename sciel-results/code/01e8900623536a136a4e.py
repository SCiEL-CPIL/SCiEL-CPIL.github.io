"""
Strategy: Full 14-point optimization with push-pull local search and multi-start.
Abandons centrally symmetric constraint. Uses directed geometric moves:
identify worst triangle, push vertices outward to increase area.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 14

def _time_left(deadline):
    return deadline - time.time()

def _min_triangle_area(points):
    """Compute minimum triangle area. Returns (min_area, i, j, k) of the worst triangle."""
    n = len(points)
    min_area = float('inf')
    worst = (0, 1, 2)
    x = points[:, 0]
    y = points[:, 1]
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((x[j]-x[i])*(y[k]-y[i]) - (x[k]-x[i])*(y[j]-y[i]))
                if area < min_area:
                    min_area = area
                    worst = (i, j, k)
    return min_area, worst

def _all_triangle_areas(points):
    """Compute all triangle areas as a flat array."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas = np.empty((n*(n-1)*(n-2))//6)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                areas[idx] = 0.5 * abs((x[j]-x[i])*(y[k]-y[i]) - (x[k]-x[i])*(y[j]-y[i]))
                idx += 1
    return areas

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_area, _ = _min_triangle_area(points)
    return min_area / hull_area

def _generate_initial_points():
    """Generate diverse initial configurations."""
    configs = []
    rng = np.random.default_rng(12345)
    
    # Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Slightly perturbed regular polygon
    for _ in range(3):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False) + rng.uniform(-0.05, 0.05, N)
        radii = 1.0 + rng.uniform(-0.05, 0.05, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        configs.append(pts)
    
    # Random points in unit disk, sorted by angle
    for _ in range(8):
        angles = np.sort(rng.uniform(0, 2*np.pi, N))
        radii = rng.uniform(0.3, 1.0, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        configs.append(pts)
    
    # Two concentric rings
    for _ in range(3):
        r1 = 0.5 + rng.uniform(-0.1, 0.1)
        r2 = 1.0 + rng.uniform(-0.1, 0.1)
        angles1 = np.linspace(0, 2*np.pi, 7, endpoint=False) + rng.uniform(-0.1, 0.1, 7)
        angles2 = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7 + rng.uniform(-0.1, 0.1, 7)
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        # Interleave by angle
        all_pts = np.vstack([pts1, pts2])
        ang = np.arctan2(all_pts[:,1], all_pts[:,0])
        idx = np.argsort(ang)
        configs.append(all_pts[idx])
    
    return configs

def _push_pull_step(points, step_size=0.02, max_iter=500, deadline=None):
    """
    Directed push-pull local search:
    - Find worst triangle
    - Push its vertices outward (away from centroid of the other points)
    - If hull degenerates or score drops, reduce step size
    """
    current = points.copy()
    current_score = _score(current)
    if current_score <= 0:
        return current, current_score
    
    best = current.copy()
    best_score = current_score
    step = step_size
    stagnant = 0
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        min_area, (i, j, k) = _min_triangle_area(current)
        hull_area = _hull_area(current)
        if hull_area < 1e-12:
            break
        
        # Centroid of the three vertices of the worst triangle
        tri_centroid = (current[i] + current[j] + current[k]) / 3.0
        
        # Direction to push: away from the centroid of ALL other points
        other_indices = [m for m in range(N) if m not in (i, j, k)]
        other_centroid = current[other_indices].mean(axis=0)
        
        # Push direction for each vertex of the worst triangle
        push_dirs = np.zeros((3, 2))
        for idx, vi in enumerate([i, j, k]):
            d = current[vi] - other_centroid
            norm = np.linalg.norm(d)
            if norm > 1e-10:
                push_dirs[idx] = d / norm
            else:
                # Fallback: push away from triangle centroid
                d = current[vi] - tri_centroid
                norm = np.linalg.norm(d)
                if norm > 1e-10:
                    push_dirs[idx] = d / norm
        
        # Try the push
        new_points = current.copy()
        new_points[i] += step * push_dirs[0]
        new_points[j] += step * push_dirs[1]
        new_points[k] += step * push_dirs[2]
        
        new_score = _score(new_points)
        
        if new_score > current_score:
            current = new_points
            current_score = new_score
            stagnant = 0
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        else:
            stagnant += 1
            if stagnant > 20:
                step *= 0.8
                stagnant = 0
                if step < 1e-6:
                    break
    
    return best, best_score

def _nelder_mead_polish(points, deadline, maxiter=1000):
    """
    Nelder-Mead polish using a vector of all 28 coordinates.
    Objective: minimize -score.
    """
    from scipy.optimize import minimize
    
    def objective_flat(flat):
        pts = flat.reshape(N, 2)
        s = _score(pts)
        if s < 1e-10:
            return 1e10
        return -s
    
    x0 = points.flatten()
    try:
        res = minimize(
            objective_flat,
            x0,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if time.time() > deadline:
            return points
        final_pts = res.x.reshape(N, 2)
        if _score(final_pts) > _score(points):
            return final_pts
    except Exception:
        pass
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    initial_configs = _generate_initial_configs()
    
    best_points = None
    best_score_val = -1.0
    
    # Phase 1: Evaluate initial configs and pick the best few for refinement
    scored_configs = []
    for cfg in initial_configs:
        if time.time() > deadline - 40:
            break
        s = _score(cfg)
        scored_configs.append((s, cfg))
    
    scored_configs.sort(key=lambda x: -x[0])
    
    # Phase 2: Push-pull local search on top initial configs
    for s, cfg in scored_configs[:6]:
        if time.time() > deadline - 30:
            break
        try:
            refined, ref_score = _push_pull_step(cfg, step_size=0.02, max_iter=300, deadline=deadline)
            if ref_score > best_score_val:
                best_points = refined
                best_score_val = ref_score
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead polish on the best point set
    if best_points is not None and time.time() < deadline - 10:
        try:
            polished = _nelder_mead_polish(best_points, deadline, maxiter=2000)
            pol_score = _score(polished)
            if pol_score > best_score_val:
                best_points = polished
                best_score_val = pol_score
        except Exception:
            pass
    
    # Phase 4: Another round of push-pull on the polished result
    if best_points is not None and time.time() < deadline - 5:
        try:
            refined2, ref2_score = _push_pull_step(best_points, step_size=0.01, max_iter=200, deadline=deadline)
            if ref2_score > best_score_val:
                best_points = refined2
                best_score_val = ref2_score
        except Exception:
            pass
    
    # Phase 5: Final Nelder-Mead polish
    if best_points is not None and time.time() < deadline - 2:
        try:
            polished2 = _nelder_mead_polish(best_points, deadline, maxiter=500)
            pol2_score = _score(polished2)
            if pol2_score > best_score_val:
                best_points = polished2
                best_score_val = pol2_score
        except Exception:
            pass
    
    # If nothing worked, fallback to regular polygon
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize to unit area
    hull_area = _hull_area(best_points)
    if hull_area > 1e-12:
        scale = np.sqrt(1.0 / hull_area)
        final_points = best_points * scale
    else:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        final_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return final_points
