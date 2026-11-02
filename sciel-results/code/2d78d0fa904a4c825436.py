"""Heilbronn 11-point problem: maximize minimum triangle area in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 11
S3H = np.sqrt(3) / 2
TRI_AREA = 0.5 * 1.0 * S3H  # area of the equilateral triangle with side 1

# Precompute all (i,j,k) triples
_TRIPLES = list(combinations(range(N), 3))
_N_TRIPLES = len(_TRIPLES)  # 165

# Convert triple indices to flat indices for vectorized area computation
_I_IDX = np.array([t[0] for t in _TRIPLES])
_J_IDX = np.array([t[1] for t in _TRIPLES])
_K_IDX = np.array([t[2] for t in _TRIPLES])


def _min_area(points):
    """Compute minimum triangle area among all triples. points: (N,2) array."""
    p_i = points[_I_IDX]
    p_j = points[_J_IDX]
    p_k = points[_K_IDX]
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    return np.min(areas)


def _all_areas(points):
    """Compute all triangle areas. points: (N,2) array."""
    p_i = points[_I_IDX]
    p_j = points[_J_IDX]
    p_k = points[_K_IDX]
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    return 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)


def _squared_min_area_objective(x, time_start):
    """
    Objective: maximize squared minimum area.
    x is 2N barycentric coords (u_i, v_i).
    Returns negative squared min area plus penalties.
    """
    u = x[0::2]
    v = x[1::2]
    
    # Penalty for being outside the triangle
    pen = 1000.0 * (
        np.sum(np.maximum(0, -u)**2) +
        np.sum(np.maximum(0, -v)**2) +
        np.sum(np.maximum(0, u + v - 1.0)**2)
    )
    
    # Convert to Cartesian
    pts = np.empty((N, 2))
    pts[:, 0] = u + 0.5 * v
    pts[:, 1] = S3H * v
    
    areas = _all_areas(pts)
    min_a = np.min(areas)
    
    # Maximize min_a^2, so minimize -min_a^2
    return -min_a**2 + pen


def _nudge_search(points, max_iter=50, step_size=0.01):
    """
    Targeted nudge search: perturb points in minimal triangles to break degeneracies.
    Returns improved points or original if no improvement.
    """
    best_pts = points.copy()
    best_score = _min_area(best_pts)
    rng = np.random.RandomState(42)
    
    for iteration in range(max_iter):
        areas = _all_areas(best_pts)
        min_idx = np.argmin(areas)
        i, j, k = _TRIPLES[min_idx]
        
        # Try nudging each point in the minimal triangle
        improved = False
        for idx in [i, j, k]:
            for _ in range(10):
                # Random small perturbation
                delta = rng.normal(0, step_size, 2)
                new_pts = best_pts.copy()
                new_pts[idx] += delta
                
                # Clamp to triangle
                for pi in range(N):
                    px, py = new_pts[pi]
                    if px < 0 or py < 0 or px > 1.0 or py > S3H:
                        # Project back
                        v_i = max(0, min(1, py / S3H))
                        u_i = max(0, min(1, px - 0.5 * v_i))
                        if u_i + v_i > 1.0:
                            s = u_i + v_i
                            u_i /= s
                            v_i /= s
                        new_pts[pi, 0] = u_i + 0.5 * v_i
                        new_pts[pi, 1] = S3H * v_i
                
                new_score = _min_area(new_pts)
                if new_score > best_score + 1e-12:
                    best_pts = new_pts
                    best_score = new_score
                    improved = True
                    break
        
        if not improved:
            # Try larger steps
            step_size *= 1.5
            if step_size > 0.1:
                break
    
    return best_pts


def construct():
    start_time = time.time()
    TIME_LIMIT = 140  # safety margin
    
    def _time_up():
        return (time.time() - start_time) > TIME_LIMIT
    
    best_points = None
    best_score = -float('inf')
    
    # Stage 1: Differential Evolution for global search
    bounds = [(0.0, 1.0)] * (2 * N)
    
    try:
        res_de = differential_evolution(
            _squared_min_area_objective,
            bounds,
            args=(start_time,),
            seed=42,
            popsize=20,
            maxiter=100,
            tol=1e-8,
            polish=False,
            init='sobol'
        )
        
        if _time_up():
            if res_de.x is not None:
                u = res_de.x[0::2]
                v = res_de.x[1::2]
                u = np.clip(u, 0, 1)
                v = np.clip(v, 0, 1)
                mask = (u + v) > 1.0
                if np.any(mask):
                    s = u[mask] + v[mask]
                    u[mask] /= s
                    v[mask] /= s
                pts = np.empty((N, 2))
                pts[:, 0] = u + 0.5 * v
                pts[:, 1] = S3H * v
                return pts
            return np.zeros((N, 2))
        
        best_x_de = res_de.x
    except Exception:
        best_x_de = np.random.RandomState(42).random(2 * N)
    
    # Convert DE result to points
    u = best_x_de[0::2]
    v = best_x_de[1::2]
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    mask = (u + v) > 1.0
    if np.any(mask):
        s = u[mask] + v[mask]
        u[mask] /= s
        v[mask] /= s
    de_points = np.empty((N, 2))
    de_points[:, 0] = u + 0.5 * v
    de_points[:, 1] = S3H * v
    
    # Stage 2: Local polish using L-BFGS-B on squared min area
    candidates = [de_points.copy()]
    
    # Add perturbed versions
    rng = np.random.RandomState(123)
    for _ in range(5):
        perturbed = de_points + rng.normal(0, 0.02, (N, 2))
        # Clamp to triangle
        for pi in range(N):
            px, py = perturbed[pi]
            if px < 0 or py < 0 or px > 1.0 or py > S3H:
                v_i = max(0, min(1, py / S3H))
                u_i = max(0, min(1, px - 0.5 * v_i))
                if u_i + v_i > 1.0:
                    s = u_i + v_i
                    u_i /= s
                    v_i /= s
                perturbed[pi, 0] = u_i + 0.5 * v_i
                perturbed[pi, 1] = S3H * v_i
        candidates.append(perturbed)
    
    # Add a few random starts
    for _ in range(3):
        pts_rand = np.random.RandomState(rng.randint(10000)).random((N, 2)) * np.array([1.0, S3H])
        # Clamp to triangle
        for pi in range(N):
            px, py = pts_rand[pi]
            if px < 0 or py < 0 or px > 1.0 or py > S3H:
                v_i = max(0, min(1, py / S3H))
                u_i = max(0, min(1, px - 0.5 * v_i))
                if u_i + v_i > 1.0:
                    s = u_i + v_i
                    u_i /= s
                    v_i /= s
                pts_rand[pi, 0] = u_i + 0.5 * v_i
                pts_rand[pi, 1] = S3H * v_i
        candidates.append(pts_rand)
    
    for cand_pts in candidates:
        if _time_up():
            break
        
        # Convert to barycentric for optimization
        u_init = np.zeros(N)
        v_init = np.zeros(N)
        for i in range(N):
            px, py = cand_pts[i]
            v_init[i] = py / S3H
            u_init[i] = px - 0.5 * v_init[i]
        
        x_init = np.column_stack([u_init, v_init]).flatten()
        
        # Local optimization
        bounds = [(0.0, 1.0)] * (2 * N)
        try:
            res = minimize(
                _squared_min_area_objective,
                x_init,
                args=(start_time,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if res.x is not None:
                u_opt = res.x[0::2]
                v_opt = res.x[1::2]
                u_opt = np.clip(u_opt, 0, 1)
                v_opt = np.clip(v_opt, 0, 1)
                mask = (u_opt + v_opt) > 1.0
                if np.any(mask):
                    s = u_opt[mask] + v_opt[mask]
                    u_opt[mask] /= s
                    v_opt[mask] /= s
                
                opt_pts = np.empty((N, 2))
                opt_pts[:, 0] = u_opt + 0.5 * v_opt
                opt_pts[:, 1] = S3H * v_opt
                
                # Verify inside triangle
                valid = True
                for i in range(N):
                    px, py = opt_pts[i]
                    v_i = py / S3H
                    u_i = px - 0.5 * v_i
                    if u_i < -1e-9 or v_i < -1e-9 or u_i + v_i > 1.0 + 1e-9:
                        valid = False
                        break
                
                if valid:
                    score = _min_area(opt_pts)
                    if score > best_score:
                        best_score = score
                        best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Stage 3: Nudge search to escape local minima
    if best_points is not None and not _time_up():
        nudged_pts = _nudge_search(best_points, max_iter=100, step_size=0.01)
        nudged_score = _min_area(nudged_pts)
        if nudged_score > best_score:
            best_score = nudged_score
            best_points = nudged_pts.copy()
        
        # Do another round of local optimization after nudging
        u_init = np.zeros(N)
        v_init = np.zeros(N)
        for i in range(N):
            px, py = best_points[i]
            v_init[i] = py / S3H
            u_init[i] = px - 0.5 * v_init[i]
        
        x_init = np.column_stack([u_init, v_init]).flatten()
        bounds = [(0.0, 1.0)] * (2 * N)
        try:
            res = minimize(
                _squared_min_area_objective,
                x_init,
                args=(start_time,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-11}
            )
            
            if res.x is not None:
                u_opt = res.x[0::2]
                v_opt = res.x[1::2]
                u_opt = np.clip(u_opt, 0, 1)
                v_opt = np.clip(v_opt, 0, 1)
                mask = (u_opt + v_opt) > 1.0
                if np.any(mask):
                    s = u_opt[mask] + v_opt[mask]
                    u_opt[mask] /= s
                    v_opt[mask] /= s
                
                opt_pts = np.empty((N, 2))
                opt_pts[:, 0] = u_opt + 0.5 * v_opt
                opt_pts[:, 1] = S3H * v_opt
                
                valid = True
                for i in range(N):
                    px, py = opt_pts[i]
                    v_i = py / S3H
                    u_i = px - 0.5 * v_i
                    if u_i < -1e-9 or v_i < -1e-9 or u_i + v_i > 1.0 + 1e-9:
                        valid = False
                        break
                
                if valid:
                    score = _min_area(opt_pts)
                    if score > best_score:
                        best_score = score
                        best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Stage 4: Final nudge + polish
    if best_points is not None and not _time_up():
        nudged_pts = _nudge_search(best_points, max_iter=50, step_size=0.005)
        nudged_score = _min_area(nudged_pts)
        if nudged_score > best_score:
            best_score = nudged_score
            best_points = nudged_pts.copy()
    
    if best_points is None:
        # Fallback: use DE result
        best_points = de_points
    
    return best_points
