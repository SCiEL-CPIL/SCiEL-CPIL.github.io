import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 26

def _compute_radius_sum(res):
    return np.sum(res[:, 2])

def _is_valid(res, tol=1e-5):
    """Check if the packing is valid."""
    if len(res) != N:
        return False
    if np.any(res[:, 2] <= 0):
        return False
    # Check boundaries
    if np.any(res[:, 0] - res[:, 2] < -tol):
        return False
    if np.any(res[:, 1] - res[:, 2] < -tol):
        return False
    if np.any(1 - res[:, 0] - res[:, 2] < -tol):
        return False
    if np.any(1 - res[:, 1] - res[:, 2] < -tol):
        return False
    
    # Check pairwise distances
    dx = res[:, 0][:, None] - res[:, 0][None, :]
    dy = res[:, 1][:, None] - res[:, 1][None, :]
    dist_sq = dx**2 + dy**2
    r_sum = res[:, 2][:, None] + res[:, 2][None, :]
    mask = ~np.eye(N, dtype=bool)
    if np.any(dist_sq[mask] < (r_sum**2)[mask] - tol):
        return False
    return True

def _fix_invalid(res, tol=1e-6):
    """Attempt to fix small violations by shrinking radii."""
    res = res.copy()
    # Shrink radii slightly
    res[:, 2] *= (1 - 1e-4)
    if not _is_valid(res, tol=1e-4):
        res[:, 2] *= 0.95
    # Final clamp to boundaries
    for i in range(N):
        max_r = min(res[i, 0], 1.0 - res[i, 0], res[i, 1], 1.0 - res[i, 1])
        if res[i, 2] > max_r:
            res[i, 2] = max_r * 0.999
    return res

def _soft(v, eps=1e-6):
    """Smooth approximation of max(0, v)."""
    return (v + np.sqrt(v * v + eps)) * 0.5

def _penalty_objective(x, deadline):
    """
    Objective function for full 78-dim optimization.
    x = [x1, y1, r1, x2, y2, r2, ...]
    Returns: -sum(r) + penalty
    """
    if time.time() > deadline:
        return 1e10
    
    res = x.reshape(N, 3)
    pos = res[:, :2]
    radii = res[:, 2]
    
    # Boundary violations
    left = _soft(radii - pos[:, 0])
    right = _soft(pos[:, 0] - (1.0 - radii))
    bot = _soft(radii - pos[:, 1])
    top = _soft(pos[:, 1] - (1.0 - radii))
    bpen = np.sum(left**2) + np.sum(right**2) + np.sum(bot**2) + np.sum(top**2)
    
    # Overlap violations
    dx = pos[:, 0][:, None] - pos[:, 0][None, :]
    dy = pos[:, 1][:, None] - pos[:, 1][None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = radii[:, None] + radii[None, :]
    viol = min_dist - dist
    np.fill_diagonal(viol, 0.0)
    viol_s = _soft(viol)
    np.fill_diagonal(viol_s, 0.0)
    open_pen = 0.5 * np.sum(viol_s**2)
    
    return -np.sum(radii) + 100.0 * (bpen + open_pen)

def _expand_struct(p):
    """
    Map a 10-dim structural vector to (pos, radii) of 26 circles.
    p = [corner_r, center_r, cx, cy, 
         ring1_off, ring1_r, ring1_r2, 
         ring2_off, ring2_r, ring2_r2]
    """
    corner_r = np.clip(p[0], 0.01, 0.30)
    center_r = np.clip(p[1], 0.01, 0.35)
    cx = np.clip(p[2], 0.2, 0.8)
    cy = np.clip(p[3], 0.2, 0.8)
    
    r1_off = np.clip(p[4], 0.1, 0.6)
    r1_r = np.clip(p[5], 0.01, 0.25)
    r1_r2 = np.clip(p[6], 0.01, 0.25)
    
    r2_off = np.clip(p[7], 0.2, 0.7)
    r2_r = np.clip(p[8], 0.01, 0.20)
    r2_r2 = np.clip(p[9], 0.01, 0.20)
    
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 4 corners
    pos[0] = [corner_r, corner_r]
    pos[1] = [1.0 - corner_r, corner_r]
    pos[2] = [corner_r, 1.0 - corner_r]
    pos[3] = [1.0 - corner_r, 1.0 - corner_r]
    radii[0:4] = corner_r
    
    # 1 center
    pos[4] = [cx, cy]
    radii[4] = center_r
    
    # 8 inner ring
    for i in range(8):
        ang = 2 * np.pi * i / 8.0
        pos[5 + i] = [cx + r1_off * np.cos(ang), cy + r1_off * np.sin(ang)]
        radii[5 + i] = r1_r if i % 2 == 0 else r1_r2
        
    # 13 outer ring
    for i in range(13):
        ang = 2 * np.pi * i / 13.0 + 0.1
        pos[13 + i] = [cx + r2_off * np.cos(ang), cy + r2_off * np.sin(ang)]
        radii[13 + i] = r2_r if i % 2 == 0 else r2_r2
        
    return pos, radii

def _struct_objective(p, deadline):
    """Objective for structural optimization."""
    pos, radii = _expand_struct(p)
    x = np.concatenate([pos.flatten(), radii])
    return _penalty_objective(x, deadline)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer
    
    best_res = None
    best_score = -1.0
    
    # Bounds for structural parameters
    struct_bounds = [
        (0.05, 0.25),  # corner_r
        (0.05, 0.30),  # center_r
        (0.3, 0.7),    # cx
        (0.3, 0.7),    # cy
        (0.15, 0.5),   # ring1_off
        (0.05, 0.2),   # ring1_r
        (0.05, 0.2),   # ring1_r2
        (0.3, 0.6),    # ring2_off
        (0.03, 0.15),  # ring2_r
        (0.03, 0.15),  # ring2_r2
    ]
    
    # Phase 1: Global Search with Differential Evolution
    candidates = []
    
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                lambda p: _struct_objective(p, deadline),
                bounds=struct_bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-5,
                polish=False,
                init='sobol'
            )
            if de_result.success:
                candidates.append(de_result.x)
                
            # Add a few random structural seeds
            rng = np.random.RandomState(42)
            for _ in range(5):
                p_rand = np.array([rng.uniform(b[0], b[1]) for b in struct_bounds])
                candidates.append(p_rand)
                
        except Exception as e:
            print(f"DE failed: {e}")
    
    # Phase 2: Local Refinement for each candidate
    for p_struct in candidates:
        if time.time() > deadline - 30:
            break
            
        pos, radii = _expand_struct(p_struct)
        x0 = np.concatenate([pos.flatten(), radii])
        
        # Bounds for L-BFGS-B
        bounds = []
        for i in range(N):
            # x, y can be anywhere, but radius constrained by position
            # We use loose bounds for x,y and tight for r
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
            bounds.append((1e-5, 0.4))
            
        try:
            res_opt = minimize(
                _penalty_objective,
                x0,
                args=(deadline,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            res_arr = res_opt.x.reshape(N, 3)
            
            # Post-process to ensure validity
            if _is_valid(res_arr):
                score = _compute_radius_sum(res_arr)
                if score > best_score:
                    best_score = score
                    best_res = res_arr
            else:
                res_fixed = _fix_invalid(res_arr)
                if _is_valid(res_fixed):
                    score = _compute_radius_sum(res_fixed)
                    if score > best_score:
                        best_score = score
                        best_res = res_fixed
                        
        except Exception as e:
            pass

    # Phase 3: Final Polish on the best result
    if best_res is not None and time.time() < deadline - 5:
        x0 = best_res.flatten()
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
            bounds.append((1e-5, 0.4))
            
        try:
            res_opt = minimize(
                _penalty_objective,
                x0,
                args=(deadline,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            res_arr = res_opt.x.reshape(N, 3)
            if _is_valid(res_arr):
                score = _compute_radius_sum(res_arr)
                if score > best_score:
                    best_score = score
                    best_res = res_arr
        except Exception:
            pass

    # Fallback if no valid result found
    if best_res is None:
        res = np.zeros((N, 3))
        side = 5
        for i in range(N):
            cx, cy = (i % 5 + 0.5) / 5.0, (i // 5 + 0.5) / 5.0
            res[i, 0] = cx
            res[i, 1] = cy
            res[i, 2] = 0.08
        best_res = _fix_invalid(res)

    return best_res
