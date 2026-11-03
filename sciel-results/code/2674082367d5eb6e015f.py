"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

def _validate(xyz):
    """Check if configuration is valid: circles inside unit square and non-overlapping."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -1e-7) or np.any(x + r > 1 + 1e-7):
        return False
    if np.any(y - r < -1e-7) or np.any(y + r > 1 + 1e-7):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-7:
                return False
    return True

def _get_delaunay_edges(pts):
    """Get Delaunay triangulation edges for proximity-based constraint reduction."""
    try:
        delaunay = Delaunay(pts)
        edges = set()
        for simplex in delaunay.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
        return list(edges)
    except Exception:
        # Fallback to all pairs
        n = len(pts)
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _physics_relax(xyz, iterations=2000):
    """
    Relax the configuration to remove overlaps using a simple force-directed method.
    This ensures a valid starting point for the optimizer.
    """
    xyz = xyz.copy()
    for _ in range(iterations):
        moved = False
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dx = xyz[i, 0] - xyz[j, 0]
                dy = xyz[i, 1] - xyz[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = xyz[i, 2] + xyz[j, 2]
                if dist < min_dist:
                    if dist < 1e-9:
                        dx, dy = 1e-5, 1e-5
                        dist = np.sqrt(dx * dx + dy * dy)
                    # Push apart
                    overlap = min_dist - dist
                    push_x = (dx / dist) * overlap * 0.5
                    push_y = (dy / dist) * overlap * 0.5
                    xyz[i, 0] += push_x
                    xyz[i, 1] += push_y
                    xyz[j, 0] -= push_x
                    xyz[j, 1] -= push_y
                    moved = True
        
        # Wall constraints
        for i in range(N):
            x, y, r = xyz[i]
            if x < r:
                xyz[i, 0] = r
                moved = True
            elif x > 1 - r:
                xyz[i, 0] = 1 - r
                moved = True
            if y < r:
                xyz[i, 1] = r
                moved = True
            elif y > 1 - r:
                xyz[i, 1] = 1 - r
                moved = True
        
        if not moved:
            break
            
    return xyz

def _objective_full(params, penalty_weight=1000.0):
    """
    Full pairwise penalty objective for final polish.
    Minimize: -sum(r) + penalty_weight * (wall_violations + overlap_violations)
    """
    n = N
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Wall violations
    wx_left = np.maximum(0.0, r - x)
    wx_right = np.maximum(0.0, r - (1.0 - x))
    wy_bottom = np.maximum(0.0, r - y)
    wy_top = np.maximum(0.0, r - (1.0 - y))
    wall_penalty = np.sum(wx_left**2 + wx_right**2 + wy_bottom**2 + wy_top**2)
    
    # All pairwise overlaps
    iu = np.triu_indices(n, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    overlap = r[iu[0]] + r[iu[1]] - dist
    active = overlap > 0
    overlap_penalty = np.sum(overlap[active] ** 2)
    
    # Negative radius penalty
    neg_r = np.maximum(0.0, -r)
    neg_penalty = np.sum(neg_r**2)
    
    return -np.sum(r) + penalty_weight * (wall_penalty + overlap_penalty + neg_penalty)

def _local_refine(xyz, time_budget=5.0):
    """
    Refine a configuration using L-BFGS-B with Delaunay-based constraints for speed,
    then full constraints for precision.
    """
    start_time = time.time()
    params = xyz.flatten()
    best_params = params.copy()
    best_obj = _objective_full(best_params, 100.0)
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((1e-6, 0.5))  # r
    
    # Phase 1: Coarse optimization with Delaunay edges (faster)
    if time.time() - start_time < time_budget * 0.5:
        def obj_delaunay(p):
            n = N
            x = p[0::3]
            y = p[1::3]
            r = p[2::3]
            pts = np.column_stack([x, y])
            edges = _get_delaunay_edges(pts)
            
            wall_penalty = 0.0
            for i in range(n):
                wall_penalty += np.maximum(0.0, r[i] - x[i])**2
                wall_penalty += np.maximum(0.0, r[i] - (1.0 - x[i]))**2
                wall_penalty += np.maximum(0.0, r[i] - y[i])**2
                wall_penalty += np.maximum(0.0, r[i] - (1.0 - y[i]))**2
            
            overlap_penalty = 0.0
            for i, j in edges:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    overlap_penalty += overlap ** 2
            
            neg_penalty = np.sum(np.maximum(0.0, -r)**2)
            return -np.sum(r) + 100.0 * (wall_penalty + overlap_penalty + neg_penalty)
        
        try:
            result = minimize(obj_delaunay, best_params, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if _objective_full(result.x, 100.0) < best_obj:
                best_params = result.x
                best_obj = _objective_full(best_params, 100.0)
        except Exception:
            pass

    # Phase 2: Fine optimization with full pairwise constraints
    for penalty_weight in [100.0, 500.0, 1000.0]:
        if time.time() - start_time > time_budget:
            break
        
        def obj_full(p):
            return _objective_full(p, penalty_weight)
        
        try:
            result = minimize(obj_full, best_params, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            obj_val = _objective_full(result.x, penalty_weight)
            if obj_val < best_obj:
                best_params = result.x
                best_obj = obj_val
        except Exception:
            pass
    
    return best_params.reshape(N, 3)

def _enforce_validity(params):
    """
    Enforce all constraints by shrinking radii and clipping positions.
    Returns a valid configuration.
    """
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)
    
    # Clip radii to fit within walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    
    # Iteratively resolve overlaps
    for _ in range(200):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 0
        if not np.any(active):
            break
        max_ov = 0.0
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            max_ov = max(max_ov, ov)
            # Shrink the smaller circle preferentially
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
        # Re-clip to walls
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    
    return p

def _initial_config_grid():
    """Simple grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.8
            cells.append([x, y, r])
    return np.array(cells[:N])

def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of radii.
    """
    global_start = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # 1. Generate initial configuration
    xyz0 = _initial_config_grid()
    
    # 2. Physics relaxation to ensure validity and good starting point
    xyz_relaxed = _physics_relax(xyz0, iterations=5000)
    
    # 3. Local refinement
    remaining_time = deadline - (time.time() - global_start)
    if remaining_time > 5.0:
        xyz_opt = _local_refine(xyz_relaxed, time_budget=remaining_time * 0.7)
    else:
        xyz_opt = xyz_relaxed
    
    # 4. Enforce validity
    xyz_fixed = _enforce_validity(xyz_opt.flatten())
    
    # 5. Final check
    if not _validate(xyz_fixed):
        # If still invalid, try a more aggressive fix
        xyz_fixed = _initial_config_grid()
        xyz_fixed = _physics_relax(xyz_fixed, iterations=10000)
        xyz_fixed = _enforce_validity(xyz_fixed)
    
    return xyz_fixed

# EVOLVE-BLOCK-END
