"""Circle packing in unit square - Simultaneous center+radius optimization."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import random


def _eval_packing(xy, r):
    """Evaluate a packing: returns (total_radius, is_valid)."""
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return np.sum(r), True


def _smooth_objective(p, n):
    """
    Smooth objective: minimize -sum(r) + penalty * sum(violations^2).
    p is a flat array of length 3n: [x0,y0,r0, x1,y1,r1, ...]
    """
    xy = p[0::3].reshape(-1, 2)
    r = p[1::3]
    
    # Ensure radii are positive
    r_pos = np.maximum(r, 0.001)
    
    penalty = 0.0
    
    # Wall violations
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r_pos[i]
        penalty += max(0.0, ri - x) ** 2
        penalty += max(0.0, ri - (1.0 - x)) ** 2
        penalty += max(0.0, ri - y) ** 2
        penalty += max(0.0, ri - (1.0 - y)) ** 2
    
    # Overlap violations
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_d = r_pos[i] + r_pos[j]
            if dist < min_d:
                penalty += (min_d - dist) ** 2
    
    return -np.sum(r_pos) + 1000.0 * penalty


def _smooth_gradient(p, n):
    """Numerical gradient of the smooth objective."""
    eps = 1e-6
    grad = np.zeros_like(p)
    base = _smooth_objective(p, n)
    for k in range(len(p)):
        p_plus = p.copy()
        p_plus[k] += eps
        grad[k] = (_smooth_objective(p_plus, n) - base) / eps
    return grad


def _init_grid(side):
    """Create grid initialization."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                positions.append([x, y])
    while len(positions) < N:
        positions.append([0.5, 0.5])
    return np.array(positions[:N])


def _init_random():
    """Random initialization."""
    positions = []
    for i in range(N):
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        positions.append([x, y])
    return np.array(positions[:N])


def _init_repulsion_relax(xy, iterations=200):
    """
    Repulsion relaxation: push circles apart to create better initial structure.
    """
    n = len(xy)
    xy = xy.copy()
    r = np.full(n, 0.03)
    
    for it in range(iterations):
        forces = np.zeros_like(xy)
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-6:
                    dx, dy = random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01)
                    dist = np.sqrt(dx * dx + dy * dy)
                # Repulsion force
                f = 1.0 / (dist * dist + 1e-4)
                forces[i, 0] += f * dx / dist
                forces[i, 1] += f * dy / dist
                forces[j, 0] -= f * dx / dist
                forces[j, 1] -= f * dy / dist
            # Wall attraction (keep inside)
            forces[i, 0] += 0.1 * (xy[i, 0] - 0.5)
            forces[i, 1] += 0.1 * (xy[i, 1] - 0.5)
        
        # Update positions
        step = 0.001 * (1.0 - it / iterations)
        xy += step * forces
        xy[:, 0] = np.clip(xy[:, 0], 0.05, 0.95)
        xy[:, 1] = np.clip(xy[:, 1], 0.05, 0.95)
    
    return xy


def _l_bfgs_optimize(xy_init, r_init, deadline, max_iter=500):
    """
    Run L-BFGS-B optimization on all 3n variables simultaneously.
    """
    n = len(xy_init)
    p0 = np.zeros(3 * n)
    p0[0::3] = xy_init[:, 0]
    p0[1::3] = xy_init[:, 1]
    p0[2::3] = r_init
    
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))  # x
        bounds.append((0.001, 0.999))  # y
        bounds.append((0.001, 0.45))   # r
    
    best_total = 0.0
    best_valid = False
    best_p = p0.copy()
    
    # Evaluate initial
    xy_e = p0[0::3].reshape(-1, 2)
    r_e = p0[2::3]
    total, valid = _eval_packing(xy_e, r_e)
    if valid:
        best_total = total
        best_valid = True
    
    try:
        result = minimize(
            _smooth_objective,
            p0,
            args=(n,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfun': 10000, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        opt_p = result.x
        opt_xy = opt_p[0::3].reshape(-1, 2)
        opt_r = np.maximum(opt_p[2::3], 0.001)
        
        total, valid = _eval_packing(opt_xy, opt_r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_p = opt_p.copy()
    except Exception:
        pass
    
    final_xy = best_p[0::3].reshape(-1, 2)
    final_r = np.maximum(best_p[2::3], 0.001)
    return final_xy, final_r, best_total, best_valid


def _local_search_joint(xy, r, deadline, grid_size=60):
    """
    Joint local search: for each circle, try moving to nearby grid positions
    and adjusting radius to maximize its own radius while maintaining validity.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = np.sum(best_r)
    else:
        best_total = 0.0
    
    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)
    
    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))
    
    max_sweeps = 30
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]
            
            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()
            
            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]
            
            old_r = r[i]
            if new_r > old_r + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True
        
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()
        
        if not improved:
            break
    
    return best_xy, best_r, best_total, best_valid


def _fine_tune(xy, r, deadline):
    """
    Fine-tune using small perturbations and L-BFGS from multiple nearby starts.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = np.sum(best_r)
    else:
        best_total = 0.0
    
    # Try small perturbations
    for attempt in range(20):
        if time.time() > deadline:
            break
        perturbed_xy = best_xy + random.gauss(0, 0.005, (n, 2))
        perturbed_xy[:, 0] = np.clip(perturbed_xy[:, 0], 0.01, 0.99)
        perturbed_xy[:, 1] = np.clip(perturbed_xy[:, 1], 0.01, 0.99)
        perturbed_r = best_r + random.gauss(0, 0.002, n)
        perturbed_r = np.clip(perturbed_r, 0.005, 0.4)
        
        out_xy, out_r, total, valid = _l_bfgs_optimize(
            perturbed_xy, perturbed_r, deadline, max_iter=200
        )
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r
    
    return best_xy, best_r, best_total, best_valid


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    # Multiple diverse initializations
    inits = []
    
    # Grid inits
    for side in [5, 6]:
        inits.append(('grid', _init_grid(side)))
    
    # Random inits with repulsion relaxation
    for _ in range(3):
        if time.time() > deadline - 60.0:
            break
        xy_rand = _init_random()
        xy_relaxed = _init_repulsion_relax(xy_rand, iterations=100)
        inits.append(('relaxed', xy_relaxed))
    
    # Run L-BFGS from each initialization
    for init_type, xy_init in inits:
        if time.time() > deadline - 30.0:
            break
        
        # Initialize radii
        if init_type == 'grid':
            side = int(np.sqrt(N))
            if side * side < N:
                side += 1
            r_init = np.full(N, 0.4 / side)
        else:
            r_init = np.full(N, 0.03)
        
        out_xy, out_r, total, valid = _l_bfgs_optimize(
            xy_init, r_init, deadline - 30.0, max_iter=400
        )
        
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r
    
    # Local search polish on best result
    if best_xy is not None and best_valid:
        if time.time() < deadline - 20.0:
            p_xy, p_r, p_total, p_valid = _local_search_joint(
                best_xy, best_r, deadline - 20.0, grid_size=80
            )
            if p_valid and p_total > best_total:
                best_total = p_total
                best_valid = True
                best_xy = p_xy
                best_r = p_r
        
        # Fine-tune
        if time.time() < deadline - 10.0:
            f_xy, f_r, f_total, f_valid = _fine_tune(
                best_xy, best_r, deadline - 10.0
            )
            if f_valid and f_total > best_total:
                best_total = f_total
                best_valid = True
                best_xy = f_xy
                best_r = f_r
        
        # Final local search
        if time.time() < deadline - 5.0:
            p_xy, p_r, p_total, p_valid = _local_search_joint(
                best_xy, best_r, deadline - 5.0, grid_size=100
            )
            if p_valid and p_total > best_total:
                best_total = p_total
                best_valid = True
                best_xy = p_xy
                best_r = p_r
    
    # Fallback if no valid result
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy_init = _init_grid(side)
        r = np.full(N, 0.35 / side)
        total, valid = _eval_packing(xy_init, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy_init
            best_r = r
        else:
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.3 / side
            return result
    
    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
    
    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.3 / side
    
    return result
# EVOLVE-BLOCK-END
