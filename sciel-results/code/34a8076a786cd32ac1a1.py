"""Circle packing in unit square - 26 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time
import random

N = 26
TOTAL_TIME_BUDGET = 145.0  # Safety margin before 150s kill


def _is_valid(xy, r, tol=1e-9):
    """Check if packing is valid (all circles inside unit square, no overlaps)."""
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -tol or x - ri < -tol or 1.0 - x - ri < -tol or y - ri < -tol or 1.0 - y - ri < -tol:
            return False, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False, 0.0
    return True, np.sum(r)


def _max_radius_at(pos, other_xy, other_r, walls=True):
    """Compute max radius for a circle at pos given others."""
    max_r = np.inf
    if walls:
        max_r = min(max_r, pos[0], 1.0 - pos[0], pos[1], 1.0 - pos[1])
    if len(other_xy) > 0:
        dx = pos[0] - other_xy[:, 0]
        dy = pos[1] - other_xy[:, 1]
        dists = np.sqrt(dx * dx + dy * dy)
        max_r = min(max_r, np.min(dists - other_r))
    return max(0.0, max_r)


def _repulsion_relax(xy, r, n_iter=200, learning_rate=0.01):
    """
    Relax circles to reduce overlaps using repulsion forces.
    This helps escape local minima in the local search.
    """
    n = len(xy)
    for _ in range(n_iter):
        forces = np.zeros((n, 2))
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy) + 1e-10
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    # Repulsion proportional to overlap
                    fx = overlap * dx / dist
                    fy = overlap * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        # Apply forces with momentum-like damping
        for i in range(n):
            xy[i, 0] = np.clip(xy[i, 0] + learning_rate * forces[i, 0], 0.01, 0.99)
            xy[i, 1] = np.clip(xy[i, 1] + learning_rate * forces[i, 1], 0.01, 0.99)
        # Recompute radii after moving
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            r[i] = _max_radius_at(xy[i], xy[mask], r[mask])
    return xy, r


def _local_search(xy, r, deadline, grid_size=100):
    """
    Iterative local search: for each circle, find best position on a grid
    that maximizes its radius given all others fixed.
    """
    n = len(xy)
    best_total = np.sum(r)
    best_xy = xy.copy()
    best_r = r.copy()
    
    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)
    
    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))
    
    max_sweeps = 60
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        # Randomize order for better convergence
        order = np.random.permutation(n)
        for i in order:
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
            new_r = r_candidates[idx]
            
            if new_r > r[i] + 1e-12:
                xy[i, 0] = candidates[idx, 0]
                xy[i, 1] = candidates[idx, 1]
                r[i] = new_r
                improved = True
        
        total = np.sum(r)
        if total > best_total + 1e-12:
            best_total = total
            best_xy = xy.copy()
            best_r = r.copy()
        
        if not improved:
            break
    
    return best_xy, best_r, best_total


def _gradient_optimize(xy, r, deadline):
    """
    Use L-BFGS-B with a smooth objective to polish the packing.
    Objective: maximize sum(r) - penalty * soft_violations
    """
    n = len(xy)
    
    def objective(params):
        xy_p = params[0::3].reshape(-1, 2)
        r_p = np.maximum(params[2::3], 0.001)
        
        # Wall violations (soft)
        wall_viol = 0.0
        for i in range(n):
            c = np.array([xy_p[i, 0] - r_p[i],
                          1.0 - xy_p[i, 0] - r_p[i],
                          xy_p[i, 1] - r_p[i],
                          1.0 - xy_p[i, 1] - r_p[i]])
            wall_viol += np.sum(np.maximum(-c, 0.0) ** 2)
        
        # Overlap violations (soft)
        overlap_viol = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy_p[i, 0] - xy_p[j, 0]
                dy = xy_p[i, 1] - xy_p[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                c = dist - r_p[i] - r_p[j]
                overlap_viol += np.maximum(-c, 0.0) ** 2
        
        penalty_weight = 500.0
        return -np.sum(r_p) + penalty_weight * (wall_viol + overlap_viol)
    
    params = np.zeros(3 * n)
    params[0::3] = xy[:, 0]
    params[1::3] = xy[:, 1]
    params[2::3] = r
    
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.45))
    
    best_total = np.sum(r)
    best_params = params.copy()
    
    try:
        result = minimize(objective, params, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-10})
        opt_params = result.x
        opt_xy = opt_params[0::3].reshape(-1, 2)
        opt_r = np.maximum(opt_params[2::3], 0.001)
        
        valid, total = _is_valid(opt_xy, opt_r)
        if valid and total > best_total:
            best_total = total
            best_params = opt_params.copy()
    except Exception:
        pass
    
    final_xy = best_params[0::3].reshape(-1, 2)
    final_r = np.maximum(best_params[2::3], 0.001)
    return final_xy, final_r, best_total


def _grow_radii(xy, r, deadline):
    """
    Try to increase radii iteratively while maintaining validity.
    Uses a greedy approach: increase each radius slightly if it remains valid.
    """
    n = len(xy)
    best_total = np.sum(r)
    best_r = r.copy()
    
    step = 0.001
    for iteration in range(2000):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            old_r = r[i]
            r[i] = old_r + step
            valid, total = _is_valid(xy, r)
            if valid:
                improved = True
                if total > best_total + 1e-12:
                    best_total = total
                    best_r = r.copy()
            else:
                r[i] = old_r
        if not improved:
            step *= 0.5
            if step < 1e-7:
                break
    
    return xy, best_r, best_total


def _random_init_with_repulsion(n_circles=N):
    """
    Initialize circles randomly, then apply repulsion relaxation
    to get a better starting configuration.
    """
    r = np.full(n_circles, 0.02)
    xy = np.random.uniform(0.05, 0.95, size=(n_circles, 2))
    # Apply repulsion to spread out
    xy, r = _repulsion_relax(xy, r, n_iter=100, learning_rate=0.02)
    return xy, r


def _hex_init():
    """Hexagonal packing initialization."""
    positions = []
    r = 0.035
    rows = 0
    y = r + 0.01
    row = 0
    while len(positions) < N:
        x = r + 0.01
        col = 0
        while len(positions) < N:
            if row % 2 == 0:
                px = x
            else:
                px = x + r * np.sqrt(3) / 2
            if px + r <= 0.99:
                positions.append([px, y, r])
            x += r * np.sqrt(3) + 0.005
            col += 1
        y += r * 1.5 + 0.003
        row += 1
    return np.array(positions[:N])


def _grid_init(side=6):
    """Grid initialization."""
    positions = []
    r = 0.5 / side - 0.005
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def _random_init():
    """Simple random initialization."""
    r = np.full(N, 0.015)
    xy = np.random.uniform(0.05, 0.95, size=(N, 2))
    return xy, r


def construct():
    start_time = time.time()
    deadline = start_time + TOTAL_TIME_BUDGET
    
    best_xy = None
    best_r = None
    best_total = 0.0
    
    # Multiple initialization strategies
    initializations = [
        ('hex', _hex_init),
        ('grid6', lambda: _grid_init(6)),
        ('grid5', lambda: _grid_init(5)),
        ('random_repulsion', lambda: _random_init_with_repulsion()),
        ('random', _random_init),
    ]
    
    for name, init_func in initializations:
        if time.time() > deadline - 25.0:
            break
        
        try:
            if isinstance(init_func, tuple):
                xy, r = init_func
            else:
                xy, r = init_func()
            
            # Local search
            if time.time() < deadline - 15.0:
                out_xy, out_r, total = _local_search(xy, r, deadline - 15.0, grid_size=90)
                if total > best_total:
                    best_total = total
                    best_xy = out_xy.copy()
                    best_r = out_r.copy()
        except Exception:
            continue
    
    # If no valid result, use fallback
    if best_xy is None:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result
    
    # Gradient polish
    if time.time() < deadline - 10.0:
        try:
            g_xy, g_r, g_total = _gradient_optimize(best_xy, best_r, deadline - 10.0)
            if g_total > best_total:
                best_total = g_total
                best_xy = g_xy.copy()
                best_r = g_r.copy()
        except Exception:
            pass
    
    # Grow radii
    if time.time() < deadline - 5.0:
        try:
            gr_xy, gr_r, gr_total = _grow_radii(best_xy, best_r, deadline - 5.0)
            if gr_total > best_total:
                best_total = gr_total
                best_xy = gr_xy.copy()
                best_r = gr_r.copy()
        except Exception:
            pass
    
    # Final local search polish
    if time.time() < deadline - 3.0:
        try:
            out_xy, out_r, total = _local_search(best_xy, best_r, deadline - 3.0, grid_size=120)
            if total > best_total:
                best_total = total
                best_xy = out_xy.copy()
                best_r = out_r.copy()
        except Exception:
            pass
    
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
    valid, total = _is_valid(final_xy, final_r)
    if not valid:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
    
    return result
