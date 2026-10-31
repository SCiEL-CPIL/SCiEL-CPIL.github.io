import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 142.0  # Safety margin before 150s limit

def _objective(v, lam_b, lam_o):
    """
    Objective function: Minimize -sum(r) + penalties.
    v: flattened array [x(26), y(26), r(26)]
    """
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:3*N]
    
    # Primary goal: maximize sum of radii
    obj = -r.sum()
    
    # Boundary penalties: circle must be inside [0,1]^2
    # r <= x, r <= 1-x, r <= y, r <= 1-y
    b_x = np.maximum(0, r - x) + np.maximum(0, r - (1.0 - x))
    b_y = np.maximum(0, r - y) + np.maximum(0, r - (1.0 - y))
    obj += lam_b * (b_x.sum() + b_y.sum())
    
    # Overlap penalties: distance(i,j) >= r_i + r_j
    # Vectorized computation of pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist2 = dx**2 + dy**2
    dist = np.sqrt(np.maximum(0, dist2))
    
    rsum = r[:, None] + r[None, :]
    
    # Overlap amount: max(0, r_i + r_j - dist_ij)
    # We square the overlap to make the penalty smoother and stronger for large violations
    overlap = np.maximum(0, rsum - dist)
    np.fill_diagonal(overlap, 0.0)
    
    obj += lam_o * (overlap**2).sum()
    
    return obj

def _force_directed_step(pos, radii, temp, k_repel=5.0, k_attract=0.1):
    """
    One step of force-directed relaxation.
    pos: (N, 2) array of positions
    radii: (N,) array of radii
    temp: temperature for random perturbation
    Returns new positions.
    """
    x, y = pos[:, 0].copy(), pos[:, 1].copy()
    forces = np.zeros((N, 2))
    
    # Repulsive forces between overlapping or close circles
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx**2 + dy**2)
            if d < 1e-10:
                dx, dy = 1e-10, 0.0
                d = 1e-10
            min_dist = radii[i] + radii[j]
            if d < min_dist * 1.5:  # Only interact if close
                # Repulsive force proportional to overlap and inverse distance
                overlap = max(0, min_dist - d)
                f = k_repel * overlap / d
                fx = f * dx / d
                fy = f * dy / d
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
    
    # Boundary repulsion
    for i in range(N):
        r = radii[i]
        # Left wall
        if x[i] < r:
            forces[i, 0] += k_repel * (r - x[i])
        # Right wall
        if x[i] > 1 - r:
            forces[i, 0] -= k_repel * (x[i] - (1 - r))
        # Bottom wall
        if y[i] < r:
            forces[i, 1] += k_repel * (r - y[i])
        # Top wall
        if y[i] > 1 - r:
            forces[i, 1] -= k_repel * (y[i] - (1 - r))
    
    # Attractive force toward center (to prevent drift)
    for i in range(N):
        forces[i, 0] -= k_attract * (x[i] - 0.5)
        forces[i, 1] -= k_attract * (y[i] - 0.5)
    
    # Apply forces with temperature-based random perturbation
    step_size = 0.01 + temp * 0.05
    new_x = x + step_size * forces[:, 0] + temp * np.random.randn(N) * 0.02
    new_y = y + step_size * forces[:, 1] + temp * np.random.randn(N) * 0.02
    
    # Clamp to [0, 1]
    new_x = np.clip(new_x, 0.0, 1.0)
    new_y = np.clip(new_y, 0.0, 1.0)
    
    return np.column_stack([new_x, new_y])

def _is_valid(arr, tol=1e-5):
    """Check if the arrangement is strictly valid."""
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    
    # Check radii non-negative
    if np.any(r < 0):
        return False
        
    # Check boundaries
    if np.any(x < -tol) or np.any(x > 1 + tol) or np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    if np.any(r > x + tol) or np.any(r > 1 - x + tol) or np.any(r > y + tol) or np.any(r > 1 - y + tol):
        return False
        
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    return True

def _make_grid_initial():
    """Grid initial guess: 6x6 grid, take first 26."""
    side = 6
    pts = []
    for i in range(side):
        for j in range(side):
            pts.append([(i + 0.5) / side, (j + 0.5) / side])
    pts = pts[:N]
    arr = np.array(pts)
    # Initial radius: half the cell size, slightly reduced
    r0 = (0.5 / side) * 0.85
    radii = np.full(N, r0)
    return np.column_stack([arr[:, 0], arr[:, 1], radii])

def _make_hex_initial():
    """Hexagonal-ish initial guess."""
    pts = []
    # 5 rows, 5 cols
    for row in range(5):
        for col in range(5):
            x = 0.1 + col * 0.2 + (0.1 if row % 2 == 1 else 0.0)
            y = 0.1 + row * 0.2
            pts.append([x, y])
    pts.append([0.5, 0.5]) # 26th point
    
    # Scale to fit [0,1] with margin
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    scale = 0.8 / max(xmax - xmin, ymax - ymin)
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    for p in pts:
        p[0] = 0.5 + (p[0] - cx) * scale
        p[1] = 0.5 + (p[1] - cy) * scale
    
    arr = np.array(pts)
    # Estimate radius from min distance
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((arr[i, 0] - arr[j, 0])**2 + (arr[i, 1] - arr[j, 1])**2)
            min_dist = min(min_dist, d)
    r0 = min_dist / 2 * 0.9
    for i in range(N):
        r0 = min(r0, arr[i, 0], 1 - arr[i, 0], arr[i, 1], 1 - arr[i, 1])
    radii = np.full(N, r0)
    return np.column_stack([arr[:, 0], arr[:, 1], radii])

def _make_random_initial(seed):
    """Random initial guess."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    
    # Calculate min distance to neighbors for radius estimation
    min_dists = np.full(N, 0.3)
    for i in range(N):
        for j in range(N):
            if i != j:
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                min_dists[i] = min(min_dists[i], d)
    
    r = 0.4 * min_dists
    # Bound by walls
    r = np.minimum(r, np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y)))
    r = np.maximum(r, 0.01)
    
    return np.column_stack([x, y, r])

def _repair(arr, max_iter=200):
    """
    Iteratively shrink radii to satisfy constraints if the optimizer failed.
    This ensures we return a valid solution even if optimization didn't converge perfectly.
    """
    x, y, r = arr[:, 0].copy(), arr[:, 1].copy(), arr[:, 2].copy()
    
    for _ in range(max_iter):
        # Check boundary violations
        max_viol = 0.0
        for i in range(N):
            d_wall = min(x[i], y[i], 1.0 - x[i], 1.0 - y[i])
            if r[i] > d_wall:
                max_viol = max(max_viol, r[i] - d_wall)
        
        # Check overlap violations
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if r[i] + r[j] > d:
                    max_viol = max(max_viol, r[i] + r[j] - d)
        
        if max_viol < 1e-9:
            break
            
        # Shrink all radii by the max violation amount (conservative)
        shrink = max_viol * 0.5
        r -= shrink
        r = np.maximum(r, 0.0)
        
    return np.column_stack([x, y, r])

def _optimize_single(init_arr, start_lam_b, start_lam_o, time_limit):
    """
    Run iterative penalty optimization starting from init_arr.
    """
    v0 = init_arr.flatten()
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
    
    best_val = -np.inf
    best_v = v0.copy()
    
    # Multi-stage: 
    # Stage 1: Low penalty to find rough packing
    # Stage 2: Medium penalty to refine
    # Stage 3: High penalty to enforce constraints strictly
    
    stages = [
        (start_lam_b, start_lam_o, 50),
        (start_lam_b * 5, start_lam_o * 5, 100),
        (start_lam_b * 20, start_lam_o * 20, 200),
        (start_lam_b * 100, start_lam_o * 100, 300),
    ]
    
    for lam_b_stage, lam_o_stage, max_iter in stages:
        if time.time() > time_limit:
            break
            
        try:
            res = minimize(
                _objective, v0,
                args=(lam_b_stage, lam_o_stage),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'maxls': 50}
            )
            v0 = res.x
        except Exception:
            break
        
        # Check validity and update best
        arr = v0.reshape(N, 3)
        if _is_valid(arr):
            val = arr[:, 2].sum()
            if val > best_val:
                best_val = val
                best_v = v0.copy()
        else:
            # Even if invalid, keep the position but note it's not the best valid yet
            # We might still want to continue optimizing from this point
            pass
            
    return best_v, best_val

def _force_directed_optimize(init_arr, time_limit, rng_seed=42):
    """
    Force-directed simulated annealing optimization.
    Returns (best_arr, best_val).
    """
    rng = np.random.RandomState(rng_seed)
    pos = init_arr[:, :2].copy()
    radii = init_arr[:, 2].copy()
    
    best_val = -np.inf
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    # Annealing schedule
    T_start = 1.0
    T_end = 0.001
    n_steps = 500
    cool_rate = (T_end / T_start) ** (1.0 / n_steps)
    
    start_time = time.time()
    for step in range(n_steps):
        if time.time() > time_limit:
            break
        
        T = T_start * (cool_rate ** step)
        
        # Try increasing radii slightly
        if step % 10 == 0:
            radii_new = radii * (1.0 + 0.01 * rng.rand(N))
            # Check if still feasible (rough check)
            feasible = True
            for i in range(N):
                if radii_new[i] > min(pos[i, 0], pos[i, 1], 1 - pos[i, 0], 1 - pos[i, 1]):
                    feasible = False
                    break
            if feasible:
                radii = radii_new
        
        # Force-directed step
        new_pos = _force_directed_step(pos, radii, T, k_repel=3.0, k_attract=0.05)
        
        # Check validity
        arr = np.column_stack([new_pos, radii])
        if _is_valid(arr):
            val = radii.sum()
            if val > best_val:
                best_val = val
                best_pos = new_pos.copy()
                best_radii = radii.copy()
        
        # Accept or reject based on Metropolis criterion (simplified: always accept if valid)
        pos = new_pos
    
    # Final polish with L-BFGS-B
    if best_val > -np.inf:
        v0 = np.column_stack([best_pos, best_radii]).flatten()
        bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
        try:
            res = minimize(
                _objective, v0,
                args=(500.0, 500.0),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14}
            )
            arr_new = res.x.reshape(N, 3)
            if _is_valid(arr_new):
                val_new = arr_new[:, 2].sum()
                if val_new > best_val:
                    best_val = val_new
                    best_pos = arr_new[:, :2].copy()
                    best_radii = arr_new[:, 2].copy()
        except Exception:
            pass
    
    best_arr = np.column_stack([best_pos, best_radii])
    return best_arr, best_val

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_arr = None
    best_val = -np.inf
    
    # Generate diverse initializations
    inits = []
    inits.append(_make_grid_initial())
    inits.append(_make_hex_initial())
    
    # Random inits with different seeds
    seeds = [42, 123, 456, 789, 1024, 2048, 3000, 4000, 5000, 6000]
    for s in seeds:
        inits.append(_make_random_initial(s))
    
    # Base lambda values
    base_lam_b = 10.0
    base_lam_o = 10.0
    
    # Phase 1: L-BFGS-B optimization on each initialization
    for init in inits:
        if time.time() > deadline - 30:
            break
            
        try:
            v_best, val = _optimize_single(init, base_lam_b, base_lam_o, deadline - 25)
        except Exception:
            continue
            
        if val > best_val:
            best_val = val
            best_arr = v_best.reshape(N, 3)
    
    # Phase 2: Force-directed simulated annealing on best and a few other inits
    if time.time() < deadline - 20:
        fd_inits = [best_arr] if best_arr is not None else []
        fd_inits.extend(inits[:3])  # Add a few more diverse starts
        
        for init in fd_inits:
            if time.time() > deadline - 15:
                break
            try:
                arr_fd, val_fd = _force_directed_optimize(init, deadline - 10, rng_seed=int(time.time() * 1000) % 10000)
                if val_fd > best_val:
                    best_val = val_fd
                    best_arr = arr_fd
            except Exception:
                continue
    
    # If no valid solution found from optimization, use a repaired version of the best attempt
    if best_arr is None:
        # Fallback: use grid and repair
        best_arr = _make_grid_initial()
        best_val = best_arr[:, 2].sum()
        
    # Final Polish: If the best solution is slightly invalid, try to repair it
    # Or, if it is valid, try one more high-precision polish step
    if _is_valid(best_arr):
        # Try to improve further with very high penalties on the valid solution
        if time.time() < deadline - 5:
            v0 = best_arr.flatten()
            bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
            try:
                res = minimize(
                    _objective, v0,
                    args=(2000.0, 2000.0),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                arr_new = res.x.reshape(N, 3)
                if _is_valid(arr_new):
                    val_new = arr_new[:, 2].sum()
                    if val_new > best_val:
                        best_val = val_new
                        best_arr = arr_new
            except Exception:
                pass
    else:
        # Repair the invalid solution
        best_arr = _repair(best_arr)
        if _is_valid(best_arr):
            best_val = best_arr[:, 2].sum()
        else:
            # Last resort: return a known valid grid
            best_arr = _make_grid_initial()
            best_val = best_arr[:, 2].sum()
            
    return best_arr
