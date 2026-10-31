"""
Circle packing in perimeter-4 rectangle.
Combines SLSQP optimization (variable width) with force-directed relaxation and local refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _time_ok(deadline, margin=1.0):
    return time.time() < deadline - margin

def _compute_distances(x, y):
    """Vectorized pairwise distance matrix."""
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    return np.sqrt(dx * dx + dy * dy)

def _objective_slsqp(z):
    """
    Objective function for SLSQP.
    z: array of shape (3*N + 1)
    z[0:N] = x coords
    z[N:2N] = y coords
    z[2N:3N] = radii
    z[3N] = width w
    Height h = 2.0 - w
    
    We want to maximize sum(r), so minimize -sum(r).
    Add penalties for overlaps and boundary violations.
    """
    x = z[0:N]
    y = z[N:2*N]
    r = z[2*N:3*N]
    w = z[3*N]
    h = 2.0 - w

    # Basic validity check to avoid huge penalties or NaNs
    if w < 0.3 or h < 0.3 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)
    
    # Penalty weight
    pw = 1000.0

    # 1. Boundary Penalties
    # x < r, x > w - r, y < r, y > h - r
    bl_x = np.maximum(0, r - x)
    br_x = np.maximum(0, x - (w - r))
    bl_y = np.maximum(0, r - y)
    br_y = np.maximum(0, y - (h - r))
    obj += pw * (np.sum(bl_x**2) + np.sum(br_x**2) + np.sum(bl_y**2) + np.sum(br_y**2))

    # 2. Overlap Penalties
    d = _compute_distances(x, y)
    r_sum = r[:, None] + r[None, :]
    # Only consider upper triangle to avoid double counting
    triu_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap = np.maximum(0, r_sum - d)
    obj += pw * np.sum(overlap[triu_mask]**2)

    # 3. Aspect ratio penalty (keep w in reasonable range [0.5, 1.5])
    if w < 0.5:
        obj += pw * (0.5 - w)**2
    elif w > 1.5:
        obj += pw * (w - 1.5)**2

    return obj

def _force_relax(z, max_iter=200, dt=0.005):
    """
    Force-directed relaxation to resolve overlaps and boundary violations.
    This makes the solution "feasible" for the final evaluation.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0:n].copy()
    y = z[n:2*n].copy()
    r = z[2*n:3*n].copy()
    w = z[3*n]
    h = 2.0 - w

    for _ in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall forces
        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 20
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 20
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 20
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 20

        # Pairwise repulsion
        d = _compute_distances(x, y)
        r_sum = r[:, None] + r[None, :]
        for i in range(n):
            for j in range(i + 1, n):
                if d[i, j] < r_sum[i, j] and d[i, j] > 1e-10:
                    overlap = r_sum[i, j] - d[i, j]
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    force_mag = overlap * 10
                    fx[i] += (dx / d[i, j]) * force_mag
                    fy[i] += (dy / d[i, j]) * force_mag
                    fx[j] -= (dx / d[i, j]) * force_mag
                    fy[j] -= (dy / d[i, j]) * force_mag

        x += dt * fx
        y += dt * fy

        # Clamp to boundaries (hard constraint)
        for i in range(n):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])

        # Decay step size
        dt *= 0.98

    z[0:n] = x
    z[n:2*n] = y
    return z

def _local_refine_radii(z, max_iter=50):
    """
    Iteratively increase radii as much as possible given fixed positions.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0:n].copy()
    y = z[n:2*n].copy()
    r = z[2*n:3*n].copy()
    w = z[3*n]
    h = 2.0 - w

    for _ in range(max_iter):
        # Calculate max possible radius for each circle
        new_r = np.zeros(n)
        for i in range(n):
            # Distance to boundaries
            dist_bounds = min(x[i] - r[i], w - x[i] - r[i], y[i] - r[i], h - y[i] - r[i])
            
            # Distance to other circles
            min_dist = dist_bounds
            for j in range(n):
                if i != j:
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    # Max radius i can have while staying away from j
                    # r_i <= d - r_j
                    # But we are growing r_i, so we check if current r_i can grow
                    # Actually, we want to find the max r_i such that r_i + r_j <= d
                    # So max r_i = d - r_j
                    min_dist = min(min_dist, d - r[j])
            
            new_r[i] = max(0.01, min_dist)
        
        # Update radii with damping to ensure stability
        r = 0.8 * r + 0.2 * new_r
    
    z[2*n:3*n] = r
    return z

def _check_validity(z, tol=1e-9):
    """Check if packing is valid, return (valid, value)."""
    n = N
    x = z[0:n]
    y = z[n:2*n]
    r = z[2*n:3*n]
    w = z[3*n]
    h = 2.0 - w

    if w <= 0.05 or h <= 0.05:
        return False, 0.0

    for i in range(n):
        if r[i] <= 0:
            return False, 0.0
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False, 0.0
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False, 0.0

    d = _compute_distances(x, y)
    r_sum = r[:, None] + r[None, :]
    triu_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(d[triu_mask] < r_sum[triu_mask] - tol):
        return False, 0.0

    val = float(np.sum(r))
    return True, val

def _run_single_optimization(init_z, deadline):
    """
    Run SLSQP, then force relax, then local refine.
    Returns best valid solution found in this path.
    """
    if not _time_ok(deadline, 2.0):
        return None

    # 1. SLSQP Optimization
    try:
        res = minimize(
            _objective_slsqp,
            init_z,
            method='SLSQP',
            options={'maxiter': 100, 'ftol': 1e-9}
        )
        z_opt = res.x
    except Exception:
        z_opt = init_z

    if not _time_ok(deadline, 2.0):
        return None

    # 2. Force Relaxation (make it feasible)
    z_relaxed = _force_relax(z_opt, max_iter=100)
    
    # 3. Local Refinement (grow radii)
    z_refined = _local_refine_radii(z_relaxed, max_iter=30)
    
    # 4. Final Force Relaxation (in case refinement caused overlaps)
    z_final = _force_relax(z_refined, max_iter=50)
    
    # 5. Check Validity
    valid, value = _check_validity(z_final)
    if valid:
        return z_final, value
    else:
        # Try to salvage by just returning the relaxed version if it's valid
        valid_rel, val_rel = _check_validity(z_relaxed)
        if valid_rel:
            return z_relaxed, val_rel
        return None

def construct():
    deadline = time.time() + 140.0
    best_value = 0.0
    best_z = None

    # Initializations
    inits = []
    
    # 1. Grid initialization in 1x1 square (w=1)
    side = int(np.ceil(np.sqrt(N)))
    xs = (np.arange(side) + 0.5) / side
    ys = (np.arange(side) + 0.5) / side
    grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij')
    init_pos = np.column_stack((grid_x.ravel()[:N], grid_y.ravel()[:N]))
    init_r = np.full(N, 0.1)
    init_w = 1.0
    z_init = np.concatenate([init_pos[:, 0], init_pos[:, 1], init_r, [init_w]])
    inits.append(z_init)

    # 2. Random initialization
    rng = np.random.default_rng(42)
    for _ in range(3):
        w_init = rng.uniform(0.7, 1.3)
        h_init = 2.0 - w_init
        x_init = rng.uniform(0.1, w_init - 0.1, N)
        y_init = rng.uniform(0.1, h_init - 0.1, N)
        r_init = rng.uniform(0.05, 0.15, N)
        z_rand = np.concatenate([x_init, y_init, r_init, [w_init]])
        inits.append(z_rand)

    # 3. Different aspect ratios
    for w_init in [0.8, 1.2]:
        h_init = 2.0 - w_init
        # Simple grid for this aspect
        side = int(np.ceil(np.sqrt(N * w_init / h_init)))
        xs = (np.arange(side) + 0.5) * w_init / side
        ys = (np.arange(side) + 0.5) * h_init / side
        grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij')
        pos = np.column_stack((grid_x.ravel()[:N], grid_y.ravel()[:N]))
        r_init = np.full(N, min(w_init, h_init) / side * 0.4)
        z_aspect = np.concatenate([pos[:, 0], pos[:, 1], r_init, [w_init]])
        inits.append(z_aspect)

    # Run optimization for each init
    for z_init in inits:
        if not _time_ok(deadline, 5.0):
            break
            
        result = _run_single_optimization(z_init, deadline)
        if result is not None:
            z_sol, val = result
            if val > best_value:
                best_value = val
                best_z = z_sol.copy()

    # Final polish on the best solution if time permits
    if best_z is not None and _time_ok(deadline, 10.0):
        # One more round of refinement
        z_polished = _local_refine_radii(best_z, max_iter=50)
        z_polished = _force_relax(z_polished, max_iter=100)
        valid, val = _check_validity(z_polished)
        if valid and val > best_value:
            best_value = val
            best_z = z_polished

    # Fallback
    if best_z is None:
        # Return a simple valid grid packing
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        xs = (np.arange(side) + 0.5) / side
        ys = (np.arange(side) + 0.5) / side
        grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij')
        pos = np.column_stack((grid_x.ravel()[:N], grid_y.ravel()[:N]))
        r = np.full(N, 0.5 / side - 0.01)
        # Ensure validity
        for i in range(N):
            # Clip radius to fit in box
            r[i] = min(r[i], pos[i, 0], w - pos[i, 0], pos[i, 1], h - pos[i, 1])
        # Simple overlap check and shrink if needed
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
                if d < r[i] + r[j]:
                    # Shrink both
                    new_r = (d - 1e-5) / 2
                    r[i] = min(r[i], new_r)
                    r[j] = min(r[j], new_r)
        best_z = np.concatenate([pos[:, 0], pos[:, 1], r, [w]])

    # Convert z to (N, 3) array [x, y, r]
    x = best_z[0:N]
    y = best_z[N:2*N]
    r = best_z[2*N:3*N]
    result_array = np.column_stack((x, y, r))
    
    return result_array

if __name__ == "__main__":
    res = construct()
    print(f"Shape: {res.shape}")
    print(f"Sum of radii: {np.sum(res[:, 2])}")
