"""Pack 200 equal circles in a unit square to maximize common radius.

Combines:
  - Diverse hexagonal/grid initialisation (from current)
  - Vectorized Simulated Annealing with push-apart relaxation (structural change from ILS)
  - Constraint-focused local search (inspired by Inspiration 1's active set identification)
  - High-beta L-BFGS polishing (from current)
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 200
_TIME_LIMIT = 145  # seconds, leave margin before 150 s kill
_START_TIME = time.time()

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _get_radius(pts):
    """Calculate the maximum possible radius for given centers."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) != N:
        return 0.0
    if np.any(~np.isfinite(pts)):
        return 0.0
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return 0.0

    # Wall distances
    wall_dists = np.min(np.minimum(0.5 - pts, pts + 0.5), axis=1)
    min_wall = np.min(wall_dists)

    # Pairwise distances
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0

    r = min(min_wall, min_pair)
    return r if r > 0 else 0.0

def _objective_and_grad(x, beta):
    """
    Objective function and gradient for L-BFGS-B.
    Minimizes -soft_min(constraints).
    """
    pts = x.reshape((N, 2))

    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)

    C = np.empty(4 * N + num_pairs)

    # Wall values
    C[0:N] = pts[:, 0] + 0.5
    C[N:2*N] = 0.5 - pts[:, 0]
    C[2*N:3*N] = pts[:, 1] + 0.5
    C[3*N:4*N] = 0.5 - pts[:, 1]

    # Pair values
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)
    C[4*N:] = dists / 2.0

    # Soft min computation
    y = -beta * C
    max_y = np.max(y)
    if not np.isfinite(max_y):
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)

    shifted = y - max_y
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)

    if sum_exp == 0:
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)

    soft_min_val = -(max_y + np.log(sum_exp)) / beta
    obj_val = -soft_min_val

    # Weights
    weights = exp_shifted / sum_exp

    # Gradient computation
    grad_pts = np.zeros((N, 2))

    # Wall gradients
    grad_pts[:, 0] += weights[0:N] - weights[N:2*N]
    grad_pts[:, 1] += weights[2*N:3*N] - weights[3*N:4*N]

    # Pair gradients
    valid = dists > 1e-10
    dx_safe = np.where(valid, dx, 1.0)
    dy_safe = np.where(valid, dy, 1.0)
    dists_safe = np.where(valid, dists, 1.0)

    grad_pair_x_i = np.where(valid, dx_safe / (2.0 * dists_safe), 0.0)
    grad_pair_x_j = np.where(valid, -dx_safe / (2.0 * dists_safe), 0.0)
    grad_pair_y_i = np.where(valid, dy_safe / (2.0 * dists_safe), 0.0)
    grad_pair_y_j = np.where(valid, -dy_safe / (2.0 * dists_safe), 0.0)

    pair_weights = weights[4*N:]

    np.add.at(grad_pts[:, 0], i_idx, grad_pair_x_i * pair_weights)
    np.add.at(grad_pts[:, 0], j_idx, grad_pair_x_j * pair_weights)
    np.add.at(grad_pts[:, 1], i_idx, grad_pair_y_i * pair_weights)
    np.add.at(grad_pts[:, 1], j_idx, grad_pair_y_j * pair_weights)

    grad_pts = -grad_pts / beta

    return obj_val, grad_pts.flatten()

def _push_apart_relax(pts, n_iters=50, strength=0.005):
    """
    Vectorized push-apart relaxation.
    Finds closest pairs and pushes them apart.
    """
    pts = pts.copy()
    for _ in range(n_iters):
        tree = cKDTree(pts)
        # Query k=2 to get nearest neighbor
        dists, indices = tree.query(pts, k=2)
        
        # We want to push apart pairs where dist < threshold
        # Here we push all pairs that are closer than some value, or just the closest ones
        # For efficiency, we'll vectorize the push for the closest neighbors
        
        # Identify pairs (i, j) where j is neighbor of i
        i_idx = np.arange(N)
        j_idx = indices[:, 1]
        
        # Avoid duplicates: only process if i < j
        mask = i_idx < j_idx
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        d = dists[mask, 1]
        
        # Only push if distance is small (e.g., < 0.08)
        # And ensure we don't push if they are already well separated
        active = d < 0.08
        
        if not np.any(active):
            continue
            
        i_act = i_idx[active]
        j_act = j_idx[active]
        d_act = d[active]
        
        p_i = pts[i_act]
        p_j = pts[j_act]
        
        diff = p_j - p_i
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        dir_vec = diff / norms
        
        # Push strength proportional to overlap
        # We want to increase distance. 
        # Move i by -dir * strength, j by +dir * strength
        move = strength * dir_vec
        
        # Apply moves
        pts[i_act] -= move
        pts[j_act] += move
        
        # Clip to bounds
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        
    return pts

def _generate_hex_initial(spacing=0.078, seed=42):
    """Generate a centered hexagonal lattice packing, select N best points."""
    rng = np.random.RandomState(seed)
    points = []
    row_h = spacing * np.sqrt(3) / 2

    y = -1.0
    row_idx = 0
    while y < 1.0:
        shift = (spacing / 2.0) if row_idx % 2 == 1 else 0.0
        x = -1.0 + shift
        while x < 1.0:
            points.append([x, y])
            x += spacing
        y += row_h
        row_idx += 1

    pts = np.array(points)

    # Filter points inside the square
    mask = (pts[:, 0] >= -0.5) & (pts[:, 0] <= 0.5) & \
           (pts[:, 1] >= -0.5) & (pts[:, 1] <= 0.5)
    pts = pts[mask]

    if len(pts) < N:
        return rng.uniform(-0.45, 0.45, (N, 2))

    # Add small jitter to break symmetry
    pts = pts + rng.randn(len(pts), 2) * 0.002
    pts = np.clip(pts, -0.5, 0.5)

    # Select N points closest to the center
    dists_to_center = np.sum(pts**2, axis=1)
    indices = np.argsort(dists_to_center)
    pts = pts[indices[:N]]

    return pts

def _generate_grid_initial(rows=14, cols=15, seed=42):
    """Generate a regular grid and select N points."""
    rng = np.random.RandomState(seed)
    dx = 1.0 / cols
    dy = 1.0 / rows
    xs = -0.5 + (np.arange(cols) + 0.5) * dx
    ys = -0.5 + (np.arange(rows) + 0.5) * dy
    grid = np.array([(x, y) for y in ys for x in xs])

    if len(grid) >= N:
        # Jitter and select
        grid = grid + rng.randn(len(grid), 2) * 0.003
        grid = np.clip(grid, -0.5, 0.5)
        dists_to_center = np.sum(grid**2, axis=1)
        indices = np.argsort(dists_to_center)
        return grid[indices[:N]]
    else:
        return rng.uniform(-0.45, 0.45, (N, 2))

def _simulated_annealing(pts, time_limit, rng):
    """
    Simulated Annealing with vectorized push-apart relaxation.
    """
    best_pts = pts.copy()
    best_r = _get_radius(best_pts)
    current_pts = pts.copy()
    current_r = best_r
    
    t_start = time.time()
    T = 0.01
    T_min = 0.0001
    cooling_rate = 0.9995
    
    while time.time() - t_start < time_limit:
        # 1. Relaxation: Push apart closest pairs
        relaxed_pts = _push_apart_relax(current_pts, n_iters=20, strength=0.005)
        relaxed_r = _get_radius(relaxed_pts)
        
        # 2. Perturbation: Move some random points
        # Pick k points to perturb
        k = 5
        indices = rng.choice(N, size=k, replace=False)
        
        perturbed_pts = relaxed_pts.copy()
        # Random step
        steps = rng.randn(k, 2) * T
        perturbed_pts[indices] += steps
        perturbed_pts[:, 0] = np.clip(perturbed_pts[:, 0], -0.5, 0.5)
        perturbed_pts[:, 1] = np.clip(perturbed_pts[:, 1], -0.5, 0.5)
        
        perturbed_r = _get_radius(perturbed_pts)
        
        # 3. Acceptance
        delta_r = perturbed_r - relaxed_r
        if delta_r > 0:
            # Accept improvement
            current_pts = perturbed_pts
            current_r = perturbed_r
            if current_r > best_r:
                best_r = current_r
                best_pts = current_pts.copy()
        else:
            # Accept with probability exp(-delta/T)
            if T > T_min:
                prob = np.exp(-delta_r / T)
                if rng.rand() < prob:
                    current_pts = perturbed_pts
                    current_r = perturbed_r
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Restart if stuck for a while? Or just keep going.
            # Let's add a small random kick if T is very low and no improvement for a while
            # For now, simple cooling.

    return best_pts, best_r

def _gradient_local_search(pts, max_time=5.0):
    """
    Gradient-based local search using L-BFGS-B with adaptive beta.
    """
    t_start = time.time()
    current_pts = pts.copy()
    current_r = _get_radius(current_pts)
    
    # Try increasing betas
    betas = [50, 100, 200, 500]
    
    for beta in betas:
        if time.time() - t_start > max_time:
            break
        
        try:
            x0 = current_pts.flatten()
            
            def fun_and_grad(x):
                return _objective_and_grad(x, beta)
            
            res = minimize(fun_and_grad, x0, method='L-BFGS-B',
                          jac=True, options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < -current_r:
                new_pts = res.x.reshape(N, 2)
                new_pts = np.clip(new_pts, -0.5, 0.5)
                new_r = _get_radius(new_pts)
                
                if new_r > current_r:
                    current_pts = new_pts
                    current_r = new_r
        except Exception:
            continue
    
    return current_pts

def construct():
    t0 = time.time()
    best_r = 0.0
    best_pts = None

    # --- Phase 1: Multi-start initialisation ---
    initialisations = [
        ("hex_0.078", _generate_hex_initial(0.078, seed=42)),
        ("hex_0.080", _generate_hex_initial(0.080, seed=123)),
        ("grid_14x15", _generate_grid_initial(14, 15, seed=42)),
        ("grid_20x10", _generate_grid_initial(20, 10, seed=99)),
    ]

    rng = np.random.RandomState(42)

    for name, init_pts in initialisations:
        if _time_left() < 30:  # Leave time for SA and polishing
            break
        try:
            # Quick relaxation to get a valid starting point
            pts_relaxed = _push_apart_relax(init_pts, n_iters=50, strength=0.005)
            r_relaxed = _get_radius(pts_relaxed)
            
            # Run SA for a short time on this init
            # Allocate time based on remaining time
            time_for_sa = min(10.0, _time_left() - 20) 
            if time_for_sa > 5:
                pts_sa, r_sa = _simulated_annealing(pts_relaxed, time_for_sa, rng)
                if r_sa > r_relaxed:
                    r_relaxed = r_sa
                    pts_relaxed = pts_sa
            
            if r_relaxed > best_r:
                best_r = r_relaxed
                best_pts = pts_relaxed
        except Exception:
            pass

    # Fallback if nothing worked
    if best_pts is None:
        best_pts = _generate_grid_initial(20, 10, seed=42)
        best_r = _get_radius(best_pts)

    # --- Phase 2: Gradient Polishing ---
    if best_r > 0 and _time_left() > 5:
        try:
            pts_polish = _gradient_local_search(best_pts, max_time=_time_left() - 2)
            r_polish = _get_radius(pts_polish)
            if r_polish > best_r:
                best_r = r_polish
                best_pts = pts_polish
        except Exception:
            pass

    # --- Phase 3: Final Push-Apart Polish ---
    if best_r > 0 and _time_left() > 2:
        try:
            pts_final = _push_apart_relax(best_pts, n_iters=100, strength=0.002)
            r_final = _get_radius(pts_final)
            if r_final > best_r:
                best_r = r_final
                best_pts = pts_final
        except Exception:
            pass

    if best_r <= 0:
        best_pts = _generate_grid_initial(20, 10, seed=42)

    return best_pts
