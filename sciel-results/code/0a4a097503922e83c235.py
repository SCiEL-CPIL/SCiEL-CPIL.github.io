"""
Sphere packing optimization using Simulated Annealing with adaptive moves.
Strategy:
1. Use Simulated Annealing (SA) to explore the landscape of sphere packings.
2. The objective is to maximize the minimum radius r = min(min_pair_dist/2, min_wall_dist).
3. SA allows escaping local minima by accepting worse configurations with a probability
   that decreases as the 'temperature' drops.
4. Moves are local perturbations: pick a random sphere and displace it by a random vector
   whose magnitude is proportional to the current temperature.
5. Adaptive move size: if acceptance rate is too low, increase move size; if too high, decrease.
6. Multi-start: run SA from several diverse initial configurations (grid, FCC, random).
7. Final polish: use a few steps of L-BFGS-B on the best SA result to refine it.
"""

import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50
CUBE_HALF = 0.5
DEADLINE = 115.0  # Safety margin for 120s limit (evaluator kills at 150s)

def _compute_radius(pts):
    """Compute the true minimum radius for a configuration."""
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    wall_dists = CUBE_HALF - np.abs(pts)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _sa_optimize(start_pts, time_limit, seed=0):
    """
    Run Simulated Annealing to maximize the minimum radius.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    start_time = time.time()
    
    # Initial state
    current_pts = start_pts.copy()
    current_r = _compute_radius(current_pts)
    
    best_pts = current_pts.copy()
    best_r = current_r
    
    # Temperature schedule
    T_start = 0.05
    T_end = 1e-6
    # Number of iterations: we'll estimate based on time budget
    # Each iteration is O(N^2) due to radius computation, ~2500 ops
    # Let's aim for ~50000 iterations total, adjusting per run
    total_iters = 50000
    iters_per_run = max(1000, total_iters // 5)
    
    # Adaptive move size
    move_scale = 0.05
    min_move = 1e-5
    max_move = 0.2
    
    # Track acceptance for adaptation
    accept_count = 0
    eval_count = 0
    window_size = 500
    
    for it in range(iters_per_run):
        if time.time() - start_time > time_limit - 2:
            break
        
        # Temperature: exponential cooling
        progress = it / max(1, iters_per_run - 1)
        T = T_start * (T_end / T_start) ** progress
        
        # Generate move: pick random sphere, random displacement
        i = rng.randrange(N)
        # Move size proportional to sqrt(T) for better exploration
        step_size = move_scale * np.sqrt(T / T_start + 0.1)
        step_size = np.clip(step_size, min_move, max_move)
        
        # Random direction
        delta = np_rng.randn(3) * step_size
        
        new_pts = current_pts.copy()
        new_pts[i] = current_pts[i] + delta
        # Clip to cube
        new_pts[i] = np.clip(new_pts[i], -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
        
        # Compute new radius
        new_r = _compute_radius(new_pts)
        eval_count += 1
        
        # Acceptance criterion
        delta_r = new_r - current_r
        if delta_r > 0:
            # Better: always accept
            current_pts = new_pts
            current_r = new_r
            accept_count += 1
            if current_r > best_r:
                best_r = current_r
                best_pts = current_pts.copy()
        else:
            # Worse: accept with probability exp(delta_r / T)
            # delta_r is negative, T is positive
            prob = np.exp(delta_r / T)
            if rng.random() < prob:
                current_pts = new_pts
                current_r = new_r
                accept_count += 1
        
        # Adaptive move size every window_size evaluations
        if eval_count % window_size == 0:
            accept_rate = accept_count / window_size
            if accept_rate < 0.1:
                # Too few accepts: increase move size
                move_scale *= 1.2
            elif accept_rate > 0.6:
                # Too many accepts: decrease move size
                move_scale *= 0.8
            move_scale = np.clip(move_scale, 0.01, 0.3)
            accept_count = 0
        
        # Occasional restart from best if stuck
        if it % 5000 == 4999 and current_r < best_r * 0.95:
            # Reheat: go back to best and increase temperature
            current_pts = best_pts.copy()
            current_r = best_r
            # Reset temperature by reducing iteration count effect
            # (This is a simple heuristic; proper reheat would adjust T schedule)
    
    return best_pts

def _generate_grid_start():
    """Generate a 50-point starting configuration from a jittered 4x4x4 grid."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    np.random.seed(42)
    grid += np.random.randn(*grid.shape) * 0.02
    grid = np.clip(grid, -0.49, 0.49)
    return grid[:N].copy()

def _generate_fcc_start():
    """Generate starting configuration from FCC lattice points."""
    points = []
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                for offset in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i + offset[0], j + offset[1], k + offset[2]])
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        points.append(p)
    points = np.array(points)
    if len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        extra = np.random.randn(N - len(points), 3) * 0.3
        points = np.vstack([points, extra])
    return points[:N].copy()

def _generate_random_start(seed):
    np.random.seed(seed)
    pts = np.random.uniform(-0.45, 0.45, size=(N, 3))
    return pts

def _soft_min_radius(pts, beta):
    """
    Compute the smooth surrogate for the minimum radius.
    Returns the estimated radius r and the gradient of -r (for minimization).
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    wall_dists = CUBE_HALF - np.abs(pts)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    wall_vals = wall_dists.flatten()
    all_vals = np.concatenate([pair_dists, wall_vals])
    
    min_val = np.min(all_vals)
    shifted = all_vals - min_val
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    
    r_val = min_val - np.log(sum_exp) / beta
    
    p = exp_terms / sum_exp
    grad = np.zeros_like(pts)
    
    p_pairs = p[:len(pair_dists)]
    diff_vecs = pts[i_idx] - pts[j_idx]
    dists_vec = np.sqrt(np.sum(diff_vecs * diff_vecs, axis=1))
    dists_vec[dists_vec == 0] = 1e-10
    unit_vecs = diff_vecs / dists_vec[:, np.newaxis]
    weighted_units = unit_vecs * (0.5 * p_pairs[:, np.newaxis])
    
    np.add.at(grad, i_idx, weighted_units)
    np.add.at(grad, j_idx, -weighted_units)
    
    p_walls = p[len(pair_dists):]
    p_walls_reshaped = p_walls.reshape(N, 3)
    grad += -p_walls_reshaped * np.sign(pts)
    
    return -r_val, -grad

def _objective_and_grad(pts_flat, beta):
    """Wrapper for L-BFGS-B."""
    pts = pts_flat.reshape(N, 3)
    val, grad = _soft_min_radius(pts, beta)
    return val, grad

def _polish_lbgfgs(start_pts, time_limit):
    """
    Run L-BFGS-B optimization on the smooth surrogate for polishing.
    """
    start_time = time.time()
    x0 = start_pts.flatten()
    bounds = [(-CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)] * (N * 3)
    
    betas = [100, 300, 800]
    current_x = x0
    best_val = _compute_radius(current_x.reshape(N, 3))
    best_x = x0
    
    for beta in betas:
        if time.time() - start_time > time_limit - 3:
            break
        try:
            res = minimize(
                _objective_and_grad,
                current_x,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.success:
                current_x = res.x
                val = _compute_radius(res.x.reshape(N, 3))
                if val > best_val:
                    best_val = val
                    best_x = res.x
        except Exception:
            pass
    
    return best_x.reshape(N, 3)

def construct():
    start_time = time.time()
    
    # Generate diverse initial points
    starts = []
    starts.append(_generate_grid_start())
    starts.append(_generate_fcc_start())
    for i in range(4):
        starts.append(_generate_random_start(100 + i))
    
    best_pts = None
    best_score = -np.inf
    
    # Run SA on each start, allocating time
    num_runs = len(starts)
    time_per_run = (DEADLINE - (time.time() - start_time) - 30) / num_runs  # Reserve 30s for polish
    
    for i, init_pts in enumerate(starts):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Run SA
        sa_pts = _sa_optimize(init_pts, time_per_run, seed=1000 + i)
        sa_score = _compute_radius(sa_pts)
        
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts
    
    # Polish the best result with L-BFGS-B
    if best_pts is not None and time.time() - start_time < DEADLINE - 10:
        remaining_time = DEADLINE - 10 - (time.time() - start_time)
        polished_pts = _polish_lbgfgs(best_pts, remaining_time)
        polished_score = _compute_radius(polished_pts)
        
        if polished_score > best_score:
            best_pts = polished_pts
            best_score = polished_score
    
    # Fallback
    if best_pts is None:
        best_pts = starts[0]
    
    # Ensure validity
    best_pts = np.clip(best_pts, -0.4999999, 0.4999999)
    
    return best_pts
