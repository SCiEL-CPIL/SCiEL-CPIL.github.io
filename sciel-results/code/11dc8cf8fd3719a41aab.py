"""
Sphere packing optimization using Simulated Annealing with adaptive moves.
Strategy:
1. Use Simulated Annealing (SA) to explore the landscape of sphere packings.
2. The objective is to maximize the minimum radius r.
3. Moves are proposed by picking a random sphere and displacing it by a random vector
   whose magnitude is scaled by the current temperature.
4. The temperature schedule is geometric, with occasional "reheat" if no improvement
   is found for a long time.
5. The SA is run from multiple diverse starting configurations (FCC, BCC, random).
6. After SA, a final polish with L-BFGS-B on the Log-Sum-Exp surrogate is applied
   to the best configuration found by SA.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50
CUBE_HALF = 0.5
DEADLINE = 115.0  # Safety margin for 120s limit

def _score(pts):
    """Compute the true radius for a configuration."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    min_wall = np.min(0.5 - np.abs(pts))
    return min(min_dist / 2.0, min_wall)

def _soft_min_radius(pts, beta):
    """
    Compute the smooth surrogate for the minimum radius and its gradient.
    Returns (value, gradient) where value = -soft_min_radius (to minimize).
    """
    n = len(pts)
    i_idx, j_idx = np.triu_indices(n, k=1)
    diff_vecs = pts[i_idx] - pts[j_idx]
    dists_vec = np.sqrt(np.sum(diff_vecs * diff_vecs, axis=1))
    dists_vec[dists_vec == 0] = 1e-10
    unit_vecs = diff_vecs / dists_vec[:, np.newaxis]
    
    pair_dists = dists_vec / 2.0
    wall_dists = CUBE_HALF - np.abs(pts)
    wall_vals = wall_dists.flatten()
    
    all_vals = np.concatenate([pair_dists, wall_vals])
    
    min_val = np.min(all_vals)
    shifted = all_vals - min_val
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    
    r_val = min_val - np.log(sum_exp) / beta
    
    p = exp_terms / sum_exp
    p_pairs = p[:len(pair_dists)]
    p_walls = p[len(pair_dists):].reshape(N, 3)
    
    grad = np.zeros_like(pts)
    weighted_units = unit_vecs * (0.5 * p_pairs[:, np.newaxis])
    np.add.at(grad, i_idx, weighted_units)
    np.add.at(grad, j_idx, -weighted_units)
    grad += -p_walls * np.sign(pts)
    
    return -r_val, -grad

def _objective_and_grad(pts_flat, beta):
    pts = pts_flat.reshape(N, 3)
    val, grad = _soft_min_radius(pts, beta)
    return val, grad

def _generate_fcc_start():
    """Generate starting configuration from FCC lattice points."""
    np.random.seed(123)
    points = []
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                for offset in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
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

def _generate_bcc_start():
    """Generate starting configuration from BCC lattice points."""
    np.random.seed(456)
    points = []
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                p = np.array([i, j, k])
                if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                    points.append(p)
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5])
                if -0.5 <= p2[0] <= 0.5 and -0.5 <= p2[1] <= 0.5 and -0.5 <= p2[2] <= 0.5:
                    points.append(p2)
    points = np.array(points)
    if len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        extra = np.random.randn(N - len(points), 3) * 0.3
        points = np.vstack([points, extra])
    return points[:N].copy()

def _generate_grid_start():
    """Generate a 50-point starting configuration from a jittered grid."""
    np.random.seed(789)
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid += np.random.randn(*grid.shape) * 0.02
    grid = np.clip(grid, -0.49, 0.49)
    return grid[:N].copy()

def _generate_random_start(seed):
    np.random.seed(seed)
    pts = np.random.uniform(-0.45, 0.45, size=(N, 3))
    return pts

def _sa_optimize(start_pts, time_limit):
    """
    Run Simulated Annealing on a starting configuration.
    Returns the best configuration found.
    """
    start_time = time.time()
    pts = start_pts.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Initial temperature
    T = 0.05
    T_min = 0.0001
    cooling_rate = 0.9995
    
    # Adaptive move size
    move_scale = 0.05
    
    # Track no-improvement count for reheat
    no_improve = 0
    reheat_threshold = 5000
    
    while time.time() - start_time < time_limit:
        # Check time more frequently
        if time.time() - start_time > time_limit - 1:
            break
        
        # Pick a random sphere to move
        i = np.random.randint(N)
        
        # Propose a random displacement
        # Scale by temperature and move_scale
        disp = np.random.randn(3) * move_scale * np.sqrt(T / 0.05)
        new_pts = pts.copy()
        new_pts[i] += disp
        
        # Clip to cube
        new_pts[i] = np.clip(new_pts[i], -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
        
        # Evaluate
        new_score = _score(new_pts)
        delta = new_score - best_score
        
        # Accept if improvement, or with probability exp(delta / T)
        if delta > 0:
            pts = new_pts
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            # delta <= 0
            if np.random.rand() < np.exp(delta / T):
                pts = new_pts
                no_improve += 1
            else:
                no_improve += 1
        
        # Cool down
        T *= cooling_rate
        
        # Reheat if stuck
        if no_improve > reheat_threshold:
            T = max(T, 0.02)
            no_improve = 0
        
        # Adaptive move size: reduce if acceptance rate is high, increase if low
        # Simple heuristic: reduce move_scale slowly
        move_scale *= 0.99999
        
        # Ensure T doesn't go below T_min
        if T < T_min:
            T = T_min
    
    return best_pts

def _optimize_lbgfgs(start_pts, time_limit, seed=0):
    """
    Run L-BFGS-B optimization on the smooth surrogate.
    """
    start_time = time.time()
    x0 = start_pts.flatten()
    
    bounds = [(-CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)] * (N * 3)
    
    betas = [20, 50, 100, 200]
    
    current_x = x0
    best_val = np.inf
    best_x = x0
    
    for beta in betas:
        if time.time() - start_time > time_limit - 5:
            break
        
        try:
            res = minimize(
                _objective_and_grad,
                current_x,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.success or res.fun < best_val:
                best_val = res.fun
                best_x = res.x
                current_x = res.x
        except Exception:
            pass
    
    best_pts = best_x.reshape(N, 3)
    return best_pts

def construct():
    start_time = time.time()
    
    # Generate diverse initial points
    starts = []
    starts.append(_generate_grid_start())
    starts.append(_generate_fcc_start())
    starts.append(_generate_bcc_start())
    
    # Add random starts
    for i in range(3):
        starts.append(_generate_random_start(100 + i))
    
    best_pts = None
    best_score = -np.inf
    
    # Score initial points to pick the best ones
    scored_starts = [(s, _score(s)) for s in starts]
    scored_starts.sort(key=lambda x: x[1], reverse=True)
    
    # Allocate time budget
    # 115s total.
    # Run SA on the top 3 starts, giving each ~20s.
    # Then run L-BFGS-B polish on the best SA result with ~50s.
    
    num_sa_runs = 3
    time_per_sa = (DEADLINE - (time.time() - start_time) - 55) / num_sa
    
    for i in range(num_sa_runs):
        if time.time() - start_time > DEADLINE - 60:
            break
        
        init_pts, init_score = scored_starts[i]
        
        # Run SA
        sa_pts = _sa_optimize(init_pts, time_per_sa)
        
        # Evaluate true score
        current_score = _score(sa_pts)
        
        if current_score > best_score:
            best_score = current_score
            best_pts = sa_pts
        
        # Check time
        if time.time() - start_time > DEADLINE - 60:
            break
    
    # If we have time left, do a final polish on the best result with L-BFGS-B
    if best_pts is not None and time.time() - start_time < DEADLINE - 15:
        remaining_time = DEADLINE - 15 - (time.time() - start_time)
        
        # Run L-BFGS-B polish
        polished_pts = _optimize_lbgfgs(best_pts, remaining_time)
        
        polished_score = _score(polished_pts)
        
        if polished_score > best_score:
            best_pts = polished_pts
            best_score = polished_score
    
    # Fallback
    if best_pts is None:
        best_pts = scored_starts[0][0]
    
    # Ensure validity
    best_pts = np.clip(best_pts, -0.4999999, 0.4999999)
    
    return best_pts
