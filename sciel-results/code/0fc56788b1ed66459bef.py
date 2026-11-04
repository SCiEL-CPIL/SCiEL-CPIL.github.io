"""
Sphere packing optimization using Multi-Start L-BFGS-B on a Log-Sum-Exp surrogate.
Combined design:
- Current program's multi-start framework with diverse initializations
- Inspiration's aggressive beta annealing (10 -> 400000)
- Inspiration's active-set refinement for binding constraints
- logsumexp for numerical stability
- 150s deadline with 8s safety margin
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 50
CUBE_HALF = 0.5
DEADLINE_OFFSET = 142.0  # 8s safety margin before 150s limit

def _get_radius(c):
    """Compute the exact maximum common radius for centers c (N,3)."""
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair_dist = np.min(d)
    # Distance to each of the 6 walls
    dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
    min_wall_dist = np.min(dist_to_walls)
    return min(min_pair_dist / 2.0, min_wall_dist)

def _soft_min_radius(pts, beta):
    """
    Compute the smooth surrogate for the minimum radius.
    Returns (value, gradient) for minimization of -soft_min_radius.
    """
    n = len(pts)
    
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return 1e6, np.zeros_like(pts.flatten())
    
    # Precomputed indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    n_pairs = len(i_idx)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    dists = np.sqrt(dists_sq)
    
    # Pair terms: d_ij / 2
    pair_dists = dists[i_idx, j_idx] / 2.0
    
    # Wall terms: 3N values (min of + and - wall for each axis)
    # 0.5 - |x| = min(0.5-x, 0.5+x)
    wall_vals = (CUBE_HALF - np.abs(pts)).flatten()
    
    # Combine all constraint values
    all_vals = np.concatenate([pair_dists, wall_vals])
    
    # Soft min using logsumexp
    neg_beta_g = -beta * all_vals
    log_sum_exp = logsumexp(neg_beta_g)
    r_val = -log_sum_exp / beta
    
    # Weights
    exp_terms = np.exp(neg_beta_g - log_sum_exp)
    p = exp_terms  # These sum to 1
    
    # Gradient of R (soft min)
    grad = np.zeros_like(pts)
    
    # Pair contributions
    p_pairs = p[:n_pairs]
    diff_vecs = pts[i_idx] - pts[j_idx]
    dists_vec = np.sqrt(np.sum(diff_vecs * diff_vecs, axis=1))
    safe_d = np.where(dists_vec > 1e-14, dists_vec, 1.0)
    unit_vecs = diff_vecs / safe_d[:, np.newaxis]
    
    # dR/dx_i for pair (i,j): p * 0.5 * (x_i - x_j) / d
    weighted_units = unit_vecs * (0.5 * p_pairs[:, np.newaxis])
    np.add.at(grad, i_idx, weighted_units)
    np.add.at(grad, j_idx, -weighted_units)
    
    # Wall contributions
    p_walls = p[n_pairs:]
    p_walls_reshaped = p_walls.reshape(n, 3)
    # s_wall = 0.5 - |x|, ds/dx = -sign(x)
    # dR/dx_wall = sum p_i * (-sign(x_i)) = -p_walls * sign(x)
    grad += -p_walls_reshaped * np.sign(pts)
    
    # We want to minimize F = -R, so return (-r_val, -grad)
    return -r_val, -grad.flatten()

def _optimize_lbgfgs(start_pts, time_budget, seed=0):
    """
    Run L-BFGS-B optimization with aggressive beta annealing.
    """
    start_time = time.time()
    x0 = start_pts.flatten()
    bounds = [(-CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)] * (N * 3)
    
    # Aggressive beta annealing schedule
    betas = [10, 20, 50, 100, 200, 500, 1000, 5000, 10000, 50000, 100000, 200000, 400000]
    
    current_x = x0
    best_val = np.inf
    best_x = x0
    
    for beta in betas:
        elapsed = time.time() - start_time
        if elapsed > time_budget - 3:
            break
        
        try:
            res = minimize(
                _soft_min_radius,
                current_x,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100,
                         'maxfun': 3000, 'maxcor': 50}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
                current_x = res.x
        except Exception:
            pass
        
        # Check time
        if time.time() - start_time > time_budget - 5:
            break
    
    return best_x.reshape(N, 3)

def _active_set_polish(c_init, time_budget):
    """
    Identify binding constraints and polish with focused optimization.
    """
    c = c_init.copy()
    r = _get_radius(c)
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    for iteration in range(10):
        elapsed = time.time()
        if elapsed > time_budget:
            break
        
        # Compute current radius
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        r_current = min(min_pair_dist / 2.0, min_wall_dist)
        
        # Adaptive tolerance
        tol = 0.003 * r_current + 1e-9
        
        # Find binding pairs
        pair_dists = d[i_idx, j_idx] / 2.0
        binding_pairs = np.where(pair_dists <= r_current + tol)[0]
        
        # Find binding walls
        dist_pos = 0.5 - c
        dist_neg = c + 0.5
        pos_flat = dist_pos.flatten()
        neg_flat = dist_neg.flatten()
        
        pos_bind_flat = np.where(pos_flat <= r_current + tol)[0]
        neg_bind_flat = np.where(neg_flat <= r_current + tol)[0]
        
        pos_bind_i = pos_bind_flat // 3
        pos_bind_k = pos_bind_flat % 3
        neg_bind_i = neg_bind_flat // 3
        neg_bind_k = neg_bind_flat % 3
        
        n_bind_pos = len(pos_bind_i)
        n_bind_neg = len(neg_bind_i)
        n_bind_pairs = len(binding_pairs)
        
        if n_bind_pairs == 0 and n_bind_pos == 0 and n_bind_neg == 0:
            break
        
        # Build binding-only objective
        def binding_obj(c_flat, beta):
            c = c_flat.reshape(N, 3)
            if np.any(np.abs(c) > 0.5 + 1e-9):
                return 1e6, np.zeros_like(c_flat)
            
            g_all = []
            grad_c = np.zeros_like(c)
            
            if n_bind_pairs > 0:
                diff_bind = c[i_idx[binding_pairs]] - c[j_idx[binding_pairs]]
                d_bind = np.sqrt(np.sum(diff_bind * diff_bind, axis=1)) / 2.0
                g_all.append(d_bind)
                
                safe_d = np.where(d_bind * 2 > 1e-14, d_bind * 2, 1.0)
                unit_vecs = diff_bind / safe_d[:, np.newaxis]
                # Weights will be applied after logsumexp
                # Store for later
                pass
            
            if n_bind_pos > 0:
                wall_pos = 0.5 - c[pos_bind_i, pos_bind_k]
                g_all.append(wall_pos)
            
            if n_bind_neg > 0:
                wall_neg = c[neg_bind_i, neg_bind_k] + 0.5
                g_all.append(wall_neg)
            
            if len(g_all) == 0:
                return 0.0, np.zeros_like(c_flat)
            
            g_concat = np.concatenate(g_all)
            neg_beta_g = -beta * g_concat
            log_sum_exp = logsumexp(neg_beta_g)
            val = -log_sum_exp / beta
            weights = np.exp(neg_beta_g - log_sum_exp)
            
            # Gradient
            offset = 0
            if n_bind_pairs > 0:
                w_pairs = weights[offset:offset+n_bind_pairs]
                offset += n_bind_pairs
                diff_bind = c[i_idx[binding_pairs]] - c[j_idx[binding_pairs]]
                d_bind_full = np.sqrt(np.sum(diff_bind * diff_bind, axis=1))
                safe_d = np.where(d_bind_full > 1e-14, d_bind_full, 1.0)
                unit_vecs = diff_bind / safe_d[:, np.newaxis]
                contrib = unit_vecs * (0.5 * w_pairs[:, np.newaxis])
                np.add.at(grad_c, i_idx[binding_pairs], contrib)
                np.add.at(grad_c, j_idx[binding_pairs], -contrib)
            
            if n_bind_pos > 0:
                w_pos = weights[offset:offset+n_bind_pos]
                offset += n_bind_pos
                # s = 0.5 - x, ds/dx = -1
                # dR/dx = sum w * (-1) = -w
                # dF/dx = -dR/dx = w
                np.add.at(grad_c, (pos_bind_i, pos_bind_k), -w_pos[:, np.newaxis])
            
            if n_bind_neg > 0:
                w_neg = weights[offset:offset+n_bind_neg]
                # s = x + 0.5, ds/dx = 1
                # dR/dx = sum w * 1 = w
                # dF/dx = -w
                np.add.at(grad_c, (neg_bind_i, neg_bind_k), w_neg[:, np.newaxis])
            
            return -val, -grad_c.flatten()
        
        # Run optimization on binding set
        betas_binding = [100, 500, 2000, 10000]
        x0 = c.flatten()
        bounds = [(-0.4999, 0.4999)] * (N * 3)
        current_x = x0
        
        for beta in betas_binding:
            if time.time() > time_budget - 2:
                break
            try:
                res = minimize(
                    binding_obj,
                    current_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50,
                             'maxfun': 2000, 'maxcor': 30}
                )
                if res.fun < _get_radius(current_x.reshape(N,3)) or True:
                    current_x = res.x
            except Exception:
                pass
        
        c = current_x.reshape(N, 3)
        r_new = _get_radius(c)
        if r_new <= r_current + 1e-15:
            break
        r = r_new
    
    return c

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
    np.random.seed(123)
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

def _generate_bcc_start():
    """Generate starting configuration from BCC lattice points."""
    np.random.seed(456)
    points = []
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                for offset in [(0,0,0), (0.5,0.5,0.5)]:
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

def _generate_hcp_start():
    """Generate starting configuration from HCP-like layered structure."""
    np.random.seed(789)
    points = []
    # Layered structure with hexagonal packing in each layer
    layers_z = np.linspace(-0.4, 0.4, 7)
    for z in layers_z:
        # Hexagonal packing in xy plane
        for row in range(-2, 3):
            for col in range(-2, 3):
                x = col * 0.35
                y = row * 0.303 + (0.1515 if col % 2 else 0)
                if abs(x) <= 0.48 and abs(y) <= 0.48:
                    points.append([x, y, z])
    points = np.array(points)
    if len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        extra = np.random.randn(N - len(points), 3) * 0.3
        points = np.vstack([points, extra])
    return points[:N].copy()

def _generate_layered_start():
    """Generate starting configuration from simple layered cubic structure."""
    np.random.seed(321)
    points = []
    # 5 layers of 10 points each
    for layer in range(5):
        z = -0.4 + layer * 0.2
        for i in range(5):
            for j in range(2):
                x = -0.36 + i * 0.18
                y = -0.18 + j * 0.36
                if abs(x) <= 0.48 and abs(y) <= 0.48:
                    points.append([x, y, z])
    points = np.array(points)
    if len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        extra = np.random.randn(N - len(points), 3) * 0.3
        points = np.vstack([points, extra])
    return points[:N].copy()

def _generate_rotated_random_start(seed):
    """Generate random start with a rotation applied."""
    np.random.seed(seed)
    pts = np.random.uniform(-0.45, 0.45, size=(N, 3))
    # Apply a random rotation
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2*np.pi)
    R = np.array([
        [np.cos(theta)*np.cos(phi), -np.sin(phi), np.sin(theta)*np.cos(phi)],
        [np.cos(theta)*np.sin(phi), np.cos(phi), np.sin(theta)*np.sin(phi)],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    pts = pts @ R.T
    pts = np.clip(pts, -0.49, 0.49)
    return pts

def _generate_random_start(seed):
    np.random.seed(seed)
    pts = np.random.uniform(-0.45, 0.45, size=(N, 3))
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_OFFSET
    
    # Generate diverse initial points
    starts = []
    starts.append(_generate_grid_start())
    starts.append(_generate_fcc_start())
    starts.append(_generate_bcc_start())
    starts.append(_generate_hcp_start())
    starts.append(_generate_layered_start())
    
    # Add rotated random starts
    for i in range(4):
        starts.append(_generate_rotated_random_start(100 + i))
    
    # Add plain random starts
    for i in range(3):
        starts.append(_generate_random_start(200 + i))
    
    # Score and sort
    def _score(pts):
        return _get_radius(pts)
    
    scored_starts = [(s, _score(s)) for s in starts]
    scored_starts.sort(key=lambda x: x[1], reverse=True)
    
    best_pts = None
    best_score = -np.inf
    
    # Run L-BFGS on top starts with time budget
    num_runs = min(6, len(scored_starts))
    time_per_run = (DEADLINE_OFFSET - 20) / num_runs  # Reserve 20s for polish
    
    for i in range(num_runs):
        if time.time() > deadline - 15:
            break
        
        init_pts, init_score = scored_starts[i]
        remaining = deadline - time.time() - 15
        t_budget = min(time_per_run, remaining)
        
        if t_budget < 5:
            break
        
        optimized_pts = _optimize_lbgfgs(init_pts, t_budget, seed=i)
        current_score = _score(optimized_pts)
        
        if current_score > best_score:
            best_score = current_score
            best_pts = optimized_pts
    
    # Active-set polish on best result
    if best_pts is not None and time.time() < deadline - 10:
        remaining_time = deadline - 10 - start_time
        polished_pts = _active_set_polish(best_pts, start_time + remaining_time)
        polished_score = _score(polished_pts)
        
        if polished_score > best_score:
            best_pts = polished_pts
            best_score = polished_score
    
    # Final high-beta polish if time remains
    if best_pts is not None and time.time() < deadline - 5:
        remaining_time = deadline - 5 - start_time
        x0 = best_pts.flatten()
        bounds = [(-CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)] * (N * 3)
        
        for beta in [50000, 100000, 200000, 400000]:
            if time.time() > deadline - 3:
                break
            try:
                res = minimize(
                    _soft_min_radius,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50,
                             'maxfun': 1000, 'maxcor': 30}
                )
                if res.fun < _score(x0.reshape(N,3)):
                    x0 = res.x
            except Exception:
                pass
        
        final_pts = x0.reshape(N, 3)
        final_score = _score(final_pts)
        
        if final_score > best_score:
            best_pts = final_pts
            best_score = final_score
    
    # Fallback
    if best_pts is None:
        best_pts = scored_starts[0][0]
    
    # Ensure validity
    best_pts = np.clip(best_pts, -0.4999999, 0.4999999)
    
    return best_pts
