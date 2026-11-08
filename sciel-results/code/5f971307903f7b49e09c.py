import numpy as np
from scipy.optimize import minimize
import time
import math

N = 21

def _soft_min_sum_and_grad(centers, W, H, beta):
    """Compute sum of soft-min radii and its gradient w.r.t. centers (vectorized).
    
    Returns (value, grad) where value = sum_i soft_min(dists_i) and grad has shape (2N,).
    """
    n = N
    # Wall distances: shape (n, 4)
    d_left = centers[:, 0]
    d_right = W - centers[:, 0]
    d_bottom = centers[:, 1]
    d_top = H - centers[:, 1]
    
    # Inter-circle half-distances: shape (n, n)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-30) / 2.0
    np.fill_diagonal(dists, np.inf)
    
    # All distances for each circle: shape (n, n+4)
    all_dists = np.concatenate([dists, np.column_stack([d_left, d_right, d_bottom, d_top])], axis=1)
    
    # Soft-min computation
    mins = np.min(all_dists, axis=1)
    shifted = all_dists - mins[:, np.newaxis]
    exp_terms = np.exp(-beta * shifted)
    log_sums = np.log(np.sum(exp_terms, axis=1))
    soft_mins = mins - (1.0 / beta) * log_sums
    value = np.sum(soft_mins)
    
    # Probabilities: p[i,j] = exp(-beta*(d_ij - m_i)) / sum_k exp(-beta*(d_ik - m_i))
    probs = exp_terms / np.sum(exp_terms, axis=1)[:, np.newaxis]
    
    # Gradient: dS_i/dx_k = -sum_j p_ij * (d d_ij/dx_k)
    # Objective = -sum S_i, so grad = -sum_i dS_i/dx_k = sum_i sum_j p_ij * (d d_ij/dx_k)
    
    grad = np.zeros((n, 2))
    
    # Inter-circle contributions: for each pair (i,j), contribution to grad[i] and grad[j]
    # d d_ij/dx_i = (x_i - x_j) / (2 * d_ij) where d_ij = 0.5 * ||c_i - c_j||
    # Actually d_ij = 0.5 * dist, so d d_ij/dx_i = (x_i - x_j) / (2 * dist_ij)
    # Let's be careful: d_ij = 0.5 * ||c_i - c_j||
    # d d_ij / d x_i = 0.5 * (x_i - x_j) / ||c_i - c_j||
    
    dist_matrix = 2.0 * dists  # full distances (not halved)
    np.fill_diagonal(dist_matrix, 1.0)  # avoid div by zero
    
    # For each i, sum over j: p_ij * (c_i - c_j) / (2 * dist_ij)
    # This is the gradient of d_ij w.r.t. c_i
    # grad_inter[i] = sum_j p_ij * (c_i - c_j) / (2 * dist_ij)
    
    # Vectorized: weight matrix P (n,n), diff matrix D (n,n,2)
    # grad_inter[i] = sum_j P[i,j] * (c_i - c_j) / (2 * dist_ij)
    
    # (c_i - c_j) / (2 * dist_ij) = diff[i,j] / (2 * dist_matrix[i,j])
    # where diff[i,j] = c_i - c_j
    
    inv_dist = 1.0 / (2.0 * dist_matrix)  # (n, n)
    inv_dist[np.eye(n, dtype=bool)] = 0.0
    
    # grad_inter[i] = sum_j P[i,j] * diff[i,j] * inv_dist[i,j]
    # diff has shape (n, n, 2), P and inv_dist have shape (n, n)
    weighted_diff = probs[:, :, np.newaxis] * diff * inv_dist[:, :, np.newaxis]
    grad_inter = np.sum(weighted_diff, axis=1)  # (n, 2)
    
    # Wall contributions for circle i:
    # d d_left/dx_i = 1, d d_right/dx_i = -1, d d_bottom/dy_i = 1, d d_top/dy_i = -1
    # grad_wall[i,0] = p_i,n+0 * 1 + p_i,n+1 * (-1) = p_left - p_right
    # grad_wall[i,1] = p_i,n+2 * 1 + p_i,n+3 * (-1) = p_bottom - p_top
    
    p_left = probs[:, n]
    p_right = probs[:, n+1]
    p_bottom = probs[:, n+2]
    p_top = probs[:, n+3]
    
    grad_wall = np.column_stack([p_left - p_right, p_bottom - p_top])
    
    # Total gradient of sum(soft_mins) w.r.t. centers
    grad_sum = grad_inter + grad_wall
    
    # Objective is -sum(soft_mins), so gradient is -grad_sum
    grad = -grad_sum.flatten()
    
    return value, grad

def _epigraph_objective(params, W, H, beta):
    """Epigraph formulation: maximize sum of t_i subject to t_i <= soft_min(dists_i).
    
    params = [x_1, y_1, ..., x_N, y_N, t_1, t_2, ..., t_N]
    We minimize -sum(t_i) subject to t_i <= soft_min(distances_i)
    
    For smoothness, we use: t_i + (1/beta) * log(sum(exp(beta*(d_j - t_i)))) <= 0
    which is equivalent to t_i <= soft_min(d_j) in the smooth sense.
    """
    n = N
    centers = params[:2*n].reshape(n, 2)
    t = params[2*n:]
    
    # Objective: minimize -sum(t)
    obj = -np.sum(t)
    
    # Constraint: for each i, t_i <= soft_min of distances to walls and half-distances to others
    # soft_min(d) = min(d) - (1/beta)*log(sum(exp(-beta*(d - min(d)))))
    # Equivalent: t_i + (1/beta)*log(sum(exp(beta*(d_j - t_i)))) <= 0
    # But this is complex. Instead, let's use a penalty approach.
    
    # Actually, let's just return the objective for L-BFGS-B with bounds on t
    # The constraints will be handled by the bounds: t_i in [0, max_possible]
    # And we'll add a penalty for violating t_i > soft_min(dist_i)
    
    penalty = 0.0
    for i in range(n):
        d_walls = np.array([centers[i, 0], W - centers[i,0],
                           centers[i, 1], H - centers[i,1]])
        dists = np.linalg.norm(centers[i] - centers, axis=1) / 2.0
        dists[i] = np.inf
        all_dists = np.concatenate([d_walls, dists])
        
        # soft_min
        m = np.min(all_dists)
        exp_terms = np.exp(-beta * (all_dists - m))
        soft_min_val = m - (1.0/beta) * np.log(np.sum(exp_terms))
        
        # Penalty if t_i > soft_min_val
        if t[i] > soft_min_val:
            penalty += (t[i] - soft_min_val)**2
    
    return obj + penalty * 1e4

def _voronoi_inradii(centers, W, H):
    """Compute the inradius of each circle's Voronoi cell (vectorized)."""
    n = len(centers)
    # Distance to walls for all circles at once
    d_left = centers[:, 0]
    d_right = W - centers[:, 0]
    d_bottom = centers[:, 1]
    d_top = H - centers[:, 1]
    wall_dists = np.column_stack([d_left, d_right, d_bottom, d_top])
    
    # Half-distance to other circles
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2)) / 2.0
    np.fill_diagonal(dists, np.inf)
    
    # For each circle, min of wall dists and inter-circle half-dists
    all_dists = np.concatenate([wall_dists, dists], axis=1)
    radii = np.min(all_dists, axis=1)
    return radii

def _soft_min(arr, beta):
    """Numerically stable soft-min approximation."""
    m = np.min(arr)
    exp_terms = np.exp(-beta * (arr - m))
    return m - (1.0 / beta) * np.log(np.sum(exp_terms))

def _soft_min_sum(centers, W, H, beta):
    """Compute sum of soft-min approximations of radii for all circles (vectorized)."""
    n = len(centers)
    # Distance to walls for all circles
    d_left = centers[:, 0]
    d_right = W - centers[:, 0]
    d_bottom = centers[:, 1]
    d_top = H - centers[:, 1]
    wall_dists = np.column_stack([d_left, d_right, d_bottom, d_top])
    
    # Half-distance to other circles
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2)) / 2.0
    np.fill_diagonal(dists, np.inf)
    
    all_dists = np.concatenate([wall_dists, dists], axis=1)
    
    # Soft-min for each circle
    mins = np.min(all_dists, axis=1)
    exp_terms = np.exp(-beta * (all_dists - mins[:, np.newaxis]))
    log_sums = np.log(np.sum(exp_terms, axis=1))
    soft_mins = mins - (1.0 / beta) * log_sums
    
    return np.sum(soft_mins)

def _objective(centers_flat, W, H, beta):
    """Negative sum of soft-min radii (to minimize)."""
    centers = centers_flat.reshape(N, 2)
    val, _ = _soft_min_sum_and_grad(centers, W, H, beta)
    return -val

def _objective_grad(centers_flat, W, H, beta):
    """Gradient of negative sum of soft-min radii."""
    centers = centers_flat.reshape(N, 2)
    _, grad = _soft_min_sum_and_grad(centers, W, H, beta)
    return grad

def _seed_hex(W, H, seed):
    """Generate initial center positions using a hexagonal-like pattern."""
    np.random.seed(seed)
    
    # Hexagonal packing: rows offset
    # Estimate rows and columns
    aspect = W / H
    if aspect >= 1:
        n_cols = max(2, int(np.sqrt(N * aspect)))
        n_rows = max(1, int(np.ceil(N / n_cols)))
    else:
        n_rows = max(2, int(np.sqrt(N / aspect)))
        n_cols = max(1, int(np.ceil(N / n_rows)))
    
    centers = []
    for row in range(n_rows):
        for col in range(n_cols):
            if len(centers) >= N:
                break
            x = (col + 0.5) * W / n_cols
            y = (row + 0.5) * H / n_rows
            # Offset every other row for hexagonal pattern
            if row % 2 == 1:
                x += W / (2 * n_cols)
            x = np.clip(x, 0.05, W - 0.05)
            y = np.clip(y, 0.05, H - 0.05)
            centers.append([x, y])
    
    centers = np.array(centers[:N])
    
    # If we don't have enough, add random points
    if len(centers) < N:
        extra = np.random.uniform(0.05, W - 0.05, (N - len(centers), 2))
        extra[:, 1] = np.random.uniform(0.05, H - 0.05, N - len(centers))
        centers = np.vstack([centers, extra])
    
    # Small random perturbation to break symmetry
    centers += np.random.normal(0, 0.02, centers.shape)
    centers[:, 0] = np.clip(centers[:, 0], 0.01, W - 0.01)
    centers[:, 1] = np.clip(centers[:, 1], 0.01, H - 0.01)
    return centers

def _seed_grid(W, H, seed):
    """Generate initial center positions using a grid with perturbation."""
    np.random.seed(seed)
    n_x = max(2, int(np.sqrt(N * W / H)))
    n_y = max(1, int(np.ceil(N / n_x)))
    
    x_coords = np.linspace(0.1, W - 0.1, n_x + 2)[1:-1] if n_x > 1 else np.array([W/2])
    y_coords = np.linspace(0.1, H - 0.1, n_y + 2)[1:-1] if n_y > 1 else np.array([H/2])
    
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    centers = np.column_stack((grid_x.flatten(), grid_y.flatten()))
    
    if len(centers) < N:
        extra_x = np.linspace(0.1, W - 0.1, N - len(centers) + 1)
        extra_y = np.linspace(0.1, H - 0.1, N - len(centers) + 1)
        extra = np.column_stack((extra_x, extra_y))
        centers = np.vstack([centers, extra])
    
    centers = centers[:N]
    centers += np.random.normal(0, 0.03, centers.shape)
    centers[:, 0] = np.clip(centers[:, 0], 0.01, W - 0.01)
    centers[:, 1] = np.clip(centers[:, 1], 0.01, H - 0.01)
    return centers

def _simulated_annealing_step(centers, W, H, T, n_steps, rng):
    """Perform SA-like exploration by perturbing a random subset of circles."""
    n = N
    best = centers.copy()
    best_score = np.sum(_voronoi_inradii(best, W, H))
    current = centers.copy()
    current_score = best_score
    
    for step in range(n_steps):
        # Pick 1-3 random circles to perturb
        n_perturb = rng.randint(1, 4)
        idx = rng.choice(n, size=n_perturb, replace=False)
        
        new_centers = current.copy()
        for i in idx:
            # Gaussian perturbation scaled by temperature
            sigma = T * 0.1
            new_centers[i, 0] += rng.normal(0, sigma)
            new_centers[i, 1] += rng.normal(0, sigma)
            new_centers[i, 0] = np.clip(new_centers[i, 0], 0.01, W - 0.01)
            new_centers[i, 1] = np.clip(new_centers[i, 1], 0.01, H - 0.01)
        
        new_score = np.sum(_voronoi_inradii(new_centers, W, H))
        
        # Accept if better or with probability exp(-(new_score - current_score)/T)
        if new_score >= current_score:
            current = new_centers
            current_score = new_score
        else:
            delta = new_score - current_score
            if rng.random() < np.exp(delta / max(T, 1e-6)):
                current = new_centers
                current_score = new_score
        
        if current_score > best_score:
            best_score = current_score
            best = current.copy()
        
        # Cool down
        T *= 0.95
    
    return best, best_score

def _optimize(W, H, initial_centers, deadline):
    """Multi-stage optimization with increasing beta, SA exploration, and final polish."""
    bounds = [(0, W)] * N + [(0, H)] * N
    
    current_centers = initial_centers.copy()
    best_centers = current_centers.copy()
    best_score = np.sum(_voronoi_inradii(current_centers, W, H))
    
    rng = np.random.default_rng(12345)
    
    # Multi-stage beta schedule with SA between stages
    beta_schedule = [10.0, 25.0, 50.0, 100.0, 200.0, 400.0]
    max_iters_schedule = [100, 150, 200, 150, 150, 100]
    ftol_schedule = [1e-10, 1e-12, 1e-12, 1e-14, 1e-14, 1e-14]
    gtol_schedule = [1e-8, 1e-10, 1e-10, 1e-12, 1e-12, 1e-12]
    
    for stage, (beta, max_iter, ftol, gtol) in enumerate(
        zip(beta_schedule, max_iters_schedule, ftol_schedule, gtol_schedule)
    ):
        if time.time() > deadline - 8:  # Leave 8s margin
            break
        
        # L-BFGS-B with analytic gradient
        try:
            res = minimize(
                _objective, current_centers.flatten(),
                jac=_objective_grad,
                args=(W, H, beta),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
            )
            current_centers = res.x.reshape(N, 2)
        except Exception:
            break
        
        # Evaluate exact score
        score = np.sum(_voronoi_inradii(current_centers, W, H))
        if score > best_score:
            best_score = score
            best_centers = current_centers.copy()
        
        # SA exploration between stages (except last)
        if stage < len(beta_schedule) - 1 and time.time() < deadline - 10:
            try:
                T = 0.05 * (1.0 - stage * 0.1)  # Decreasing temperature
                n_sa_steps = 50
                sa_centers, sa_score = _simulated_annealing_step(
                    best_centers, W, H, T, n_sa_steps, rng
                )
                if sa_score > best_score:
                    best_score = sa_score
                    best_centers = sa_centers.copy()
                # Also try SA from current
                sa_centers2, sa_score2 = _simulated_annealing_step(
                    current_centers, W, H, T, n_sa_steps, rng
                )
                if sa_score2 > best_score:
                    best_score = sa_score2
                    best_centers = sa_centers2.copy()
            except Exception:
                pass
    
    # Final high-precision polish: use very high beta with tight tolerances
    if time.time() < deadline - 3:
        try:
            beta_final = 800.0
            res_final = minimize(
                _objective, best_centers.flatten(),
                jac=_objective_grad,
                args=(W, H, beta_final),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            final_centers = res_final.x.reshape(N, 2)
            final_score = np.sum(_voronoi_inradii(final_centers, W, H))
            if final_score > best_score:
                best_score = final_score
                best_centers = final_centers
        except Exception:
            pass
    
    return best_centers, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    best_score = -np.inf
    best_config = None

    # Search over rectangle widths (perimeter = 4 => H = 2 - W)
    # Include more width values for better coverage
    widths = np.array([0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2])
    
    # Use both grid and hexagonal seeds
    seed_funcs = [_seed_grid, _seed_hex]
    
    n_restarts_per_seed = 4

    for W in widths:
        if time.time() > deadline:
            break
        H = 2.0 - W

        for seed_func in seed_funcs:
            for restart in range(n_restarts_per_seed):
                if time.time() > deadline - 10:  # Leave 10s for final evaluations
                    break

                seed = 42 + restart * 10 + (0 if seed_func == _seed_grid else 5)
                centers = seed_func(W, H, seed=seed)
                
                try:
                    final_centers, score = _optimize(W, H, centers, deadline)
                    
                    if score > best_score:
                        best_score = score
                        radii = _voronoi_inradii(final_centers, W, H)
                        best_config = np.column_stack((final_centers, radii))
                        
                except Exception:
                    pass
        
        # After finishing all seeds for this width, try a second pass on best config
        # with a different random seed for SA to potentially improve further
        if best_config is not None and time.time() < deadline - 15:
            try:
                # Re-optimize the best config with a fresh SA exploration
                best_c = best_config[:, :2]
                best_c, new_score = _optimize(W, H, best_c, deadline)
                if new_score > best_score:
                    best_score = new_score
                    radii = _voronoi_inradii(best_c, W, H)
                    best_config = np.column_stack((best_c, radii))
            except Exception:
                pass

    # Fallback if no valid result
    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_config
