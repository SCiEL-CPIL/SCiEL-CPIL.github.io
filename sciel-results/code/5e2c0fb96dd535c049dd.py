import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _energy_and_grad(x, beta=50.0):
    """
    x: shape (N, 3)
    Returns energy and gradient using log-sum-exp soft-min.
    """
    x = x.reshape(N, 3)
    
    # Compute pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero on diagonal
    mask = ~np.eye(N, dtype=bool)
    dist_sq_masked = np.where(mask, dist_sq, 1e10)
    dists = np.sqrt(dist_sq_masked)
    dists = np.where(mask, dists, 1e10)
    
    # Wall distances
    d_left = x + 0.5
    d_right = 0.5 - x
    wall_dists = np.concatenate([d_left.flatten(), d_right.flatten()])
    
    # All distances (pairs + walls)
    pair_dists = dists[mask]
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Soft-min via log-sum-exp
    min_d = np.min(all_dists)
    shifted = -beta * (all_dists - min_d)
    exp_shifted = np.exp(shifted)
    log_sum = np.log(np.sum(exp_shifted)) + beta * min_d
    energy = -log_sum
    
    # Gradient weights
    weights = exp_shifted / np.sum(exp_shifted)
    
    # Pair gradients (vectorized)
    pair_weights = weights[:len(pair_dists)]
    
    # Create index arrays
    i_idx, j_idx = np.where(mask)
    
    # Compute weighted differences
    weighted_diff = pair_weights[:, np.newaxis] * diff[i_idx, j_idx] / dists[i_idx, j_idx, np.newaxis]
    
    # Sum contributions
    grad_pairs = np.zeros((N, 3))
    np.add.at(grad_pairs, i_idx, weighted_diff)
    np.add.at(grad_pairs, j_idx, -weighted_diff)
    
    # Wall gradients
    wall_weights = weights[len(pair_dists):]
    grad_walls = np.zeros((N, 3))
    
    # Left walls
    left_weights = wall_weights[:N*3].reshape(N, 3)
    grad_walls += left_weights
    
    # Right walls
    right_weights = wall_weights[N*3:].reshape(N, 3)
    grad_walls -= right_weights
    
    total_grad = grad_pairs + grad_walls
    
    return energy, total_grad.flatten()

def _calc_score(pts):
    """Calculate the actual score (radius) for a set of points."""
    if pts.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
        
    # Pair distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1e10)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    # Wall distances
    min_wall_dist = np.min(0.5 - np.abs(pts))
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _initial_grid():
    """Create a better initial grid: 5x5x4 arrangement."""
    x_ticks = np.linspace(-0.42, 0.42, 5)
    y_ticks = np.linspace(-0.42, 0.42, 5)
    z_ticks = np.linspace(-0.375, 0.375, 4)
    
    grid = np.array([(x, y, z) for x in x_ticks for y in y_ticks for z in z_ticks])
    return grid[:N]

def _simulated_annealing(x0, beta_start=20.0, beta_end=100.0, n_iter=500, 
                         t_start=0.02, t_end=0.001, deadline=None):
    """
    Simulated annealing with smooth energy.
    """
    x = x0.copy()
    best_x = x.copy()
    best_score = _calc_score(x)
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Current temperature and beta
        t = t_start * (t_end / t_start) ** (it / max(n_iter, 1))
        beta = beta_start * (beta_end / beta_start) ** (it / max(n_iter, 1))
        
        # Propose a move: random perturbation
        step_size = t * 0.5
        x_prop = x + np.random.normal(0, step_size, x.shape)
        x_prop = np.clip(x_prop, -0.49, 0.49)
        
        # Compute energies
        e_curr, _ = _energy_and_grad(x, beta)
        e_prop, _ = _energy_and_grad(x_prop, beta)
        
        delta_e = e_prop - e_curr
        
        # Accept if better or with probability exp(-delta_e / t)
        if delta_e < 0 or (t > 1e-10 and np.random.random() < np.exp(-delta_e / t)):
            x = x_prop
            score = _calc_score(x)
            if score > best_score:
                best_score = score
                best_x = x.copy()
    
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    # Initial grid
    grid = _initial_grid()
    best_x = grid.copy()
    best_score = _calc_score(best_x)
    
    np.random.seed(42)
    
    # Strategy 1: Simulated annealing with smooth energy
    if time.time() < deadline - 30:
        sa_x, sa_score = _simulated_annealing(
            grid, beta_start=20.0, beta_end=80.0, n_iter=300,
            t_start=0.03, t_end=0.005, deadline=deadline - 30
        )
        if sa_score > best_score:
            best_score = sa_score
            best_x = sa_x
    
    # Strategy 2: L-BFGS-B polish from best solution
    if time.time() < deadline - 10:
        def objective(x_flat):
            e, g = _energy_and_grad(x_flat, beta=100.0)
            return e, g
            
        bounds = [(-0.5, 0.5)] * 3 * N
        
        try:
            res = minimize(objective, best_x.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14})
            if res.success:
                new_x = res.x.reshape(N, 3)
                new_x = np.clip(new_x, -0.5, 0.5)
                new_score = _calc_score(new_x)
                if new_score > best_score:
                    best_score = new_score
                    best_x = new_x
        except Exception:
            pass
    
    # Strategy 3: Multiple restarts with jitter if time remains
    if time.time() < deadline - 20:
        jitter_levels = [0.01, 0.02, 0.03]
        for jitter in jitter_levels:
            if time.time() > deadline - 10:
                break
            
            x0 = grid + np.random.uniform(-jitter, jitter, grid.shape)
            x0 = np.clip(x0, -0.49, 0.49)
            
            # Quick SA
            if time.time() < deadline - 15:
                sa_x, sa_score = _simulated_annealing(
                    x0, beta_start=30.0, beta_end=100.0, n_iter=150,
                    t_start=0.02, t_end=0.003, deadline=deadline - 15
                )
                if sa_score > best_score:
                    best_score = sa_score
                    best_x = sa_x
            
            # Polish
            if time.time() < deadline - 5:
                try:
                    res = minimize(objective, best_x.flatten(), method='L-BFGS-B', jac=True,
                                   bounds=bounds,
                                   options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-14})
                    if res.success:
                        new_x = res.x.reshape(N, 3)
                        new_x = np.clip(new_x, -0.5, 0.5)
                        new_score = _calc_score(new_x)
                        if new_score > best_score:
                            best_score = new_score
                            best_x = new_x
                except Exception:
                    pass
    
    # Final high-precision polish
    if time.time() < deadline - 2:
        try:
            res_final = minimize(objective, best_x.flatten(), method='L-BFGS-B', jac=True,
                                 bounds=bounds,
                                 options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-16})
            if res_final.success:
                final_x = res_final.x.reshape(N, 3)
                final_x = np.clip(final_x, -0.5, 0.5)
                final_score = _calc_score(final_x)
                if final_score > best_score:
                    best_score = final_score
                    best_x = final_x
        except Exception:
            pass

    return best_x
