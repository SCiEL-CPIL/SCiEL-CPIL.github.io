"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

def _score(pts):
    """Compute the packing radius for a set of points."""
    n = len(pts)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = dists.min()
    # Wall distances
    wall_dists = (0.5 - np.abs(pts)).min(axis=1)
    min_wall = wall_dists.min()
    return min(min_pair_dist / 2.0, min_wall)

def _soft_min_penalty(pts, beta=200.0):
    """
    Differentiable penalty that approximates the negative of the minimum clearance.
    We want to maximize the minimum of {pair_dists/2, wall_dists}.
    Equivalently, minimize -min(...) which is max(-...).
    
    We'll use a soft-max on the negative clearances.
    Let c_i be the clearances (pair_dists/2 and wall_dists).
    We want to minimize the soft-max of -c_i, which approximates -min(c_i).
    
    Actually, let's define the objective to minimize as:
    f(pts) = soft_max(-clearances) 
    where clearances = [pair_dists/2, wall_dists]
    
    soft_max(x) = (1/beta) * log(sum(exp(beta * x)))
    
    This is smooth and approximates max(x) = -min(-x) = -min(clearances).
    So minimizing soft_max(-clearances) approximates minimizing -min(clearances) = maximizing min(clearances).
    """
    n = len(pts)
    # Pairwise clearances: dist/2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_clearances = dists / 2.0
    pair_clearances = pair_clearances[pair_clearances < np.inf]
    
    # Wall clearances
    wall_clearances = 0.5 - np.abs(pts)
    wall_clearances = wall_clearances.flatten()
    
    # All clearances
    all_clearances = np.concatenate([pair_clearances, wall_clearances])
    neg_clearances = -all_clearances
    
    # Soft max with stability
    max_neg = np.max(neg_clearances)
    if max_neg > 500:  # Avoid overflow
        exp_terms = np.exp(beta * (neg_clearances - max_neg))
        soft_max = max_neg + (1.0 / beta) * np.log(np.sum(exp_terms))
    else:
        exp_terms = np.exp(beta * neg_clearances)
        soft_max = (1.0 / beta) * np.log(np.sum(exp_terms))
    
    return soft_max

def _soft_min_penalty_grad(pts, beta=200.0):
    """Gradient of the soft-max penalty."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_clearances = dists / 2.0
    pair_mask = pair_clearances < np.inf
    pair_clearances_flat = pair_clearances[pair_mask]
    
    wall_clearances = (0.5 - np.abs(pts)).flatten()
    
    all_clearances = np.concatenate([pair_clearances_flat, wall_clearances])
    neg_clearances = -all_clearances
    
    max_neg = np.max(neg_clearances)
    shifted = neg_clearances - max_neg
    exp_terms = np.exp(beta * shifted)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    # Gradient of soft_max w.r.t. neg_clearances is weights
    # d(soft_max)/d(pts) = d(soft_max)/d(neg_clearances) * d(neg_clearances)/d(pts)
    # d(neg_clearances)/d(pts) = -d(all_clearances)/d(pts)
    
    grad = np.zeros_like(pts)
    
    # Wall gradient contribution
    wall_weights = weights[-n:]
    # wall_clearance_i = 0.5 - |x_i|, so d(wall_clearance_i)/dx_i = -sign(x_i)
    # d(neg_wall_i)/dx_i = sign(x_i)
    # d(soft_max)/dx_i = sum_j w_j * d(neg_c_j)/dx_i
    # For wall j corresponding to point i: d(neg_wall_i)/dx_i = sign(x_i)
    for i in range(n):
        grad[i] += wall_weights[i] * np.sign(pts[i])
    
    # Pair gradient contribution
    pair_weights = weights[:len(pair_clearances_flat)]
    # Reconstruct which pairs
    pair_idx = np.nonzero(pair_mask)
    for k in range(len(pair_idx[0])):
        i, j = pair_idx[0][k], pair_idx[1][k]
        w = pair_weights[k]
        # pair_clearance = dist/2
        # d(dist)/dx_i = (x_i - x_j)/dist
        # d(pair_clearance)/dx_i = (x_i - x_j)/(2*dist)
        # d(neg_pair)/dx_i = -(x_i - x_j)/(2*dist)
        d_vec = pts[i] - pts[j]
        dist_val = np.linalg.norm(d_vec)
        if dist_val > 1e-12:
            grad[i] += w * (-d_vec / (2.0 * dist_val))
            grad[j] += w * (d_vec / (2.0 * dist_val))
    
    return grad

def _initial_grid():
    """Create an initial grid-based configuration."""
    m = 4  # 4x4x4 = 64 slots, use 50
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _optimize_single(pts, max_iters=2000, beta_start=50.0, beta_end=500.0):
    """Optimize a single configuration using L-BFGS-B with annealed beta."""
    x0 = pts.flatten()
    
    # We'll do a simple annealing schedule
    n_points = len(pts)
    n_flat = n_points * 3
    
    # Phase 1: Lower beta for broader exploration
    res1 = minimize(
        lambda x: _soft_min_penalty(x.reshape(-1, 3), beta=beta_start),
        x0,
        jac=lambda x: _soft_min_penalty_grad(x.reshape(-1, 3), beta=beta_start).flatten(),
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5)] * n_flat,
        options={'maxiter': max_iters // 2, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    # Phase 2: Higher beta for sharper optimization
    x1 = res1.x
    res2 = minimize(
        lambda x: _soft_min_penalty(x.reshape(-1, 3), beta=beta_end),
        x1,
        jac=lambda x: _soft_min_penalty_grad(x.reshape(-1, 3), beta=beta_end).flatten(),
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5)] * n_flat,
        options={'maxiter': max_iters // 2, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    
    return res2.x.reshape(-1, 3)

def _perturb(pts, scale=0.05):
    """Add random perturbation."""
    return pts + np.random.normal(0, scale, size=pts.shape)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    np.random.seed(42)
    
    best_pts = _initial_grid()
    best_score = _score(best_pts)
    
    # Multiple restarts with different initializations and perturbations
    n_restarts = 8
    restart_scale = 0.03
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Create perturbed initial configuration
        if restart == 0:
            pts = best_pts.copy()
        else:
            pts = _initial_grid() + np.random.normal(0, restart_scale * (1 + restart * 0.5), size=(N, 3))
            # Clip to stay in bounds
            pts = np.clip(pts, -0.49, 0.49)
        
        # Optimize
        try:
            optimized = _optimize_single(pts, max_iters=1500)
            # Clip to ensure validity
            optimized = np.clip(optimized, -0.5, 0.5)
            
            score = _score(optimized)
            if score > best_score:
                best_score = score
                best_pts = optimized.copy()
        except Exception:
            continue
    
    # Final polish with higher beta
    if time.time() < deadline - 5.0:
        try:
            x0 = best_pts.flatten()
            n_flat = N * 3
            res_final = minimize(
                lambda x: _soft_min_penalty(x.reshape(-1, 3), beta=1000.0),
                x0,
                jac=lambda x: _soft_min_penalty_grad(x.reshape(-1, 3), beta=1000.0).flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * n_flat,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            final_pts = np.clip(res_final.x.reshape(-1, 3), -0.5, 0.5)
            final_score = _score(final_pts)
            if final_score > best_score:
                best_pts = final_pts
                best_score = final_score
        except Exception:
            pass
    
    # Ensure all points are strictly inside
    best_pts = np.clip(best_pts, -0.4999, 0.4999)
    
    return best_pts
# EVOLVE-BLOCK-END
