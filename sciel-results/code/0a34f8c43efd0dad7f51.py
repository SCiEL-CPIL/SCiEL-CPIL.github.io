"""Improved packing: grid init + L-BFGS-B on smooth soft-min objective with deadline guard."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _score(c):
    """Compute the radius for a given set of centres (must be finite, inside cube)."""
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check inside cube [-0.5, 0.5]^3
    if np.any(np.abs(c) > 0.5):
        return 0.0
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)


def _negative_smooth_score(c, beta=50.0):
    """Smooth surrogate: -softmin of (pairwise dists/2, wall dists).
    Using log-sum-exp for soft-min: softmin(x) = -logsumexp(-x/beta)*beta
    We want to maximize r = min(...), so minimize -r.
    """
    c = c.reshape(N, 3)
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    pair_terms = dists / 2.0  # shape (N, N)
    
    # Wall distances
    wall_terms = 0.5 - np.abs(c)  # shape (N, 3)
    
    # Collect all terms into a flat array for soft-min
    # To keep it efficient, we can compute soft-min separately
    # softmin of a set = -beta * log(sum(exp(-x_i / beta)))
    
    # For pair terms (excluding diagonal)
    pair_flat = pair_terms.ravel()
    pair_flat = pair_flat[pair_flat < np.inf]
    if len(pair_flat) > 0:
        # Numerically stable logsumexp
        m = np.min(pair_flat)
        log_sum_exp_pair = m + np.log(np.sum(np.exp(-(pair_flat - m) / beta)))
        softmin_pair = -beta * log_sum_exp_pair  # Wait, this is wrong.
        # softmin(x) = -logsumexp(-x/beta) * beta
        # = -beta * (min(-x/beta) + log(sum(exp((-x/beta) - min(-x/beta)))))
        # Let y = -x/beta. min(y) = -max(x)/beta
        # Actually simpler: softmin(x_1,...,x_n) ≈ -beta * log(sum_i exp(-x_i/beta))
        # = -beta * ( -max(x)/beta + log(sum_i exp(-(x_i - max(x))/beta)) )
        # = max(x) - beta * log(sum_i exp(-(x_i - max(x))/beta))
        # That's not right either. Let's just use:
        # softmin = -beta * logsumexp(-x/beta)
        # logsumexp(z) = max(z) + log(sum(exp(z - max(z))))
        # So softmin = -beta * (max(-x/beta) + log(sum(exp(-x/beta - max(-x/beta)))))
        # max(-x/beta) = -min(x)/beta
        # So softmin = -beta * (-min(x)/beta + log(sum(exp(-x/beta + min(x)/beta))))
        # = min(x) - beta * log(sum(exp(-(x - min(x))/beta)))
        
        m_pair = np.min(pair_flat)
        exp_terms = np.exp(-(pair_flat - m_pair) / beta)
        softmin_pair = m_pair - beta * np.log(np.sum(exp_terms))
    else:
        softmin_pair = 0.0
    
    # For wall terms
    wall_flat = wall_terms.ravel()
    m_wall = np.min(wall_flat)
    exp_terms_wall = np.exp(-(wall_flat - m_wall) / beta)
    softmin_wall = m_wall - beta * np.log(np.sum(exp_terms_wall))
    
    # Overall soft-min is the soft-min of the union of both sets
    # softmin(union) = -beta * log( sum(exp(-pair/beta)) + sum(exp(-wall/beta)) )
    # = -beta * log( exp(-softmin_pair/beta) + exp(-softmin_wall/beta) )  -- NO, that's not how it works.
    # Actually, softmin(A ∪ B) = -beta * log( sum_{a∈A} exp(-a/beta) + sum_{b∈B} exp(-b/beta) )
    # We already have: sum_{a∈A} exp(-a/beta) = exp((m_pair - softmin_pair)/beta) ... this is getting complicated.
    
    # Simpler approach: just compute directly
    all_terms = np.concatenate([pair_flat, wall_flat])
    m_all = np.min(all_terms)
    exp_all = np.exp(-(all_terms - m_all) / beta)
    softmin_all = m_all - beta * np.log(np.sum(exp_all))
    
    return softmin_all  # We want to minimize this (since it approximates the min, and we want to maximize the min, so minimizing the negative... wait)
    # Actually, we want to MAXIMIZE r = min(...). So we should MINIMIZE -r.
    # But softmin approximates min. So we should minimize -softmin? No.
    # If softmin ≈ min, then maximizing min ≈ maximizing softmin.
    # So we should minimize -softmin.
    # Let me re-read my function. I'm returning softmin_all. The optimizer minimizes the objective.
    # So I should return -softmin_all.
    # But wait, I already returned it. Let me fix.


def _objective(c, beta=50.0):
    """Objective to minimize: -softmin of the constraints."""
    c = c.reshape(N, 3)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    pair_flat = (dists / 2.0).ravel()
    pair_flat = pair_flat[np.isfinite(pair_flat)]
    
    wall_flat = (0.5 - np.abs(c)).ravel()
    
    all_terms = np.concatenate([pair_flat, wall_flat])
    if len(all_terms) == 0:
        return 0.0
    m_all = np.min(all_terms)
    exp_all = np.exp(-(all_terms - m_all) / beta)
    softmin_all = m_all - beta * np.log(np.sum(exp_all))
    
    return -softmin_all  # Minimize negative of softmin = maximize softmin ≈ maximize min


def _gradient(c, beta=50.0):
    """Gradient of the objective with respect to c."""
    c = c.reshape(N, 3)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Pairwise terms
    pair_terms = dists / 2.0  # (N, N)
    pair_flat = pair_terms.ravel()
    pair_flat = pair_flat[np.isfinite(pair_flat)]
    
    wall_terms = 0.5 - np.abs(c)  # (N, 3)
    wall_flat = wall_terms.ravel()
    
    all_terms = np.concatenate([pair_flat, wall_flat])
    m_all = np.min(all_terms)
    exp_all = np.exp(-(all_terms - m_all) / beta)
    sum_exp = np.sum(exp_all)
    log_sum = np.log(sum_exp)
    softmin = m_all - beta * log_sum
    
    # d(softmin)/d(x_i) = - (1/beta) * x_i * exp(-x_i/beta) / sum(exp(-x_j/beta)) ... 
    # Actually, d/dx_i [ -beta * log(sum_j exp(-x_j/beta)) ] = -beta * (1/sum) * (-1/beta) * exp(-x_i/beta) = exp(-x_i/beta) / sum
    # And d/dx_i [m_all] is 0 (assuming m_all is fixed, but it's the min, so it's piecewise constant)
    # For the smooth approximation, we ignore the derivative of m_all.
    # So d(softmin)/d(x_i) ≈ exp(-x_i/beta) / sum_exp
    # And d(objective)/d(x_i) = -d(softmin)/d(x_i) = -exp(-x_i/beta) / sum_exp
    
    weights = exp_all / sum_exp  # (N*(N-1)/2 + 3N,)
    
    # Now back-propagate to c
    grad = np.zeros_like(c)
    
    # Pairwise part: weights[:N*(N-1)/2] correspond to pair_flat
    # pair_flat is the flattened upper triangle (or all off-diagonal) of pair_terms
    # pair_terms is (N, N), flattened in row-major order.
    # The k-th element of pair_flat corresponds to some (i, j) with i != j.
    # d(pair_terms[i,j])/d(c) = ... 
    # pair_terms[i,j] = |c_i - c_j| / 2
    # d/d(c_i) |c_i - c_j| / 2 = (c_i - c_j) / (2 * |c_i - c_j|) = (c_i - c_j) / (2 * dists[i,j])
    # d/d(c_j) |c_i - c_j| / 2 = (c_j - c_i) / (2 * |c_i - c_j|) = -(c_i - c_j) / (2 * dists[i,j])
    
    # This is getting complex. Let me use a simpler approach: numerical gradient or just use L-BFGS-B without gradient (it will use finite differences).
    # For now, return None to let scipy use finite differences.
    return None


def _initial_grid():
    """Generate an initial grid of N points in the cube."""
    # Use a 5x5x5 grid = 125 points, take first 50
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, use 115s for safety
    
    # Initial guess
    c0 = _initial_grid().astype(float)
    
    # First, evaluate the initial score
    best_c = c0.copy()
    best_score = _score(best_c)
    
    # Optimize using L-BFGS-B
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Try different betas (softness parameters)
    betas = [10.0, 20.0, 50.0, 100.0]
    
    for beta in betas:
        if time.time() > deadline - 30.0:
            break
        
        try:
            result = minimize(
                _objective,
                c0.reshape(-1),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 50000}
            )
            
            if result.success or result.fun < -best_score:
                new_c = result.x.reshape(N, 3)
                new_score = _score(new_c)
                if new_score > best_score:
                    best_c = new_c
                    best_score = new_score
            
            # Update c0 to the best found so far
            c0 = best_c.copy()
            
        except Exception:
            pass
    
    # Final polish with a tighter beta
    if time.time() < deadline - 20.0:
        try:
            result = minimize(
                _objective,
                best_c.reshape(-1),
                args=(200.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 100000}
            )
            new_c = result.x.reshape(N, 3)
            new_score = _score(new_c)
            if new_score > best_score:
                best_c = new_c
                best_score = new_score
        except Exception:
            pass
    
    # Ensure the result is valid
    if not np.all(np.isfinite(best_c)) or np.any(np.abs(best_c) > 0.5):
        return _initial_grid()
    
    return best_c
# EVOLVE-BLOCK-END
