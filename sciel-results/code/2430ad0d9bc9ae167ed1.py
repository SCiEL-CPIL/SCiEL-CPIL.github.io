"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def _compute_radius(c):
    """Compute the valid radius for a configuration."""
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check bounds
    if np.any(c < -0.5) or np.any(c > 0.5):
        return 0.0
    
    # Min distance to walls
    wall_dists = np.min(np.minimum(0.5 - np.abs(c), 0.5 + np.abs(c)), axis=1)
    r_wall = np.min(wall_dists)
    
    # Min distance between pairs
    # Use a simple loop or vectorized for N=50 (1225 pairs)
    min_pair_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(c[i] - c[j])
            if d < min_pair_dist:
                min_pair_dist = d
    
    r_pair = min_pair_dist / 2.0
    return min(r_wall, r_pair)

def _energy(c, beta):
    """
    Smooth energy function:
    1. Log-sum-exp of inverse distances (pairwise repulsion)
    2. Log-sum-exp of inverse distances to walls (wall repulsion)
    """
    c = c.reshape(-1, 3)
    
    # Pairwise terms: -log(sum(exp(-beta * dist))) approximates -beta * min_dist
    # We want to maximize min_dist, so we minimize -min_dist.
    # E_pair = -log(sum(exp(-beta * d_ij)))
    # For large beta, this approximates -beta * min(d_ij).
    
    d = 1.0 / beta
    # Vectorized pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid log(0) or division by zero for i==j (d=0)
    # We only care about i != j
    mask = ~np.eye(N, dtype=bool)
    dists = dists[mask]
    
    # Stability: shift by max
    m = np.max(dists)
    e = np.exp(-dists / d)
    # If d is small, exp(-d/d) = exp(-1) approx. 
    # Actually, let's use the standard log-sum-exp trick:
    # log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
    # Here x_i = -d_i / d. So max(x) = -min(d)/d.
    # We want E = -log(sum(exp(-d_i/d))) = min(d)/d - log(sum(exp(-d_i/d + min(d)/d)))
    # Let's just compute directly for N=50 (1225 elements)
    
    # To avoid underflow/overflow issues with large beta, we can scale.
    # But for beta ~ 50-100, d ~ 0.01-0.02. 
    # exp(-100 * 0.1) is very small.
    # The dominant term is the min distance.
    
    # Let's use a numerically stable logsumexp
    # logsumexp(-dists/d)
    # = -min(dists)/d + log(sum(exp(-(dists - min(dists))/d)))
    
    min_d = np.min(dists)
    # If min_d is 0, we have overlap. Penalty should be huge.
    # If min_d is 0, dists - min_d has 0. exp(0)=1.
    
    shifted = (dists - min_d) / d
    # Clip to avoid overflow in exp if d is very small? 
    # dists >= min_d, so shifted >= 0.
    # exp(shifted) can be large if d is small.
    # But sum of exps is bounded by N^2 * exp(max_shift).
    # max_shift = (max_d - min_d)/d.
    # If d=0.01, max_d~1, shift~100. exp(100) is huge but finite in float64?
    # exp(709) is max. So d should not be too small.
    # Beta < 700 should be safe.
    
    log_sum = np.log(np.sum(np.exp(shifted)))
    e_pair = -(min_d / d) + log_sum # Wait.
    
    # Let's re-derive.
    # We want to MINIMIZE energy.
    # We want to MAXIMIZE min_d.
    # f(min_d) = -min_d.
    # Approximation: -log(sum(exp(-beta * d_i)))
    # Let S = sum(exp(-beta * d_i)).
    # log(S) = log(exp(-beta*min_d) * sum(exp(-beta*(d_i-min_d))))
    #        = -beta*min_d + log(sum(exp(-beta*(d_i-min_d))))
    # So -log(S) = beta*min_d - log(sum(exp(-beta*(d_i-min_d))))
    # As beta -> inf, the sum is dominated by the term where d_i is min.
    # sum(exp(0) + small terms) ~ 1 + small. log(1+small) ~ small.
    # So -log(S) ~ beta*min_d.
    # Minimizing -log(S) is equivalent to minimizing min_d? NO.
    # Minimizing -log(S) means making -log(S) small (negative).
    # -log(S) is negative if S > 1.
    # If d_i are large, exp(-beta*d_i) is small, S < 1, -log(S) > 0.
    # If d_i are small, exp(-beta*d_i) is large, S > 1, -log(S) < 0.
    # So we want S to be large? No, we want min_d to be large.
    # If min_d is large, S is small, -log(S) is large positive.
    # So MINIMIZING -log(S) leads to SMALL min_d. That's wrong.
    
    # We want to MAXIMIZE min_d.
    # So we should MINIMIZE log(S)?
    # If min_d is large, S is small, log(S) is large negative.
    # Minimizing log(S) leads to more negative values -> larger min_d.
    # Yes!
    
    # So Energy_pair = log(sum(exp(-beta * d_i)))
    # = -beta*min_d + log(sum(exp(-beta*(d_i-min_d))))
    
    # Let's compute this.
    # log_sum = -min_d/d + log(sum(exp(-(dists - min_d)/d)))
    
    e_pair = -min_d / d + log_sum
    
    # Wall terms
    # Distance to nearest wall for each point
    # d_wall_i = min(0.5 - x, x+0.5, 0.5-y, y+0.5, 0.5-z, z+0.5)
    # Actually, distance to boundary of [-0.5, 0.5]
    # d_wall = min(0.5 - c, c + 0.5) for each coordinate, then min over coords.
    
    d_walls = np.min(np.minimum(0.5 - c, 0.5 + c), axis=1)
    
    # Same logic: Energy_wall = log(sum(exp(-beta * d_wall_i)))
    min_dw = np.min(d_walls)
    shifted_w = (d_walls - min_dw) / d
    log_sum_w = np.log(np.sum(np.exp(shifted_w)))
    e_wall = -min_dw / d + log_sum_w
    
    return e_pair + e_wall

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # Initialization: Jittered grid
    # 50 points. 3x3x6 = 54. Or 3x4x4 = 48 (add 2).
    # Let's use a 3x4x4 grid + 2 random points?
    # Or just 50 random points? Random is bad.
    # Grid is good.
    
    # 3x4x4 grid
    xs = np.linspace(-0.4, 0.4, 3)
    ys = np.linspace(-0.4, 0.4, 4)
    zs = np.linspace(-0.4, 0.4, 4)
    
    grid_points = []
    for z in zs:
        for y in ys:
            for x in xs:
                grid_points.append([x, y, z])
                
    # We have 48 points. Add 2 more.
    # Place them in the gaps?
    # Or just start with 50 random points near the grid?
    # Let's use a 5x5x2 grid? 50 points exactly.
    # 5x5x2 = 50.
    # Side 1. 5 points in x, 5 in y, 2 in z.
    # Spacing: 1/5 = 0.2 in x, 1/5 = 0.2 in y, 1/2 = 0.5 in z.
    # This is very anisotropic.
    
    # Better: 3x4x5 = 60. Too many.
    # 4x4x3 = 48.
    # Let's stick to 48 grid points + 2 random points inside.
    
    rng = np.random.default_rng(42)
    
    c_init = np.array(grid_points[:48])
    # Add 2 random points
    c_rand = rng.uniform(-0.45, 0.45, size=(2, 3))
    c_init = np.vstack([c_init, c_rand])
    
    # Jitter to break symmetry
    c_init += rng.normal(0, 0.01, size=c_init.shape)
    
    # Clamp to bounds
    c_init = np.clip(c_init, -0.49, 0.49)
    
    best_c = c_init.copy()
    best_r = _compute_radius(best_c)
    
    # Optimization
    # Start with small beta, increase it.
    betas = [5, 10, 20, 50, 100, 200]
    
    current_c = c_init.copy()
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
            
        # Define function for L-BFGS-B
        def fun(x):
            return _energy(x, beta)
        
        def jac(x):
            # Numerical gradient is slow. Let's use analytic?
            # Or use approx_grad.
            # For N=50, 150 vars. Approx grad is 150 function evals.
            # Function eval is ~100us? 150 * 100us = 15ms per step.
            # L-BFGS-B might take 100 steps. 1.5s per beta.
            # 6 betas = 9s. Acceptable.
            # But we can try analytic gradient.
            # E = log(sum(exp(-beta*d))) + log(sum(exp(-beta*d_w)))
            # d = ||c_i - c_j||
            # d/d(c_ik) d = (c_ik - c_jk) / d
            # dE/d(c_ik) = sum_{j != i} ( exp(-beta*d_ij) * (-beta) * (c_ik - c_jk)/d_ij ) / sum(exp(-beta*d_ij))
            #             + (for wall terms)
            
            # Let's stick to approx_grad for robustness, but it might be slow.
            # Let's try to implement analytic gradient.
            
            x = x.reshape(-1, 3)
            grad = np.zeros_like(x)
            
            # Pairwise
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            
            # Mask diagonal
            mask = ~np.eye(N, dtype=bool)
            
            # Weights
            # w_ij = exp(-beta * d_ij) / sum(exp(-beta * d_kl))
            # Note: sum is over all pairs k<l.
            # The gradient for point i involves sum over j.
            # Let S_pair = sum_{k<l} exp(-beta * d_kl)
            # dE_pair/d(c_ik) = sum_{j != i} [ -beta * exp(-beta * d_ij) / S_pair * (c_ik - c_jk) / d_ij ]
            
            # Compute S_pair
            # To be efficient, we can compute the matrix of exp terms.
            exp_terms = np.exp(-beta * dists)
            np.fill_diagonal(exp_terms, 0)
            
            # S_pair is sum of upper triangle?
            # The sum in the log is over all i<j.
            # sum_{i<j} exp(-beta*d_ij).
            # This is 0.5 * sum_{i != j} exp(-beta*d_ij).
            S_pair = 0.5 * np.sum(exp_terms)
            
            # Gradient contribution from pair i,j to i:
            # term_ij = -beta * exp(-beta * d_ij) / S_pair * (c_i - c_j) / d_ij
            # Sum over j.
            
            # Vectorized:
            # grad_pair[i] = sum_j term_ij
            # term matrix G where G[i,j] = -beta * exp(-beta * d_ij) / S_pair * (c_i - c_j) / d_ij
            # Note: G[i,j] = -G[j,i]
            
            # Handle d=0
            safe_dists = np.where(dists > 1e-8, dists, 1e-8)
            
            # Factor = -beta * exp(-beta * d) / S_pair / d
            factor = -beta * exp_terms / S_pair / safe_dists
            np.fill_diagonal(factor, 0)
            
            # grad_pair = sum_j factor[i,j] * (c_i - c_j)
            # c_i - c_j = diff[i,j]
            # grad_pair[i] = sum_j factor[i,j] * diff[i,j]
            
            # factor is (N,N), diff is (N,N,3)
            # grad_pair = np.einsum('ij,ijk->k', factor, diff) ? No.
            # grad_pair[i,k] = sum_j factor[i,j] * diff[i,j,k]
            grad_pair = np.einsum('ij,ijk->ik', factor, diff)
            
            # Wall terms
            # d_wall_i = min(0.5 - x, x+0.5, ...)
            # This is non-differentiable at the center.
            # For smoothness, we can use a soft-min for walls too?
            # Or just use the gradient of the active constraint.
            # If c_ik is the min, grad is +/- beta * exp(-beta * d_wall) / S_wall
            # Let's use a soft approximation for walls in the energy, but analytic grad for hard min is tricky.
            # Let's just use numerical gradient for walls? Or include walls in the pairwise logic?
            # No, walls are different.
            
            # Let's just use numerical gradient for the whole thing if analytic is too complex.
            # But numerical is slow.
            # Let's try to implement analytic for walls.
            # E_wall = log(sum_i exp(-beta * d_wall_i))
            # S_wall = sum_i exp(-beta * d_wall_i)
            # dE_wall/d(c_ik) = (1/S_wall) * sum_i exp(-beta * d_wall_i) * d(d_wall_i)/d(c_ik)
            # d(d_wall_i)/d(c_ik) is 0 except for the coordinate that defines the min distance.
            # If c_ik is close to 0.5, d_wall = 0.5 - c_ik, deriv = -1.
            # If c_ik is close to -0.5, d_wall = c_ik + 0.5, deriv = 1.
            # If in the middle, deriv = 0 (or undefined).
            
            # We can compute the active wall for each point.
            # d_walls = np.min(np.minimum(0.5 - x, 0.5 + x), axis=1)
            # active_coord: which coordinate and which side?
            
            # Let's just use numerical gradient for the whole function.
            # It's safer.
            return None # Force approx_grad
            
        # Use approx_grad
        res = opt.minimize(fun, current_c, method='L-BFGS-B', jac='2-point',
                           options={'maxiter': 100, 'gtol': 1e-6, 'maxfun': 500})
        
        if res.success or res.fun < _energy(current_c, beta):
            # Check validity
            c_new = res.x
            if np.all(c_new > -0.5) and np.all(c_new < 0.5):
                r_new = _compute_radius(c_new)
                if r_new > best_r:
                    best_r = r_new
                    best_c = c_new.copy()
                # Update current_c for next beta
                current_c = c_new
        
    # Final check
    if _compute_radius(best_c) > best_r:
        best_r = _compute_radius(best_c)
        
    return best_c

# EVOLVE-BLOCK-END
