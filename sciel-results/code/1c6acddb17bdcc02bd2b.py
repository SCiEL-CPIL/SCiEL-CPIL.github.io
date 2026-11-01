"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _compute_radius(centers):
    """Compute the maximum common radius for a given set of centers."""
    # Distance to walls
    # Centers are in [-0.5, 0.5]
    # Distance to wall for center c is min(|c_x|, |c_y|, |c_z|) because wall is at +/- 0.5
    # Wait, if cube is [-0.5, 0.5], distance to wall x=-0.5 is x - (-0.5) = x + 0.5.
    # Distance to wall x=0.5 is 0.5 - x.
    # So min distance to any wall is min(0.5 - c_x, 0.5 + c_x, 0.5 - c_y, 0.5 + c_y, 0.5 - c_z, 0.5 + c_z)
    # This is equivalent to 0.5 - max(|c_x|, |c_y|, |c_z|)
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists, axis=1)
    
    # The radius for sphere i is half the min of its wall dist and min pair dist
    # r_i = min(wall_dist_i, min_pair_dist_i / 2)
    radii = np.minimum(wall_dists, min_pair_dist / 2.0)
    
    return np.min(radii)

def _logsumexp(x, axis=None):
    """Numerically stable log-sum-exp."""
    x_max = np.max(x, axis=axis, keepdims=True)
    # Handle case where x is all -inf (should not happen with positive distances)
    x_max = np.where(x_max == -np.inf, 0, x_max)
    return np.log(np.sum(np.exp(x - x_max), axis=axis)) + np.squeeze(x_max, axis=axis)

def _smooth_min(x, beta):
    """
    Approximate the minimum of x using a log-sum-exp.
    min(x) ~= -1/beta * log(sum(exp(-beta * x)))
    As beta -> inf, this converges to min(x).
    """
    return -1.0 / beta * _logsumexp(-beta * x, axis=0)

def _objective_and_grad(flat_centers, beta):
    """
    Objective function to minimize: -1 / smooth_min(constraints)
    Actually, we want to maximize r. So we minimize -r.
    r = min_i (min(wall_dist_i, pair_dist_i/2))
    
    Let d_i be the vector of distances for sphere i (1 wall dist, N-1 pair dists).
    r_i = min(d_i)
    r = min(r_i)
    
    We use a two-level smoothing:
    1. Smooth min over distances for each sphere: r_i_smooth = -1/beta * log(sum(exp(-beta * d_i)))
    2. Smooth min over spheres: r_smooth = -1/beta * log(sum(exp(-beta * r_i_smooth)))
    
    To maximize r_smooth, we minimize -r_smooth.
    """
    centers = flat_centers.reshape((N, 3))
    
    # 1. Compute distances
    # Wall distances: 0.5 - max(|c|)
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # For each sphere i, we have a list of distances: [wall_dist_i, dists[i, 0]/2, ..., dists[i, i-1]/2, ...]
    # Note: Pairwise distance contribution to radius is dist/2.
    # So for sphere i, the relevant distances are:
    # d_i = [wall_dist_i, dists[i,:]/2] with diagonal removed.
    
    # Let's build a matrix D of shape (N, N) where D[i, j] is the effective distance constraint.
    # D[i, j] = wall_dist_i if j == 0? No, let's just concatenate.
    
    # Efficiently:
    # Create a matrix of all pairwise half-distances
    pair_half_dists = dists / 2.0
    np.fill_diagonal(pair_half_dists, np.inf)
    
    # We need to combine wall_dist_i and pair_half_dists[i, :] for each i.
    # Let M be a matrix of shape (N, N+1).
    # M[i, 0] = wall_dist_i
    # M[i, 1:] = pair_half_dists[i, :]
    
    M = np.zeros((N, N + 1))
    M[:, 0] = wall_dists
    M[:, 1:] = pair_half_dists
    
    # 1. Smooth min over columns (distances) for each row (sphere)
    # r_i_smooth = -1/beta * log(sum_j exp(-beta * M[i, j]))
    # We can do this vectorized.
    # Note: M contains inf on diagonal for pairs, but wall dist is finite.
    # exp(-beta * inf) = 0.
    
    # To avoid overflow/underflow issues in exp, we use the stable logsumexp logic manually or rely on the helper.
    # However, _logsumexp takes an axis.
    
    # Let's compute r_i_smooth
    # We want min_j M[i,j].
    # r_i_smooth = -1/beta * logsumexp(-beta * M[i, :])
    
    # Vectorized logsumexp along axis 1
    # x = -beta * M
    x = -beta * M
    # max along axis 1
    x_max = np.max(x, axis=1, keepdims=True)
    # Handle -inf
    x_max = np.where(x_max == -np.inf, 0, x_max)
    # logsumexp
    lse = np.log(np.sum(np.exp(x - x_max), axis=1)) + np.squeeze(x_max, axis=1)
    r_i_smooth = -1.0 / beta * lse
    
    # 2. Smooth min over spheres
    # r_smooth = -1/beta * logsumexp(-beta * r_i_smooth)
    y = -beta * r_i_smooth
    y_max = np.max(y, keepdims=True)
    y_max = np.where(y_max == -np.inf, 0, y_max)
    lse_r = np.log(np.sum(np.exp(y - y_max))) + np.squeeze(y_max)
    r_smooth = -1.0 / beta * lse_r
    
    # Objective to minimize: -r_smooth
    return -r_smooth

def _grad_objective(flat_centers, beta):
    """
    Gradient of the objective function.
    f = -r_smooth
    r_smooth = -1/beta * log(sum_k exp(-beta * r_k_smooth))
    r_k_smooth = -1/beta * log(sum_j exp(-beta * d_kj))
    
    Let S_k = sum_j exp(-beta * d_kj)
    r_k = -1/beta * log(S_k)
    
    Let T = sum_k exp(-beta * r_k)
    r = -1/beta * log(T)
    
    f = -r = 1/beta * log(T)
    
    df/dflat = df/dr * dr/dr_k * dr_k/dd_kj * dd_kj/dflat
    
    1. df/dr = 1/beta * (1/T) * dT/dr
       dT/dr_k = T_k * (-beta) where T_k = exp(-beta * r_k)
       So dT/dr = sum_k (-beta * T_k)
       df/dr = 1/beta * (1/T) * (-beta * sum T_k) = - sum(T_k) / T
       Let w_k = T_k / T. This is a probability distribution over spheres.
       df/dr = - sum(w_k) = -1. 
       Wait. f = -r. dr/dr_k is positive? 
       If r_k increases, T increases? No, T = sum exp(-beta r_k). If r_k increases, exp(-beta r_k) decreases. T decreases.
       log(T) decreases. r = -1/beta log(T) increases.
       So dr/dr_k > 0.
       
       Let's re-derive carefully.
       r = -1/beta * log( sum_k exp(-beta r_k) )
       Let U = sum_k exp(-beta r_k).
       r = -1/beta ( log U )
       dr/dr_k = -1/beta * (1/U) * dU/dr_k
       dU/dr_k = exp(-beta r_k) * (-beta) = -beta exp(-beta r_k)
       dr/dr_k = -1/beta * (1/U) * (-beta exp(-beta r_k)) = exp(-beta r_k) / U
       Let w_k = exp(-beta r_k) / U. Note sum(w_k) = 1.
       So dr/dr_k = w_k.
       
       f = -r.
       df/dr_k = -w_k.
       
    2. dr_k/d d_kj
       r_k = -1/beta * log( sum_j exp(-beta d_kj) )
       Let V_k = sum_j exp(-beta d_kj).
       dr_k/dd_kj = -1/beta * (1/V_k) * dV_k/dd_kj
       dV_k/dd_kj = exp(-beta d_kj) * (-beta) = -beta exp(-beta d_kj)
       dr_k/dd_kj = -1/beta * (1/V_k) * (-beta exp(-beta d_kj)) = exp(-beta d_kj) / V_k
       Let v_kj = exp(-beta d_kj) / V_k. Note sum_j(v_kj) = 1.
       
    3. Chain rule:
       df/d d_kj = df/dr_k * dr_k/d d_kj = -w_k * v_kj
       
    4. dd_kj/dflat
       d_kj is either wall_dist_k or pair_dist_kj/2.
       
       Case A: j == 0 (Wall)
       d_k0 = 0.5 - max(|c_kx|, |c_ky|, |c_kz|)
       Let m_k = max(|c_kx|, |c_ky|, |c_kz|)
       d_k0 = 0.5 - m_k
       dd_k0/dc_kx = -sign(c_kx) if |c_kx| == m_k else 0
       Similarly for y, z.
       
       Case B: j > 0 (Pair with sphere l = j-1)
       d_kj = 0.5 * ||c_k - c_l||
       dd_kj/dc_k = 0.5 * (c_k - c_l) / ||c_k - c_l||
       dd_kj/dc_l = 0.5 * (c_l - c_k) / ||c_k - c_l||
       
    We need to accumulate gradients for all flat_centers.
    """
    centers = flat_centers.reshape((N, 3))
    
    # Compute distances again
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_half_dists = dists / 2.0
    np.fill_diagonal(pair_half_dists, np.inf)
    
    M = np.zeros((N, N + 1))
    M[:, 0] = wall_dists
    M[:, 1:] = pair_half_dists
    
    # Compute weights v_kj (prob over distances for sphere k)
    x = -beta * M
    x_max = np.max(x, axis=1, keepdims=True)
    x_max = np.where(x_max == -np.inf, 0, x_max)
    exp_x = np.exp(x - x_max)
    V_k = np.sum(exp_x, axis=1, keepdims=True)
    # v_kj = exp_x / V_k
    v = exp_x / V_k  # Shape (N, N+1)
    
    # Compute r_k_smooth
    r_k = -1.0 / beta * (np.log(np.squeeze(V_k, axis=1)) + np.squeeze(x_max, axis=1))
    
    # Compute weights w_k (prob over spheres)
    y = -beta * r_k
    y_max = np.max(y, keepdims=True)
    y_max = np.where(y_max == -np.inf, 0, y_max)
    exp_y = np.exp(y - y_max)
    U = np.sum(exp_y, keepdims=True)
    w = exp_y / U  # Shape (N, 1)
    
    # Combine weights: weight for distance d_kj is -w_k * v_kj
    # grad_d[k, j] = -w[k, 0] * v[k, j]
    grad_d = -w * v  # Shape (N, N+1)
    
    # Accumulate gradients into flat_centers gradient
    grad_flat = np.zeros_like(flat_centers)
    
    # 1. Wall gradients (j=0)
    # d_k0 = 0.5 - max(|c_k|)
    # dd_k0/dc_k = -sign(c_k) if argmax(|c_k|) == c_k
    abs_centers = np.abs(centers)
    max_idx = np.argmax(abs_centers, axis=1)
    
    for k in range(N):
        g_wall = grad_d[k, 0]
        if g_wall != 0:
            dim = max_idx[k]
            c_val = centers[k, dim]
            # d(0.5 - |c|)/dc = -sign(c)
            grad_flat[k * 3 + dim] += g_wall * (-np.sign(c_val))
    
    # 2. Pair gradients (j > 0)
    # d_kj corresponds to pair (k, l) where l = j - 1
    # Note: M[k, j] is dist between k and l=j-1.
    # We iterate over k and j.
    # To be efficient, we can vectorize, but N=50 is small.
    for k in range(N):
        for j in range(1, N + 1):
            l = j - 1
            if k == l:
                continue # Diagonal is inf, weight should be 0
            g_pair = grad_d[k, j]
            if g_pair != 0:
                diff_vec = centers[k] - centers[l]
                dist_val = np.linalg.norm(diff_vec)
                if dist_val > 1e-9:
                    # dd_kj/dc_k = 0.5 * diff_vec / dist_val
                    grad_vec = 0.5 * diff_vec / dist_val
                    grad_flat[k * 3 : (k + 1) * 3] += g_pair * grad_vec
                    # dd_kj/dc_l = 0.5 * (c_l - c_k) / dist_val = - grad_vec
                    grad_flat[l * 3 : (l + 1) * 3] += g_pair * (-grad_vec)
    
    return grad_flat

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initial Configuration: Stratified random or Grid
    # Grid is a good start
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Take first 50
    centers = grid[:N].copy()
    
    # Add small noise
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.01, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Annealing beta values
    # Start low (smooth) to find basin, end high (sharp) to refine
    betas = [10.0, 50.0, 200.0, 1000.0, 5000.0]
    
    current_centers = centers.copy()
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        flat_centers = current_centers.flatten()
        
        # L-BFGS-B optimization
        # Bounds: each coordinate in [-0.5, 0.5]
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(
                _objective_and_grad,
                flat_centers,
                jac=_grad_objective,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            new_centers = res.x.reshape((N, 3))
            
            # Ensure validity
            if np.all(np.isfinite(new_centers)):
                current_radius = _compute_radius(new_centers)
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_centers = new_centers.copy()
                    current_centers = new_centers.copy()
                else:
                    # Even if not better than global best, keep current_centers if it's valid 
                    # and better than the start of this beta phase?
                    # L-BFGS-B minimizes -r_smooth. If it converged, it's a local min of the smooth obj.
                    # We should probably keep the result if it's valid, as it's closer to the sharp min.
                    if np.all(np.isfinite(new_centers)):
                        current_centers = new_centers.copy()
                        
        except Exception as e:
            # If optimization fails, keep previous
            pass
            
    # Final refinement with high beta if time permits
    if time.time() < deadline - 5:
        beta = 10000.0
        flat_centers = current_centers.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        try:
            res = minimize(
                _objective_and_grad,
                flat_centers,
                jac=_grad_objective,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            new_centers = res.x.reshape((N, 3))
            if np.all(np.isfinite(new_centers)):
                current_radius = _compute_radius(new_centers)
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_centers = new_centers.copy()
        except:
            pass

    return best_centers
# EVOLVE-BLOCK-END
