"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0

    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Ensure centers are strictly inside for validity check in score, 
    # though the optimizer handles constraints via bounds.
    if min_wall_dist < 0:
        return 0.0

    # Pairwise distances
    # Using cKDTree for efficiency in this evaluation function
    tree = cKDTree(centers)
    # Query pairs with a large distance to ensure we get the minimum
    # For N=100, full pairwise is fast enough, but let's stick to efficient method
    # Actually, for N=100, cdist is very fast. Let's use cdist for simplicity and vectorization.
    from scipy.spatial.distance import cdist
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)

    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective_and_grad(x, beta):
    """
    Objective function and gradient for L-BFGS-B.
    Minimizes the soft-min approximation of the radius.
    Returns (obj, grad).
    """
    centers = x.reshape(N, 3)
    
    # 1. Wall distances: d_i = 0.5 - |c_i|
    # We want to maximize min(d_i), so we minimize -min(d_i) approx.
    # Soft min of values v is approx: (1/beta) * ln(sum(exp(-beta * v)))
    # Here v = dist_to_wall. We want to minimize the soft-min of these distances.
    # Wait, the score is r = min(min_wall, min_pair/2).
    # We want to maximize r. So we minimize -r.
    # Let's define the "constraints" as distances.
    # Wall distances: w_i = 0.5 - |c_i|
    # Pair distances: p_ij = ||c_i - c_j|| / 2
    # We want to maximize min(all w_i, all p_ij).
    # This is equivalent to minimizing the soft-min of these values.
    
    # Compute wall distances
    # w_i = 0.5 - max(|x|, |y|, |z|) is NOT correct.
    # The distance to the nearest wall for point (x,y,z) in [-0.5, 0.5] is:
    # min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y, 0.5 - z, 0.5 + z)
    # Which is 0.5 - max(|x|, |y|, |z|).
    # So w_i = 0.5 - np.max(np.abs(centers[i]), axis=1)
    
    abs_centers = np.abs(centers)
    max_coord = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_coord
    
    # Compute pairwise distances / 2
    # Use cdist for vectorization
    from scipy.spatial.distance import cdist
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Extract unique pairs to avoid double counting and diagonal
    iu = np.triu_indices(N, k=1)
    pair_vals = pair_dists[iu]
    wall_vals = wall_dists
    
    # Combine all distances
    all_vals = np.concatenate([wall_vals, pair_vals])
    
    # Soft-min: (1/beta) * ln(sum(exp(-beta * v)))
    # We want to minimize this value.
    # As beta -> inf, this approaches min(all_vals).
    
    # Numerical stability
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    exp_sum = np.sum(np.exp(neg_beta_x - max_val))
    
    if exp_sum == 0:
        obj = 0.0
        grad = np.zeros_like(x)
        return obj, grad
        
    obj = (max_val + np.log(exp_sum)) / beta
    
    # Gradient calculation
    # d/dbeta is not needed, we fix beta.
    # d/xc_j [ (1/beta) ln(sum(exp(-beta v_k))) ]
    # = (1/beta) * (1/sum(exp)) * sum( d(exp(-beta v_k))/dx * exp(-beta v_k) )
    # = (1/sum(exp)) * sum( -beta * (dv_k/dx) * exp(-beta v_k) )
    # = -beta * sum( (dv_k/dx) * p_k )  where p_k = exp(-beta v_k) / sum(exp)
    
    weights = np.exp(neg_beta_x - max_val) / exp_sum
    
    # Gradient with respect to centers
    grad_centers = np.zeros_like(centers)
    
    # Wall gradient
    # w_i = 0.5 - max(|x|, |y|, |z|)
    # dw_i/dx_j = -sign(x_j) if x_j is the max coordinate, else 0
    # Actually, it's a subgradient. For differentiability, we can use a soft-max or just pick the argmax.
    # Let's use the argmax coordinate for the gradient contribution.
    argmax_idx = np.argmax(abs_centers, axis=1)
    
    for i in range(N):
        w_idx = i
        w_val = all_vals[w_idx]
        weight = weights[w_idx]
        
        # dw/dx
        # w = 0.5 - max(|c|)
        # d(0.5 - max(|c|))/dc = - sign(c_argmax) * 1 (if unique max)
        # If there are ties, the subgradient is a convex combination. 
        # For L-BFGS, a single subgradient is usually sufficient.
        ax = argmax_idx[i]
        grad_centers[i, ax] -= weight * np.sign(centers[i, ax]) * beta
        
    # Pair gradient
    # p_ij = ||c_i - c_j|| / 2
    # dp_ij/dc_i = (c_i - c_j) / (2 * ||c_i - c_j||) * beta? No.
    # Let's re-derive:
    # Term in sum is exp(-beta * p_ij).
    # d/dx (exp(-beta p)) = exp(-beta p) * (-beta * dp/dx)
    # Contribution to grad is: sum_k weights_k * (-beta * dp_k/dx)
    # weights_k = exp(-beta p_k) / sum(exp(-beta p))
    # So grad = -beta * sum_k (dp_k/dx * exp(-beta p_k) / sum(exp))
    
    # We have weights for all pairs in `pair_vals`.
    # We need to map these back to indices.
    # iu[0] and iu[1] give the row and col indices for the upper triangle.
    # pair_vals[k] corresponds to pair (iu[0][k], iu[1][k])
    
    # Vectorized gradient update for pairs
    # dp_ij/dc_i = (c_i - c_j) / (2 * dist_ij)
    # dp_ij/dc_j = (c_j - c_i) / (2 * dist_ij)
    
    # We only have dists, not the vectors.
    # diffs = centers[iu[0]] - centers[iu[1]]
    # dists_unique = np.sqrt(np.sum(diffs**2, axis=1))
    # unit_vecs = diffs / dists_unique[:, np.newaxis]
    
    # grad_pair_i += -beta * weight * unit_vec / 2
    # grad_pair_j += -beta * weight * (-unit_vec) / 2 = beta * weight * unit_vec / 2
    
    # Wait, p_ij = dist / 2.
    # dp/dc_i = (c_i - c_j) / (2 * dist)
    # So d(exp)/dc_i = exp * (-beta) * (c_i - c_j) / (2 * dist)
    
    # Let's compute this vectorized.
    i_idx, j_idx = iu
    diffs = centers[i_idx] - centers[j_idx]
    dists_unique = np.sqrt(np.sum(diffs**2, axis=1))
    
    # Avoid division by zero
    mask = dists_unique > 1e-12
    unit_vecs = np.zeros_like(diffs)
    unit_vecs[mask] = diffs[mask] / dists_unique[mask, np.newaxis]
    
    # weights_pair = weights[N:] (since first N are wall dists)
    weights_pair = weights[N:]
    
    # grad contribution for i: -beta * weight * unit_vec / 2
    # grad contribution for j: +beta * weight * unit_vec / 2
    
    grad_contrib_i = -0.5 * beta * weights_pair[:, np.newaxis] * unit_vecs
    grad_contrib_j = 0.5 * beta * weights_pair[:, np.newaxis] * unit_vecs
    
    # Scatter back
    np.add.at(grad_centers, i_idx, grad_contrib_i)
    np.add.at(grad_centers, j_idx, grad_contrib_j)
    
    # Total gradient is grad_centers flattened
    grad = grad_centers.flatten()
    
    return obj, grad

def _build_halton_init(n=100, seed=42):
    """
    Build a Halton sequence based initialization for good space-filling properties.
    """
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    
    rng = np.random.default_rng(seed)
    centers = np.zeros((n, 3))
    for i in range(1, n+1):
        centers[i-1, 0] = -0.5 + halton(i, 2)
        centers[i-1, 1] = -0.5 + halton(i, 3)
        centers[i-1, 2] = -0.5 + halton(i, 5)
    
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_grid_init(n=100, seed=42):
    """
    Build a simple cubic lattice initialization.
    """
    # 5x5x4 = 100
    nx, ny, nz = 5, 5, 4
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.003, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers[:n]

def _build_random_init(n=100, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-0.45, 0.45, (n, 3))
    return centers

def _optimize_l_bfgs(centers, beta, max_iter, deadline):
    """
    Run L-BFGS-B optimization.
    """
    x0 = centers.flatten()
    
    # Bounds: centers must be in [-0.5, 0.5]
    # To ensure spheres are inside, we technically need to be in [-0.5+r, 0.5-r]
    # But since r is a variable we are optimizing, we just constrain to [-0.5, 0.5]
    # and let the objective function penalize going too close to walls.
    # However, if a center goes outside [-0.5, 0.5], the wall distance becomes negative,
    # which is bad. The objective function handles this.
    # We use bounds [-0.5, 0.5] to keep points in the cube.
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    result = minimize(
        _soft_min_objective_and_grad,
        x0,
        args=(beta,),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={
            'maxiter': max_iter,
            'ftol': 1e-12,
            'gtol': 1e-8,
            'disp': False
        }
    )
    
    c_opt = result.x.reshape(N, 3)
    # Ensure finite
    if not np.all(np.isfinite(c_opt)):
        return centers, 0.0
        
    r_opt = _compute_radius(c_opt)
    return c_opt, r_opt

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s
    
    # Initializations
    inits = []
    inits.append(_build_grid_init(N, seed=42))
    inits.append(_build_halton_init(N, seed=42))
    inits.append(_build_random_init(N, seed=42))
    
    best_centres = None
    best_r = -1.0
    
    # Strategy:
    # 1. Run L-BFGS on each init with a moderate beta.
    # 2. Keep the best result.
    # 3. If time remains, polish the best result with a higher beta (sharper min) 
    #    or a few Nelder-Mead steps on the raw radius.
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
            
        # Moderate beta for initial convergence
        beta = 500.0
        max_iter = 200
        
        c_opt, r_opt = _optimize_l_bfgs(init, beta, max_iter, deadline)
        
        if r_opt > best_r:
            best_r = r_opt
            best_centres = c_opt.copy()
            
    # Polish phase:
    # If we have time, try to sharpen the minimum.
    if best_centres is not None and time.time() < deadline - 10.0:
        # Increase beta to make the soft-min closer to the true min
        beta_high = 2000.0
        max_iter_polish = 100
        
        c_polished, r_polished = _optimize_l_bfgs(best_centres, beta_high, max_iter_polish, deadline)
        
        if r_polished > best_r:
            best_r = r_polished
            best_centres = c_polished.copy()
            
        # Final Nelder-Mead on raw radius for local snapping
        if time.time() < deadline - 5.0:
            def neg_radius(x):
                c = x.reshape(N, 3)
                return -_compute_radius(c)
            
            x0 = best_centres.flatten()
            # Nelder-Mead is slow, use small maxiter
            res_nm = minimize(
                neg_radius,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 50, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            c_nm = res_nm.x.reshape(N, 3)
            r_nm = -res_nm.fun
            
            if r_nm > best_r:
                best_r = r_nm
                best_centres = c_nm.copy()
                
    # Final validation and cleanup
    if best_centres is None:
        # Fallback
        best_centres = _build_grid_init(N, seed=42)
        
    # Ensure no non-finite values
    if not np.all(np.isfinite(best_centres)):
        best_centres = _build_grid_init(N, seed=42)
        
    # Clip to valid range just in case
    best_centres = np.clip(best_centres, -0.5, 0.5)
    
    return best_centres

# EVOLVE-BLOCK-END
