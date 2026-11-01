"""
Improved sphere packing using Hybrid Simulated Annealing and Soft-Min L-BFGS.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_MARGIN = 5.0  # seconds before the 150s limit
TOTAL_TIME_LIMIT = 150.0

# Precompute pair indices for efficiency
iu = np.triu_indices(N, k=1)
n_pairs = len(iu[0])

def _compute_radius(pts):
    """
    Compute the exact largest common radius for given centers.
    Returns 0.0 if invalid.
    """
    if pts.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    # Wall distances
    # Distance from center to nearest wall
    wall_dists = 0.5 - np.abs(pts)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # Use broadcasting for pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    r = min(min_wall, min_pair / 2.0)
    return r

def _soft_min_obj_grad(x, beta):
    """
    Objective function and gradient for the soft-min approximation.
    We want to maximize the minimum margin (radius).
    Equivalent to minimizing the soft-min of the negative margins.
    
    Margins:
    - For each pair (i,j): d_ij / 2
    - For each point i, each axis k: 0.5 - |c_ik|
    
    We define G(x) = (1/beta) * log( sum( exp(-beta * margin_k) ) )
    Minimizing G(x) maximizes min(margin_k).
    """
    pts = x.reshape(N, 3)
    
    # 1. Wall margins
    # margin_w = 0.5 - |c|
    # d(margin_w)/d(c) = -sign(c)
    wall_margins = 0.5 - np.abs(pts)
    wall_margins_flat = wall_margins.flatten()
    
    # 2. Pair margins
    # margin_p = dist / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12) # Add epsilon for stability
    np.fill_diagonal(dists, np.inf)
    
    pair_margins = dists / 2.0
    pair_margins_flat = pair_margins[iu]
    
    # Combine all margins
    all_margins = np.concatenate([wall_margins_flat, pair_margins_flat])
    
    # Compute soft-min
    # To avoid overflow in exp, subtract max
    neg_beta_margins = -beta * all_margins
    max_neg = np.max(neg_beta_margins)
    exp_vals = np.exp(neg_beta_margins - max_neg)
    sum_exp = np.sum(exp_vals)
    
    # Objective value
    obj = (max_neg + np.log(sum_exp)) / beta
    
    # Weights for gradient
    weights = exp_vals / sum_exp
    
    # Gradient calculation
    grad = np.zeros_like(pts)
    
    # Wall gradient contribution
    # dG/d(wall_margin_k) = -weights[k]
    # d(wall_margin_k)/d(c_i) = -sign(c_i)
    # So dG/d(c_i) += -weights[k] * (-sign(c_i)) = weights[k] * sign(c_i)
    wall_weights = weights[:3*N].reshape(N, 3)
    grad += wall_weights * np.sign(pts)
    
    # Pair gradient contribution
    # dG/d(pair_margin_ij) = -weights[idx]
    # d(pair_margin_ij)/d(c_i) = (c_i - c_j) / dist_ij
    # d(pair_margin_ij)/d(c_j) = (c_j - c_i) / dist_ij
    # So dG/d(c_i) += -weights[idx] * (c_i - c_j) / dist_ij
    
    pair_weights = weights[3*N:]
    
    # We need to sum contributions for each point i
    # For each pair (i,j), i < j:
    # grad[i] -= pair_weights[idx] * (pts[i] - pts[j]) / dists[i,j]
    # grad[j] -= pair_weights[idx] * (pts[j] - pts[i]) / dists[i,j]
    
    ii, jj = iu
    # Vectorized calculation
    # diff[i,j] = pts[i] - pts[j]
    # We have dists matrix.
    # Let's construct the force vector
    
    # Create a matrix of unit vectors
    # u[i,j] = (pts[i] - pts[j]) / dists[i,j]
    # We only need upper triangle.
    
    # To avoid O(N^2) memory allocation for the full gradient update in a naive way,
    # we can loop or use sparse accumulation. N=100 is small enough for a loop or simple vectorization.
    
    # Vectorized approach:
    # diff is (N, N, 3)
    # dists is (N, N)
    
    # We want to sum w_ij * u_ij for each i
    # Let's create a weight matrix W where W[i,j] = pair_weights[idx] if i<j else 0
    # But we can just iterate over the pairs since N=100, pairs=4950.
    # Iteration is fast enough in Python if we don't do heavy math per iter, but vectorization is better.
    
    # Let's try vectorized accumulation.
    # grad_pair = - sum_j w_ij * (pts[i] - pts[j]) / dists[i,j]
    
    # Create full weight matrix
    W = np.zeros((N, N))
    W[ii, jj] = pair_weights
    
    # Unit vectors
    # dists has inf on diagonal, so 1/dists is 0 there.
    # diff / dists[:, :, None]
    # Be careful with dists=0 (coincident points). We added 1e-12 so it's safe.
    unit_diff = diff / dists[:, :, np.newaxis]
    
    # grad[i] = - sum_j W[i,j] * unit_diff[i,j]
    # Note: W is symmetric if we fill both, but we only have upper triangle weights.
    # The gradient for point i comes from all j.
    # For i < j: term is -w_ij * (pts[i]-pts[j])/d_ij added to grad[i]
    #           and -w_ij * (pts[j]-pts[i])/d_ij added to grad[j] = +w_ij * (pts[i]-pts[j])/d_ij
    # So if we define F_ij = w_ij * (pts[i]-pts[j])/d_ij, then:
    # grad[i] -= F_ij
    # grad[j] += F_ij
    
    # Let's compute F for all i,j
    # F[i,j] = W[i,j] * unit_diff[i,j]
    F = W[:, :, np.newaxis] * unit_diff
    
    # grad[i] = - sum_j F[i,j]
    grad -= np.sum(F, axis=1)
    
    # Wait, the symmetry check:
    # F[j,i] should be related to F[i,j].
    # W[j,i] is 0 in our current W (only upper triangle filled).
    # So this simple sum won't work directly because W is not symmetric.
    
    # Let's make W symmetric.
    W_sym = W + W.T
    # Now W_sym[i,j] = w_ij for i!=j.
    # F_sym[i,j] = W_sym[i,j] * unit_diff[i,j]
    # F_sym[j,i] = W_sym[j,i] * unit_diff[j,i] = w_ij * (-unit_diff[i,j]) = -F_sym[i,j]
    
    # grad[i] = - sum_j F_sym[i,j]
    # Let's verify:
    # grad[i] = - sum_{j != i} w_ij * (pts[i]-pts[j])/d_ij
    # This matches the derivation: dG/dc_i = - sum_j w_ij * d(pair_margin)/dc_i = - sum_j w_ij * (pts[i]-pts[j])/d_ij
    
    F_sym = W_sym[:, :, np.newaxis] * unit_diff
    grad -= np.sum(F_sym, axis=1)
    
    return obj, grad.flatten()

def _local_refine(x0, beta, deadline):
    """
    Run L-BFGS-B to refine the configuration.
    """
    bounds = [(-0.5, 0.5)] * (3 * N)
    
    try:
        res = minimize(
            _soft_min_obj_grad,
            x0,
            args=(beta,),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        return np.clip(res.x, -0.5, 0.5)
    except Exception:
        return x0

def _simulated_annealing_loop(start_pts, deadline):
    """
    Main search loop combining SA and L-BFGS.
    """
    current_pts = start_pts.copy()
    best_pts = start_pts.copy()
    best_radius = _compute_radius(best_pts)
    
    # Initial refinement to get a good baseline
    current_pts = _local_refine(current_pts.flatten(), beta=20.0, deadline=deadline)
    current_pts = current_pts.reshape(N, 3)
    r_curr = _compute_radius(current_pts)
    if r_curr > best_radius:
        best_radius = r_curr
        best_pts = current_pts.copy()
    
    # SA Parameters
    # We will increase beta (steepness) over time
    # Start with low beta (smooth landscape), end with high beta (sharp landscape)
    beta_start = 10.0
    beta_end = 5000.0
    
    # Temperature for SA acceptance
    # We accept if new_obj < old_obj or with prob exp(-delta/T)
    # As beta increases, the objective becomes more "hard"
    
    # Schedule
    n_steps = 500 # Total SA iterations
    # We check deadline frequently
    
    rng = np.random.default_rng(42)
    
    # We will iterate, gradually increasing beta
    # and doing SA moves
    
    # To be efficient, we will group steps into "epochs" with fixed beta
    n_epochs = 10
    steps_per_epoch = n_steps // n_epochs
    
    for epoch in range(n_epochs):
        if time.time() > deadline:
            break
            
        # Interpolate beta
        progress = epoch / float(n_epochs)
        beta = beta_start * (beta_end / beta_start) ** progress
        
        # Temperature decay
        # T should be related to the scale of the objective
        # Objective scale is roughly 1/beta * log(N)
        # Let's just use a heuristic T
        T = 0.05 * (1.0 - progress * 0.9) + 0.001
        
        # Perturbation scale
        # Should be small enough to stay in basin, large enough to jump
        sigma = 0.02 * (1.0 - progress * 0.8) + 0.002
        
        # Get current objective
        obj_curr, _ = _soft_min_obj_grad(current_pts.flatten(), beta)
        
        for step in range(steps_per_epoch):
            if time.time() > deadline:
                break
                
            # Generate candidate
            perturbation = rng.normal(0, sigma, current_pts.shape)
            cand_pts = np.clip(current_pts + perturbation, -0.5, 0.5)
            
            # Evaluate candidate
            try:
                obj_cand, _ = _soft_min_obj_grad(cand_pts.flatten(), beta)
            except Exception:
                continue
            
            delta = obj_cand - obj_curr
            
            # Accept?
            if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
                current_pts = cand_pts
                obj_curr = obj_cand
                
                # Check radius
                r_cand = _compute_radius(cand_pts)
                if r_cand > best_radius:
                    best_radius = r_cand
                    best_pts = cand_pts.copy()
            
            # Every 5 steps, do a short local refinement to polish
            if step % 5 == 4:
                # Use a slightly higher beta for refinement to focus on min
                ref_beta = beta * 2.0
                refined_pts = _local_refine(current_pts.flatten(), ref_beta, deadline)
                refined_pts = refined_pts.reshape(N, 3)
                
                r_ref = _compute_radius(refined_pts)
                if r_ref > best_radius:
                    best_radius = r_ref
                    best_pts = refined_pts.copy()
                
                # Update current state to refined if it's better in objective
                obj_ref, _ = _soft_min_obj_grad(refined_pts.flatten(), beta)
                if obj_ref < obj_curr:
                    current_pts = refined_pts
                    obj_curr = obj_ref
    
    # Final high-beta refinement on the best solution found
    if time.time() < deadline - 5:
        final_beta = 10000.0
        final_pts = _local_refine(best_pts.flatten(), final_beta, deadline)
        final_pts = final_pts.reshape(N, 3)
        r_final = _compute_radius(final_pts)
        if r_final > best_radius:
            best_radius = r_final
            best_pts = final_pts
    
    return best_pts

def _generate_initial_grid():
    """
    Generate a good initial grid.
    For N=100, a 5x5x4 grid is 100 points.
    Spacing: 1/5 = 0.2, 1/4 = 0.25.
    Radius approx min(0.1, 0.125) = 0.1.
    This is a decent start.
    """
    # Try 5x5x4
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.5 + 1.0/(2*nx), 0.5 - 1.0/(2*nx), nx)
    ys = np.linspace(-0.5 + 1.0/(2*ny), 0.5 - 1.0/(2*ny), ny)
    zs = np.linspace(-0.5 + 1.0/(2*nz), 0.5 - 1.0/(2*nz), nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Add small noise to break symmetry
    rng = np.random.default_rng(123)
    pts += rng.normal(0, 0.005, pts.shape)
    pts = np.clip(pts, -0.49, 0.49)
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + TOTAL_TIME_LIMIT - DEADLINE_MARGIN
    
    # 1. Initial Grid
    pts_init = _generate_initial_grid()
    
    # 2. Run Hybrid SA + L-BFGS
    best_pts = _simulated_annealing_loop(pts_init, deadline)
    
    # 3. One last check and clip
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts

# Test run to ensure it doesn't crash immediately
if __name__ == "__main__":
    pts = construct()
    r = _compute_radius(pts)
    print(f"Radius: {r}")
