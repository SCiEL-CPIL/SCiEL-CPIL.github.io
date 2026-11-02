"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 145.0  # safety margin before 150s kill


def _hex_lattice_init(n):
    """Generate hexagonal lattice points and select n closest to center."""
    # Estimate spacing for ~n points in unit disk
    # Area of hex lattice per point ~ s^2 * sqrt(3)/2
    # Area of unit disk = pi
    # n * s^2 * sqrt(3)/2 ~ pi  =>  s ~ sqrt(2*pi / (n*sqrt(3)))
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    # Add margin
    s *= 0.95
    
    k = int(1.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0:
                points.append((x, y, r))
    
    points = np.array(points)
    # Sort by distance from center, take closest n
    idx = np.argsort(points[:, 2])
    selected = points[idx[:n]][:, :2]
    
    if len(selected) < n:
        # Fallback: fill remaining randomly
        remaining = n - len(selected)
        extra = np.random.randn(remaining, 2) * 0.3
        selected = np.vstack([selected, extra])
    
    # Add small random noise to break symmetry
    noise = np.random.randn(n, 2) * 0.001
    selected = selected + noise
    
    return selected


def _compute_score(pts):
    """Compute the packing score (minimum radius)."""
    n = len(pts)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Distance to boundary
    radii = np.sqrt(np.sum(pts**2, axis=1))
    min_boundary = 1.0 - np.max(radii)
    
    return min(min_dist / 2.0, min_boundary)


def _soft_min_objective_and_grad(pts, beta):
    """
    Compute negative soft-min objective and its gradient efficiently.
    Uses O(N) pairwise computation instead of O(N^2) matrix storage.
    Returns (value, grad).
    """
    n = len(pts)
    if n < 2:
        return 0.0, np.zeros_like(pts)
    
    # Pairwise distances - compute in chunks to save memory
    # For efficiency, we'll compute the soft-min and gradient together
    # using a two-pass approach: first compute all constraint values,
    # then compute weights, then accumulate gradients.
    
    # Compute pairwise distances and boundary distances
    # We need to be careful about memory for N=300 (44850 pairs + 300 bounds = 45150 constraints)
    
    # First pass: compute all constraint values
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Boundary distances
    radii_sq = np.sum(pts**2, axis=1)
    radii = np.sqrt(radii_sq)
    boundary_dists = 1.0 - radii
    
    # Extract upper triangle pairs
    ii, jj = np.triu_indices(n, k=1)
    pair_dists = dists[ii, jj]
    pair_vals = pair_dists / 2.0
    
    # Combine all constraint values
    all_vals = np.concatenate([pair_vals, boundary_dists])
    n_pairs = len(ii)
    
    # Clip very small values to avoid numerical issues
    all_vals = np.maximum(all_vals, 1e-15)
    
    # Compute log-sum-exp of -beta * all_vals
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    exp_vals = np.exp(neg_beta_x - max_val)
    sum_exp = np.sum(exp_vals)
    log_sum_exp = max_val + np.log(sum_exp)
    
    # Objective: -softmin = log_sum_exp / beta
    obj_val = log_sum_exp / beta
    
    # Compute weights
    weights = exp_vals / sum_exp
    w_pairs = weights[:n_pairs]
    w_bounds = weights[n_pairs:]
    
    # Gradient computation - O(N) accumulation
    grad = np.zeros_like(pts)
    
    # 1. Pairwise contributions
    # For each pair (i, j): contribution to grad[i] = -w_ij * (pts[i] - pts[j]) / (2 * d_ij)
    #                              contribution to grad[j] = -w_ij * (pts[j] - pts[i]) / (2 * d_ij)
    # We can accumulate this as: grad[i] += -w_ij * (pts[i] - pts[j]) / (2 * d_ij)
    
    # Vectorized accumulation using scatter-add
    # For each pair, we need to add to grad[i] and grad[j]
    # Let's use a different approach: compute the weighted sum of (pts[j] - pts[i]) for each i
    
    # Create arrays for vectorized operation
    safe_dists = np.maximum(pair_dists, 1e-10)
    w_over_d = w_pairs / (2.0 * safe_dists)
    
    # For each point i, sum over all j: w_over_d_ij * (pts[j] - pts[i])
    # This is equivalent to: for each i, sum_j A[i,j] * pts[j] - pts[i] * sum_j A[i,j]
    # where A[i,j] = w_over_d_ij
    
    # Build sparse-like accumulation
    # grad_pairs[i] = -pts[i] * (sum of w_over_d for pairs involving i) + sum_j (w_over_d_ij * pts[j])
    
    # Compute row sums and weighted sums
    row_sum = np.zeros(n)
    weighted_sum = np.zeros((n, 2))
    
    # For upper triangle pairs, add to both i and j
    np.add.at(row_sum, ii, w_over_d)
    np.add.at(row_sum, jj, w_over_d)
    
    # weighted_sum[i] += w_over_d * pts[j] for each pair (i,j)
    # weighted_sum[j] += w_over_d * pts[i] for each pair (i,j)
    np.add.at(weighted_sum, ii, w_over_d[:, np.newaxis] * pts[jj])
    np.add.at(weighted_sum, jj, w_over_d[:, np.newaxis] * pts[ii])
    
    grad_pairs = -pts * row_sum[:, np.newaxis] + weighted_sum
    
    # 2. Boundary contributions
    # grad_bounds[i] = w_bounds[i] * pts[i] / |pts[i]|
    safe_radii = np.maximum(radii, 1e-10)
    grad_bounds = w_bounds[:, np.newaxis] * pts / safe_radii[:, np.newaxis]
    
    grad = grad_pairs + grad_bounds
    
    return obj_val, grad


def _local_polish(pts, max_iters=300, tol=1e-11):
    """
    Local optimization using exact minimum with projected gradient descent.
    Focuses on improving the minimum constraint.
    """
    pts = pts.copy()
    n = len(pts)
    best_score = _compute_score(pts)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 3:
            break
        
        # Compute all constraint values
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.sqrt(np.sum(pts**2, axis=1))
        boundary_dists = 1.0 - radii
        
        # Find the minimum constraint
        ii, jj = np.triu_indices(n, k=1)
        pair_vals = dists[ii, jj] / 2.0
        all_vals = np.concatenate([pair_vals, boundary_dists])
        min_idx = np.argmin(all_vals)
        min_val = all_vals[min_idx]
        
        if min_val < 0:
            # Infeasible, need to fix
            break
        
        # Compute subgradient of -min
        grad = np.zeros_like(pts)
        
        if min_idx < len(ii):
            # Pair constraint
            i, j = ii[min_idx], jj[min_idx]
            d = dists[i, j]
            if d > 1e-10:
                diff_ij = pts[i] - pts[j]
                grad[i] -= diff_ij / (2 * d)
                grad[j] += diff_ij / (2 * d)
        else:
            # Boundary constraint
            i = min_idx - len(ii)
            r = radii[i]
            if r > 1e-10:
                grad[i] -= pts[i] / r
        
        # Gradient step
        lr = min(0.01, 1e-4 / (np.linalg.norm(grad) + 1e-10))
        pts_new = pts + lr * grad
        
        # Project back to feasible region (inside unit disk)
        radii_new = np.sqrt(np.sum(pts_new**2, axis=1))
        mask = radii_new >= 1.0
        if np.any(mask):
            pts_new[mask] *= (0.999 / radii_new[mask])[:, np.newaxis]
        
        new_score = _compute_score(pts_new)
        if new_score > best_score - tol:
            pts = pts_new
            best_score = new_score
        
        # Check convergence
        if np.linalg.norm(pts_new - pts) < 1e-12:
            break
    
    return pts, best_score


def construct():
    """Construct 300 circle centers in unit disk to maximize minimum radius."""
    start_time = time.time()
    
    # Initialize with hexagonal lattice
    pts = _hex_lattice_init(N)
    
    # Ensure valid initial points (inside unit disk)
    radii = np.sqrt(np.sum(pts**2, axis=1))
    if np.any(radii >= 1.0):
        pts = pts * (0.95 / np.max(radii))
    
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    # Adaptive beta schedule - more aggressive
    beta_schedule = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    
    for beta in beta_schedule:
        if time.time() > _DEADLINE - 12:
            break
        
        # Define objective and gradient closures
        def obj(x):
            if time.time() > _DEADLINE - 5:
                raise ValueError("timeout")
            x = x.reshape(N, 2)
            val, _ = _soft_min_objective_and_grad(x, beta)
            return val
        
        def grad(x):
            if time.time() > _DEADLINE - 5:
                raise ValueError("timeout")
            x = x.reshape(N, 2)
            _, g = _soft_min_objective_and_grad(x, beta)
            return g.ravel()
        
        try:
            # Use tighter tolerances for higher beta
            if beta >= 1000:
                maxiter, ftol, gtol, maxfun = 200, 1e-14, 1e-10, 300
            else:
                maxiter, ftol, gtol, maxfun = 300, 1e-12, 1e-8, 500
            
            result = minimize(
                obj,
                pts.ravel(),
                jac=grad,
                method='L-BFGS-B',
                options={
                    'maxiter': maxiter,
                    'ftol': ftol,
                    'gtol': gtol,
                    'maxfun': maxfun
                }
            )
            
            new_pts = result.x.reshape(N, 2)
            
            # Check validity
            radii = np.sqrt(np.sum(new_pts**2, axis=1))
            if np.all(radii < 1.0) and np.all(np.isfinite(new_pts)):
                new_score = _compute_score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
                    pts = new_pts.copy()
                    
        except Exception:
            pass
    
    # Local polish using exact minimum
    if time.time() < _DEADLINE - 10:
        try:
            polished_pts, polished_score = _local_polish(best_pts, max_iters=500)
            if polished_score > best_score:
                best_score = polished_score
                best_pts = polished_pts.copy()
        except Exception:
            pass
    
    # Final high-precision soft-min polish
    if time.time() < _DEADLINE - 8:
        for beta in [100000, 200000]:
            if time.time() > _DEADLINE - 5:
                break
            
            def obj(x):
                if time.time() > _DEADLINE - 5:
                    raise ValueError("timeout")
                x = x.reshape(N, 2)
                val, _ = _soft_min_objective_and_grad(x, beta)
                return val
            
            def grad(x):
                if time.time() > _DEADLINE - 5:
                    raise ValueError("timeout")
                x = x.reshape(N, 2)
                _, g = _soft_min_objective_and_grad(x, beta)
                return g.ravel()
            
            try:
                result = minimize(
                    obj,
                    best_pts.ravel(),
                    jac=grad,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 150,
                        'ftol': 1e-15,
                        'gtol': 1e-12,
                        'maxfun': 200
                    }
                )
                
                new_pts = result.x.reshape(N, 2)
                radii = np.sqrt(np.sum(new_pts**2, axis=1))
                if np.all(radii < 1.0) and np.all(np.isfinite(new_pts)):
                    new_score = _compute_score(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = new_pts.copy()
                        
            except Exception:
                pass
    
    return best_pts
# EVOLVE-BLOCK-END
