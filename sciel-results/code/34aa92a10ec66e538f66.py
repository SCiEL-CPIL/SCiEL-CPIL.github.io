"""
Strategy: Binary-search on radius with smooth log-sum-exp epigraph solved by L-BFGS-B.

Instead of force-directed dynamics or population-based search, this approach:
1. Frames the problem as: for a target radius r, find centres such that all
   pairwise distances >= 2r and all wall distances >= r.
2. Uses a smooth log-sum-exp penalty on the constraint violations (scaled by 1/r),
   making the objective differentiable and amenable to L-BFGS-B.
3. Binary-searches on r to find the maximum feasible radius.
4. Seeds from multiple structured initializations (grid, hexagonal, random).
5. Polishes the best result with Nelder-Mead on the exact nonsmooth objective.

This exploits differentiability for fast convergence to local optima, which
gradient-free physics methods struggle with on the nonsmooth min-distance landscape.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

N = 50

def compute_radius(centres):
    """Compute the radius for a given set of centres."""
    if centres.shape != (N, 2):
        return 0.0
    if np.any(~np.isfinite(centres)):
        return 0.0
    if np.any(np.abs(centres) > 0.5 + 1e-9):
        return 0.0
    dists = pdist(centres, metric='euclidean')
    min_pair_dist = np.min(dists)
    dist_to_walls = np.array([
        0.5 - np.abs(centres[:, 0]),
        0.5 - np.abs(centres[:, 1])
    ])
    min_wall_dist = np.min(dist_to_walls)
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return max(0.0, r)

def _smooth_penalty(centres, r_target, alpha=50.0):
    """
    Smooth log-sum-exp penalty for constraint violations.
    
    Constraints:
    - Pairwise: |c_i - c_j| >= 2*r_target  -> violation = 2*r_target - |c_i - c_j|
    - Wall: |c_ik| <= 0.5 - r_target       -> violation = |c_ik| - (0.5 - r_target)
    
    We use log-sum-exp of (violation/alpha) to get a smooth approximation of max(violation).
    """
    if r_target <= 0:
        return 0.0, np.zeros_like(centres)
    
    # Pairwise violations
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]  # (N, N, 2)
    dists_sq = np.sum(diff**2, axis=2)  # (N, N)
    np.fill_diagonal(dists_sq, 1e-10)
    dists = np.sqrt(dists_sq)
    
    # violation_ij = 2*r_target - dist_ij (positive if constraint violated)
    pair_viol = 2.0 * r_target - dists
    np.fill_diagonal(pair_viol, -np.inf)  # no self-constraint
    
    # Wall violations
    d_left = centres[:, 0] + 0.5
    d_right = 0.5 - centres[:, 0]
    d_bottom = centres[:, 1] + 0.5
    d_top = 0.5 - centres[:, 1]
    
    # violation = |c| - (0.5 - r_target) for each coordinate
    wall_viol_x_left = centres[:, 0] + r_target - 0.5   # positive if too close to left
    wall_viol_x_right = 0.5 - centres[:, 0] - r_target  # positive if too close to right
    wall_viol_y_bottom = centres[:, 1] + r_target - 0.5
    wall_viol_y_top = 0.5 - centres[:, 1] - r_target
    
    # Combine all violations into one array for log-sum-exp
    # Scale by alpha for smoothness
    all_viol = np.concatenate([
        pair_viol.flatten(),
        wall_viol_x_left,
        wall_viol_x_right,
        wall_viol_y_bottom,
        wall_viol_y_top
    ])
    
    # Log-sum-exp: alpha * log(sum(exp(viol/alpha)))
    # This approximates max(viol) when alpha is small, and is smooth
    max_viol = np.max(all_viol)
    if max_viol < -100:
        return 0.0, np.zeros_like(centres)
    
    # For numerical stability, subtract max
    scaled = all_viol / alpha
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    penalty = alpha * log_sum_exp
    
    # Gradient computation
    # d(penalty)/d(centres) = sum_j [exp(viol_j/alpha) / sum_k exp(viol_k/alpha)] * d(viol_j)/d(centres)
    
    exp_scaled = np.exp(scaled - max_scaled)
    weights = exp_scaled / np.sum(exp_scaled)  # (N*(N-1) + 4N,)
    
    grad = np.zeros_like(centres)
    
    # Pairwise gradient
    pair_weights = weights[:N*(N-1)].reshape(N, N)
    # violation_ij = 2*r - dist_ij
    # d(viol_ij)/d(c_i) = -diff_ij / dist_ij
    # d(viol_ij)/d(c_j) = +diff_ij / dist_ij
    # Since we sum over all i,j, and the constraint is symmetric, we need to be careful.
    # Actually, pair_viol is (N,N) with diagonal set to -inf.
    # For each pair (i,j), i!=j:
    #   d(viol_ij)/d(c_i) = -(c_i - c_j)/dist_ij
    #   d(viol_ij)/d(c_j) = +(c_i - c_j)/dist_ij
    
    # Vectorized: for each i, sum over j != i of w_ij * (-(c_i - c_j)/dist_ij)
    # = -sum_j w_ij * (c_i - c_j)/dist_ij
    
    inv_dists = np.zeros_like(dists)
    mask = ~np.eye(N, dtype=bool)
    inv_dists[mask] = 1.0 / dists[mask]
    
    # grad_i = -sum_j w_ij * diff_ij * inv_dists_ij
    # diff_ij = c_i - c_j
    # So grad_i = -sum_j w_ij * (c_i - c_j) / dist_ij
    
    # Vectorized:
    # For each i: grad_i = -sum_j pair_weights[i,j] * diff[i,j,:] * inv_dists[i,j]
    weighted_diff = pair_weights[:, :, np.newaxis] * diff * inv_dists[:, :, np.newaxis]
    grad -= np.sum(weighted_diff, axis=1)
    
    # Wall gradients
    # wall_viol_x_left = c_x + r - 0.5, d/dc_x = 1
    w_xl = weights[N*(N-1):N*(N-1)+N]
    w_xr = weights[N*(N-1)+N:N*(N-1)+2*N]
    w_yb = weights[N*(N-1)+2*N:N*(N-1)+3*N]
    w_yt = weights[N*(N-1)+3*N:N*(N-1)+4*N]
    
    grad[:, 0] += w_xl - w_xr
    grad[:, 1] += w_yb - w_yt
    
    return penalty, grad

def _solve_for_radius(r_target, init_centres, maxiter=2000, alpha=50.0):
    """
    Try to find centres feasible for radius r_target using L-BFGS-B.
    Returns (success, centres).
    """
    def objective_and_grad(x):
        c = x.reshape(N, 2)
        # Clip to ensure we stay in bounds for gradient computation
        c = np.clip(c, -0.5 + 1e-12, 0.5 - 1e-12)
        penalty, grad = _smooth_penalty(c, r_target, alpha=alpha)
        # Add a small regularization to keep centres in bounds
        # Penalty for being outside the feasible region
        bound_penalty = 0.0
        bound_grad = np.zeros_like(c)
        margin = 0.5 - r_target
        # If any coordinate exceeds [-margin, margin], add quadratic penalty
        c_x = c[:, 0]
        c_y = c[:, 1]
        # Left/right
        left_viol = c_x - (-margin)  # c_x + margin
        right_viol = c_x - margin
        bottom_viol = c_y - (-margin)
        top_viol = c_y - margin
        
        # Quadratic penalty for boundary violations
        qp = 0.0
        qg = np.zeros_like(c)
        if np.any(left_viol > 0):
            qp += np.sum(left_viol[left_viol > 0]**2)
            qg[:, 0] += 2.0 * np.maximum(left_viol, 0)
        if np.any(right_viol > 0):
            qp += np.sum(right_viol[right_viol > 0]**2)
            qg[:, 0] -= 2.0 * np.maximum(right_viol, 0)
        if np.any(bottom_viol > 0):
            qp += np.sum(bottom_viol[bottom_viol > 0]**2)
            qg[:, 1] += 2.0 * np.maximum(bottom_viol, 0)
        if np.any(top_viol > 0):
            qp += np.sum(top_viol[top_viol > 0]**2)
            qg[:, 1] -= 2.0 * np.maximum(top_viol, 0)
        
        total_obj = penalty + 100.0 * qp
        total_grad = grad + 200.0 * qg
        
        return total_obj, total_grad.flatten()
    
    x0 = np.clip(init_centres, -0.5, 0.5).flatten()
    
    res = minimize(
        objective_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
    )
    
    new_c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
    
    # Check if feasible: compute actual radius
    r_actual = compute_radius(new_c)
    success = r_actual >= r_target * (1.0 - 1e-6)
    
    return success, new_c

def _grid_init():
    """Generate a grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _hex_init():
    """Generate a hexagonal (close-packed) initialization."""
    # Hexagonal packing: rows offset by half spacing
    # For 50 circles, use roughly 7x7 or 8x7 grid with hex offsets
    points = []
    # Try to fit as many as possible in a hex pattern
    row_spacing = 1.0 / 8.0  # approximate
    col_spacing = row_spacing * np.sqrt(3) / 2
    y = -0.45
    row = 0
    while len(points) < N and y < 0.45:
        x_start = -0.45 + (col_spacing / 2 if row % 2 == 1 else 0)
        x = x_start
        while x < 0.45 and len(points) < N:
            points.append((x, y))
            x += col_spacing
        y += row_spacing * 0.866  # sqrt(3)/2
        row += 1
    
    if len(points) < N:
        # Fill remaining with random
        for _ in range(N - len(points)):
            points.append((np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4)))
    
    return np.array(points[:N])

def _random_init():
    """Generate a random initialization within the square."""
    return np.random.uniform(-0.4, 0.4, size=(N, 2))

def _polish(centres, maxiter=2000):
    """
    High-precision polish using Nelder-Mead on the exact nonsmooth objective.
    """
    def objective(x):
        c = x.reshape(N, 2)
        r = compute_radius(c)
        return -r
    
    res = minimize(objective, centres.flatten(), method='Nelder-Mead',
                   options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
    
    new_c = res.x.reshape(N, 2)
    np.clip(new_c, -0.5, 0.5, out=new_c)
    return new_c

def _binary_search_radius(init_centres, r_low, r_high, max_iter=15, time_limit=None):
    """
    Binary search on radius using L-BFGS-B feasibility checks.
    """
    best_centres = init_centres.copy()
    best_r = compute_radius(best_centres)
    
    for i in range(max_iter):
        if time_limit is not None and time.time() > time_limit:
            break
        
        r_mid = (r_low + r_high) / 2.0
        if r_mid <= best_r:
            r_low = r_mid
            continue
        
        success, new_c = _solve_for_radius(r_mid, init_centres, maxiter=1000, alpha=50.0)
        
        if success:
            r_low = r_mid
            r_actual = compute_radius(new_c)
            if r_actual > best_r:
                best_r = r_actual
                best_centres = new_c
                init_centres = new_c.copy()
        else:
            r_high = r_mid
    
    return best_centres, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Set seeds for reproducibility
    np.random.seed(42)
    
    # Multiple initializations
    inits = [
        ('grid', _grid_init()),
        ('hex', _hex_init()),
        ('rand1', _random_init()),
        ('rand2', _random_init()),
    ]
    
    best_centres = None
    best_r = 0.0
    
    # For each initialization, run binary search on radius
    for name, init_c in inits:
        if time.time() > deadline - 20:
            break
        
        # Get initial radius
        r_init = compute_radius(init_c)
        
        # Set search bounds
        # Theoretical upper bound: for N circles in unit square, r <= 1/(2*sqrt(N)) approximately
        # But more carefully: area argument gives r <= sqrt(1/N) / (2*sqrt(pi)) ~ 0.0798 for N=50
        # Use a slightly higher bound for search
        r_low = r_init
        r_high = 0.075  # slightly above known best
        
        # If initial is already good, narrow the search
        if r_init > 0.07:
            r_low = r_init
            r_high = min(r_init * 1.05, 0.075)
        
        # Run binary search with time check
        time_limit = deadline - 15
        c_result, r_result = _binary_search_radius(
            init_c, r_low, r_high, max_iter=12, time_limit=time_limit
        )
        
        if r_result > best_r:
            best_r = r_result
            best_centres = c_result
    
    # Polish the best result
    if best_centres is not None and time.time() < deadline - 10:
        c_polished = _polish(best_centres, maxiter=1000)
        r_polished = compute_radius(c_polished)
        
        if r_polished > best_r:
            best_r = r_polished
            best_centres = c_polished
    
    # Final tiny local search
    if best_centres is not None and time.time() < deadline - 3:
        for _ in range(10):
            if time.time() > deadline - 3:
                break
            c_try = best_centres + np.random.normal(0, 0.0005, size=(N, 2))
            np.clip(c_try, -0.5, 0.5, out=c_try)
            r_try = compute_radius(c_try)
            if r_try > best_r:
                best_r = r_try
                best_centres = c_try
    
    # Final validation
    if best_centres is None or np.any(~np.isfinite(best_centres)):
        best_centres = _grid_init()
    
    best_centres = np.clip(best_centres, -0.5, 0.5)
    
    return best_centres
