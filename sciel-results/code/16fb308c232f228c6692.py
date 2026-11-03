"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(P):
    """Calculate the radius score for a given set of points."""
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return -np.inf
    
    # Distance to boundary
    d_bound = 1.0 - np.linalg.norm(P, axis=1)
    r_bound = np.min(d_bound)
    
    # Pairwise distances
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    r_pair = min_pair_dist / 2.0
    
    return min(r_pair, r_bound)

def _get_ring_initialization(r_target, ring_radii=None, ring_angles=None):
    """
    Generate initial points based on concentric rings matching the known optimal structure.
    Ring structure: 1 + 6 + 12 + 18 + 13 = 50
    """
    P = np.zeros((N, 2))
    
    # Ring 0: 1 point at center
    P[0] = [0.0, 0.0]
    count = 1
    
    ring_counts = [6, 12, 18, 13]
    
    if ring_radii is None:
        # Empirical ring radii ratios for near-optimal packing
        ring_radii = [2.0 * r_target, 3.3 * r_target, 4.6 * r_target, 5.5 * r_target]
    
    if ring_angles is None:
        # Stagger angles for better packing
        ring_angles = [0.0, np.pi/12.0, 0.0, np.pi/18.0]
    
    for k, n_pts in enumerate(ring_counts):
        R = ring_radii[k]
        base_angle = ring_angles[k]
        angles = base_angle + np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        for i in range(n_pts):
            if count >= N:
                break
            P[count, 0] = R * np.cos(angles[i])
            P[count, 1] = R * np.sin(angles[i])
            count += 1
            
    return P

def _soft_min(values, beta=50.0):
    """
    Smooth approximation of min using log-sum-exp (negative for min).
    soft_min(v) ≈ -1/beta * log(sum(exp(-beta * v_i)))
    """
    min_val = np.min(values)
    shifted = values - min_val
    return min_val - np.log(np.sum(np.exp(-beta * shifted))) / beta

def _soft_max(values, beta=50.0):
    """
    Smooth approximation of max using log-sum-exp.
    soft_max(v) ≈ 1/beta * log(sum(exp(beta * v_i)))
    """
    max_val = np.max(values)
    shifted = values - max_val
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta

def _smooth_energy(P, r, beta=50.0):
    """
    Smooth energy: penalize violations of constraints using soft functions.
    We want all pairwise distances >= 2r and all points within distance (1-r) from origin.
    
    Violation for pair (i,j): max(0, 2r - dist_ij)
    Violation for point i: max(0, |p_i| + r - 1)
    
    We use a smooth version that is differentiable everywhere.
    """
    # Pairwise distances
    diff = P[:, np.newaxis, :] - P[np.newaxis, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(dist_sq + 1e-15)  # small epsilon for numerical stability
    np.fill_diagonal(dist, np.inf)
    
    # Boundary distances
    dist_origin = np.linalg.norm(P, axis=1)
    
    # Violations
    violations_cc = 2.0 * r - dist  # positive means overlap
    violations_cb = dist_origin + r - 1.0  # positive means outside
    
    # Only penalize actual violations (positive values)
    # Use soft-plus: smooth_plus(x) = log(1 + exp(beta*x))/beta ≈ max(0, x) for large beta
    def soft_plus(x, beta=50.0):
        return np.log1p(np.exp(beta * x)) / beta
    
    E_cc = np.sum(soft_plus(violations_cc, beta))
    E_cb = np.sum(soft_plus(violations_cb, beta))
    
    return E_cc + E_cb

def _smooth_energy_grad(P, r, beta=50.0):
    """
    Gradient of the smooth energy function.
    """
    grad = np.zeros_like(P)
    
    # Pairwise
    diff = P[:, np.newaxis, :] - P[np.newaxis, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(dist_sq + 1e-15)
    np.fill_diagonal(dist, np.inf)
    
    # Boundary
    dist_origin = np.linalg.norm(P, axis=1)
    
    # soft_plus derivative: sigmoid(beta * x)
    def soft_plus_grad(x, beta=50.0):
        # sigmoid(beta*x)
        beta_x = beta * x
        # Clamp for numerical stability
        beta_x = np.clip(beta_x, -500, 500)
        return 1.0 / (1.0 + np.exp(-beta_x))
    
    # Pairwise gradient
    violations_cc = 2.0 * r - dist
    g_cc = soft_plus_grad(violations_cc, beta)
    
    # dE_cc/dP_i = -sum_j g_cc_ij * (P_i - P_j) / dist_ij
    inv_dist = np.zeros_like(dist)
    np.divide(1.0, dist, out=inv_dist, where=dist > 0.001)
    
    W = g_cc * inv_dist
    np.fill_diagonal(W, 0.0)
    
    # grad_i = -sum_j W_ij * (P_i - P_j) = -P_i * sum_j W_ij + sum_j W_ij * P_j
    sum_W = np.sum(W, axis=1)
    term1 = P * sum_W[:, np.newaxis]
    term2 = P.T @ W.T  # (N,2) @ (N,N).T -> need to be careful
    # Actually: sum_j W_ij * P_j = (W @ P.T).T
    term2 = (W @ P.T).T
    
    grad -= term1 - term2  # grad = -(term1 - term2) = term2 - term1
    
    # Boundary gradient
    violations_cb = dist_origin + r - 1.0
    g_cb = soft_plus_grad(violations_cb, beta)
    
    inv_dist_origin = np.zeros_like(dist_origin)
    np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 0.001)
    
    grad += (g_cb * inv_dist_origin)[:, np.newaxis] * P
    
    return grad

def _optimize_for_r(P_init, r_target, max_iter=500, tol=1e-12, beta=50.0, deadline=None):
    """
    Optimize positions P to minimize smooth energy for a fixed r_target.
    Returns (P_opt, min_energy, success).
    """
    def objective(P_flat):
        P = P_flat.reshape(N, 2)
        return _smooth_energy(P, r_target, beta)
    
    def grad(P_flat):
        P = P_flat.reshape(N, 2)
        return _smooth_energy_grad(P, r_target, beta).flatten()
        
    res = minimize(objective, P_init.flatten(), jac=grad, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10, 'maxls': 50})
    P_opt = res.x.reshape(N, 2)
    min_energy = objective(P_opt)
    
    # Check if solution is valid
    score = _compute_score(P_opt)
    success = score >= r_target - 1e-9
    
    return P_opt, min_energy, success

def _try_multiple_starting_points(r_target, max_starts=5, deadline=None):
    """
    Try multiple starting configurations and return the best one.
    """
    best_P = None
    best_score = -np.inf
    
    # Starting point 1: ring structure
    P_ring = _get_ring_initialization(r_target)
    P_opt, E, success = _optimize_for_r(P_ring, r_target, max_iter=400, tol=1e-12, deadline=deadline)
    score = _compute_score(P_opt)
    if score > best_score:
        best_score = score
        best_P = P_opt.copy()
    
    # Starting point 2: perturbed ring
    if deadline is None or time.time() < deadline - 20.0:
        P_perturbed = P_ring + np.random.normal(0, 0.02 * r_target, size=(N, 2))
        P_opt2, E2, success2 = _optimize_for_r(P_perturbed, r_target, max_iter=300, tol=1e-12, deadline=deadline)
        score2 = _compute_score(P_opt2)
        if score2 > best_score:
            best_score = score2
            best_P = P_opt2.copy()
    
    # Starting point 3: different ring angles
    if deadline is None or time.time() < deadline - 15.0:
        ring_radii = [2.0 * r_target, 3.2 * r_target, 4.7 * r_target, 5.6 * r_target]
        ring_angles = [np.pi/6.0, 0.0, np.pi/10.0, 0.0]
        P_alt = _get_ring_initialization(r_target, ring_radii, ring_angles)
        P_opt3, E3, success3 = _optimize_for_r(P_alt, r_target, max_iter=300, tol=1e-12, deadline=deadline)
        score3 = _compute_score(P_opt3)
        if score3 > best_score:
            best_score = score3
            best_P = P_opt3.copy()
    
    return best_P, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer before 150s limit
    
    # Known optimal is ~0.1258254895304
    # Start with a conservative r and binary search up
    
    # First, get a good initial solution
    r_init = 0.12
    P_init = _get_ring_initialization(r_init)
    
    # Optimize at r_init to get a good starting point
    P_opt, E, success = _optimize_for_r(P_init, r_init, max_iter=300, tol=1e-12, deadline=deadline)
    P_best = P_opt.copy()
    score_best = _compute_score(P_best)
    
    # Binary search for the maximum feasible r
    r_low = r_init
    r_high = 0.135
    
    # Verify r_low is feasible
    if not success:
        # Try to fix
        P_opt, E, success = _optimize_for_r(P_best, r_low * 0.95, max_iter=200, tol=1e-12, deadline=deadline)
        P_best = P_opt
        score_best = _compute_score(P_best)
        r_low = r_low * 0.95
    
    # Binary search iterations
    for iteration in range(12):
        if time.time() > deadline - 20.0:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Try multiple starting points for r_mid
        P_try, score_try = _try_multiple_starting_points(r_mid, max_starts=3, deadline=deadline)
        
        if P_try is not None and score_try >= r_mid - 1e-8:
            # Feasible, try higher r
            r_low = r_mid
            if score_try > score_best:
                score_best = score_try
                P_best = P_try.copy()
        else:
            # Infeasible, try lower r
            r_high = r_mid
    
    # Final polish: optimize at the best feasible r with tighter tolerance and higher beta
    if time.time() < deadline - 5.0:
        # Use higher beta for sharper penalty
        P_ring = _get_ring_initialization(r_low)
        
        # Try with different beta values
        for beta_val in [80.0, 100.0]:
            if time.time() > deadline - 5.0:
                break
            P_polish, E_polish, success_polish = _optimize_for_r(
                P_ring, r_low, max_iter=800, tol=1e-14, beta=beta_val, deadline=deadline
            )
            score_polish = _compute_score(P_polish)
            if score_polish > score_best:
                score_best = score_polish
                P_best = P_polish.copy()
        
        # Also try polishing from current best
        if time.time() < deadline - 5.0:
            P_polish2, E_polish2, success_polish2 = _optimize_for_r(
                P_best, r_low, max_iter=500, tol=1e-14, beta=100.0, deadline=deadline
            )
            score_polish2 = _compute_score(P_polish2)
            if score_polish2 > score_best:
                score_best = score_polish2
                P_best = P_polish2.copy()
    
    # Final verification
    final_score = _compute_score(P_best)
    if final_score < 0.1:
        # Fallback: return a valid but suboptimal solution
        P_fallback = _get_ring_initialization(0.12)
        return P_fallback
    
    return P_best
# EVOLVE-BLOCK-END
