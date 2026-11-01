"""
Strategy: Binary search on target radius r with violation-minimization feasibility checks
using L-BFGS-B. The key insight is that for a given target radius r, we can check
feasibility by minimizing a penalty that is zero iff all constraints are satisfied.
This directly targets the optimization of the minimum distance, allowing the optimizer
to focus on resolving specific overlaps rather than balancing all distances simultaneously.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _score(cents):
    """Compute the packing radius (the score)."""
    n = len(cents)
    if n == 0:
        return 0.0
    # Wall distances
    wall_min = np.min(np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1])))
    # Pairwise distances
    if n > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists) / 2.0
    else:
        pair_min = 0.5
    return min(wall_min, pair_min)

def _violation_loss(cents, r, log_weight=10.0):
    """
    Compute a smooth violation penalty for the given target radius r.
    The loss is zero iff all circles of radius r fit in the square without overlap.
    We use a log-sum-exp-like smooth approximation of the max violation.
    """
    n = len(cents)
    eps = 1e-10
    
    # Pairwise: want |c_i - c_j| >= 2r for all i<j
    # Violation: max(0, 2r - |c_i - c_j|)
    diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
    d_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(d_sq, np.inf)
    dists = np.sqrt(np.maximum(d_sq, 0.0))
    
    # Pairwise violations: 2r - d_ij (positive means overlap)
    pair_viol = 2.0 * r - dists
    # Only upper triangle
    tri_idx = np.triu_indices(n, k=1)
    pair_viol = pair_viol[tri_idx]
    # Soft max via log-sum-exp (with clamping)
    pair_viol_clamped = np.maximum(pair_viol, -1.0 / log_weight)
    pair_loss = np.log(np.sum(np.exp(log_weight * pair_viol_clamped))) / log_weight
    
    # Wall violations: want |c_ix| + r <= 0.5, i.e., 0.5 - |c_ix| >= r
    # Violation: max(0, r - (0.5 - |x|)) = max(0, r - 0.5 + |x|)
    wx = 0.5 - np.abs(cents[:, 0])
    wy = 0.5 - np.abs(cents[:, 1])
    wall_viol_x = r - wx
    wall_viol_y = r - wy
    
    wall_viol_x_clamped = np.maximum(wall_viol_x, -1.0 / log_weight)
    wall_viol_y_clamped = np.maximum(wall_viol_y, -1.0 / log_weight)
    wall_loss_x = np.log(np.sum(np.exp(log_weight * wall_viol_x_clamped))) / log_weight
    wall_loss_y = np.log(np.sum(np.exp(log_weight * wall_viol_y_clamped))) / log_weight
    
    return pair_loss + wall_loss_x + wall_loss_y

def _violation_loss_grad(cents, r, log_weight=10.0):
    """
    Analytical gradient of the violation loss.
    """
    n = len(cents)
    eps = 1e-10
    
    diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
    d_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(d_sq, np.inf)
    dists = np.sqrt(np.maximum(d_sq, eps))
    
    # Pairwise part
    pair_viol = 2.0 * r - dists
    tri_idx = np.triu_indices(n, k=1)
    pair_viol_flat = pair_viol[tri_idx]
    pair_viol_clamped = np.maximum(pair_viol_flat, -1.0 / log_weight)
    
    # Soft max gradient: exp(log_weight * v_clamped) / sum(exp(...)) * log_weight * (v == v_clamped)
    exp_vals = np.exp(log_weight * pair_viol_clamped)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / (sum_exp + eps)
    
    # Only where not clamped
    active = pair_viol_flat >= (-1.0 / log_weight)
    weights = weights * active
    
    # Build gradient for pairwise
    grad_pair = np.zeros_like(cents)
    for k in range(len(tri_idx[0])):
        i = tri_idx[0][k]
        j = tri_idx[1][k]
        if not active[k]:
            continue
        w = weights[k]
        # d/d(c_i) [2r - |c_i - c_j|] = -(c_i - c_j)/|c_i - c_j|
        # d/d(c_j) [2r - |c_i - c_j|] = +(c_i - c_j)/|c_i - c_j|
        d = diff[i, j]
        dist = dists[i, j] + eps
        grad_dir = d / dist
        grad_pair[i] += -w * grad_dir
        grad_pair[j] += w * grad_dir
    
    # Wall part
    wx = 0.5 - np.abs(cents[:, 0])
    wy = 0.5 - np.abs(cents[:, 1])
    wall_viol_x = r - wx
    wall_viol_y = r - wy
    
    wall_viol_x_clamped = np.maximum(wall_viol_x, -1.0 / log_weight)
    wall_viol_y_clamped = np.maximum(wall_viol_y, -1.0 / log_weight)
    
    exp_x = np.exp(log_weight * wall_viol_x_clamped)
    sum_x = np.sum(exp_x)
    weights_x = exp_x / (sum_x + eps)
    active_x = wall_viol_x >= (-1.0 / log_weight)
    weights_x = weights_x * active_x
    
    exp_y = np.exp(log_weight * wall_viol_y_clamped)
    sum_y = np.sum(exp_y)
    weights_y = exp_y / (sum_y + eps)
    active_y = wall_viol_y >= (-1.0 / log_weight)
    weights_y = weights_y * active_y
    
    # d/d(c_ix) [r - (0.5 - |c_ix|)] = sign(c_ix)
    sign_x = np.sign(cents[:, 0])
    sign_y = np.sign(cents[:, 1])
    # Handle zero sign
    sign_x[cents[:, 0] == 0] = 0
    sign_y[cents[:, 1] == 0] = 0
    
    grad_wall = np.zeros_like(cents)
    grad_wall[:, 0] += weights_x * sign_x
    grad_wall[:, 1] += weights_y * sign_y
    
    return grad_pair + grad_wall

def _feasible(cents, r):
    """Check if circles of radius r fit in the square."""
    return _violation_loss(cents, r, log_weight=50.0) < 1e-8

def _try_radius(r, init_cents, deadline):
    """
    Try to find a feasible configuration for target radius r.
    Returns (cents, feasible_flag).
    """
    def objective(x):
        return _violation_loss(x.reshape(-1, 2), r)
    
    def gradient(x):
        return _violation_loss_grad(x.reshape(-1, 2), r).flatten()
    
    x0 = init_cents.flatten().copy()
    
    bounds = [(-0.5 + r - 1e-9, 0.5 - r + 1e-9)] * (2 * N)
    
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'maxiter': 2000,
            'ftol': 1e-12,
            'gtol': 1e-10,
            'maxls': 50
        }
    )
    
    cents = result.x.reshape(-1, 2)
    loss = _violation_loss(cents, r)
    
    return cents, loss < 1e-8

def _construct_hexagonal():
    """Create a hexagonal packing arrangement for 50 circles."""
    best_cents = None
    best_r = 0

    for n_rows in [5, 6, 7, 8]:
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]

        max_rc = max(row_counts)
        r_horiz = 1.0 / (2.0 * max_rc)
        r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
        r = min(r_horiz, r_vert)

        if r <= best_r:
            continue

        centers = []
        total_y = (n_rows - 1) * 2 * r * np.sqrt(3.0) / 2.0
        y_start = -0.5 + r + total_y / 2.0

        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * 2 * r * np.sqrt(3.0) / 2.0
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            offset = (2 * r * np.sqrt(3.0) / 2.0) * 0.5 * (row_idx % 2)
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                centers.append([x, y])

        centers = np.array(centers[:N])
        r_actual = _score(centers)

        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()

    if best_cents is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
        best_r = _score(best_cents)

    return best_cents, best_r

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill
    
    # Step 1: Get initial hexagonal arrangement
    cents_hex, r0 = _construct_hexagonal()
    best_cents = cents_hex.copy()
    best_r = r0
    
    # Step 2: Binary search on radius
    # Start with a range around the known best
    r_low = 0.065
    r_high = 0.075
    
    # Verify r_low is feasible
    cents_low, feas_low = _try_radius(r_low, cents_hex, deadline)
    if feas_low:
        best_r = max(best_r, _score(cents_low))
        best_cents = cents_low.copy()
    else:
        r_low = 0.060
        cents_low, feas_low = _try_radius(r_low, cents_hex, deadline)
        if feas_low:
            best_r = max(best_r, _score(cents_low))
            best_cents = cents_low.copy()
    
    # Binary search for max feasible radius
    num_iter = 0
    while r_high - r_low > 1e-6:
        if time.time() > deadline - 10:
            break
        
        num_iter += 1
        r_mid = 0.5 * (r_low + r_high)
        
        # Try from multiple initializations
        found_feasible = False
        best_mid_cents = None
        
        # Try from hexagonal
        cents_mid, feas = _try_radius(r_mid, cents_hex, deadline)
        if feas:
            found_feasible = True
            best_mid_cents = cents_mid
        else:
            # Try from perturbed hexagonal
            rng = np.random.default_rng(num_iter * 1000)
            for pert_trial in range(3):
                if time.time() > deadline - 10:
                    break
                pert = cents_hex + rng.normal(0, 0.01 * (1.0 + pert_trial), size=cents_hex.shape)
                pert = np.clip(pert, -0.49, 0.49)
                cents_mid, feas = _try_radius(r_mid, pert, deadline)
                if feas:
                    found_feasible = True
                    best_mid_cents = cents_mid
                    break
        
        if found_feasible:
            r_low = r_mid
            if best_mid_cents is not None:
                r_actual = _score(best_mid_cents)
                if r_actual > best_r:
                    best_r = r_actual
                    best_cents = best_mid_cents.copy()
        else:
            r_high = r_mid
    
    # Final refinement: try to improve from best found
    if time.time() < deadline - 20:
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            rng = np.random.default_rng(999 + trial)
            pert = best_cents + rng.normal(0, 0.005, size=best_cents.shape)
            pert = np.clip(pert, -0.49, 0.49)
            
            # Try slightly larger radius
            r_target = best_r * 1.001
            cents_ref, feas = _try_radius(r_target, pert, deadline)
            if feas:
                r_actual = _score(cents_ref)
                if r_actual > best_r:
                    best_r = r_actual
                    best_cents = cents_ref.copy()
    
    # Final safety check
    best_cents = np.clip(best_cents, -0.5, 0.5)
    
    return best_cents
