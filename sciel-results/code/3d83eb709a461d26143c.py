"""
Strategy: Nesterov-accelerated projected gradient ascent on the exact min-distance
objective with vectorized k-NN active-set subgradients, followed by L-BFGS-B polish
on a log-sum-exp surrogate for tight local convergence.

Key structural differences from the MD/annealing approach:
1. Deterministic gradient-based optimization (no stochastic noise in the main loop)
2. Exact min-distance subgradient via k-NN (no smoothing approximation in the main phase)
3. Nesterov momentum for faster convergence
4. Continuation: start with soft k-NN (larger k) for broad exploration, reduce k for precision
5. Final L-BFGS-B polish on LSE surrogate for sub-micron convergence
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _compute_radius(centres):
    """Compute the maximum common radius for given centres in [-0.5, 0.5]^3."""
    n = len(centres)
    face_dist = 0.5 - np.max(np.abs(centres), axis=1)
    min_face = np.min(face_dist)
    
    if n < 2:
        return min_face
    
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_face, min_pair / 2.0)


def _active_set_subgradient(centres, k=5, eps=1e-10):
    """
    Compute a subgradient of the min-distance objective using k-NN active set.
    
    The objective is r = min(min_pair_dist/2, min_wall_dist).
    We use a weighted sum of gradients from the k closest constraints.
    
    Returns: (gradient, current_r)
    """
    n = len(centres)
    grad = np.zeros_like(centres)
    
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = 0.5 - np.max(np.abs(centres), axis=1)
    
    # Find the k closest pairs
    # Flatten pair distances (upper triangle)
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    pair_indices = list(zip(iu[0], iu[1]))
    
    # Get k smallest pair distances
    k_pair = min(k, len(pair_dists))
    idx_sorted = np.argsort(pair_dists)
    active_pair_idx = idx_sorted[:k_pair]
    
    # Find the k closest walls
    k_wall = min(k, n)
    wall_idx_sorted = np.argsort(wall_dists)
    active_wall_idx = wall_idx_sorted[:k_wall]
    
    # Current radius
    min_pair_val = np.min(pair_dists) / 2.0
    min_wall_val = np.min(wall_dists)
    current_r = min(min_pair_val, min_wall_val)
    
    # Weighted subgradient: weights based on how close each constraint is to the minimum
    # Use inverse distance weighting (closer to min gets more weight)
    
    # Pair contributions
    for ai in active_pair_idx:
        i, j = pair_indices[ai]
        d = dists[i, j]
        if d < eps:
            continue
        # Weight: how close is this pair to the minimum?
        weight = 1.0 / (d - min_pair_val + 1e-6)
        # Gradient of d_ij wrt centres[i] = (c_i - c_j) / d
        u = (centres[i] - centres[j]) / d
        # We want to maximize min distance, so gradient of d_ij/2 wrt c_i is u/2
        grad[i] += weight * u / 2.0
        grad[j] -= weight * u / 2.0
    
    # Wall contributions
    for wi in active_wall_idx:
        i = wi
        d_wall = wall_dists[i]
        weight = 1.0 / (d_wall - min_wall_val + 1e-6)
        # Gradient of wall_dist wrt c_i: -sign(c_ik) for the axis with max |c_ik|
        c = centres[i]
        abs_c = np.abs(c)
        axis = np.argmax(abs_c)
        g = np.zeros(3)
        g[axis] = -np.sign(c[axis])
        grad[i] += weight * g
    
    # Normalize gradient to unit norm for stable stepping
    gnorm = np.linalg.norm(grad)
    if gnorm > eps:
        grad /= gnorm
    
    return grad, current_r


def _nesterov_ascent(centres, max_iter=3000, lr=0.002, momentum=0.9, 
                     k=5, time_limit=None, deadline=None):
    """
    Nesterov-accelerated projected gradient ascent on the exact min-distance objective.
    Uses k-NN active set for subgradient computation.
    """
    c = centres.copy()
    v = np.zeros_like(c)
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Nesterov lookahead
        c_lookahead = c + momentum * v
        
        # Compute subgradient at lookahead point
        grad, r = _active_set_subgradient(c_lookahead, k=k)
        
        # Update velocity (Nesterov)
        v = momentum * v + lr * grad
        
        # Projected step
        c = c + v
        
        # Project back to cube [-0.5, 0.5]^3
        c = np.clip(c, -0.4999, 0.4999)
        
        # Track best
        r_current = _compute_radius(c)
        if r_current > best_r:
            best_r = r_current
            best_c = c.copy()
    
    return best_c, best_r


def _lse_surrogate(centres, beta=500.0):
    """
    Log-sum-exp surrogate for the min-distance objective.
    Maximizing this smooth function approximates maximizing the exact min distance.
    """
    n = len(centres)
    
    # Pairwise distances / 2
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    # Wall distances
    wall_vals = 0.5 - np.max(np.abs(centres), axis=1)
    
    # Combine all constraints
    # LSE of all pair distances/2 and wall distances
    # We want to maximize the minimum, so we minimize the LSE of (-values)
    # Or equivalently, maximize -LSE(-values) = log-sum-exp trick
    
    # Collect all constraint values
    iu = np.triu_indices(n, k=1)
    pair_flat = pair_vals[iu]
    all_vals = np.concatenate([pair_flat, wall_vals])
    
    # LSE: log(sum(exp(beta * v))) / beta  (this approximates max)
    # We want to maximize this
    # For optimization, we minimize -LSE
    max_val = np.max(all_vals)
    if max_val > 10.0 / beta:
        # Overflow protection
        lse = max_val + np.log(np.sum(np.exp(beta * (all_vals - max_val)))) / beta
    else:
        lse = np.log(np.sum(np.exp(beta * all_vals))) / beta
    
    return -lse  # Negative because we minimize


def _lse_gradient(centres, beta=500.0):
    """Gradient of the LSE surrogate (for minimization)."""
    n = len(centres)
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    wall_vals = 0.5 - np.max(np.abs(centres), axis=1)
    
    iu = np.triu_indices(n, k=1)
    pair_flat = pair_vals[iu]
    all_vals = np.concatenate([pair_flat, wall_vals])
    
    max_val = np.max(all_vals)
    exp_vals = np.exp(beta * (all_vals - max_val))
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # softmax weights
    
    # Gradient of -LSE wrt centres
    # d(-LSE)/dc = -(1/beta) * dLSE/dc
    # dLSE/dc_i = sum_j w_j * d(v_j)/dc_i
    # where v_j are the constraint values
    
    grad = np.zeros_like(centres)
    
    # Pair contributions
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        d = dists[i, j]
        if d < 1e-10:
            continue
        w = weights[idx] * beta  # dLSE/d(pair_val) * d(pair_val)/d(dist) = w * beta * (1/2)
        # Actually: LSE = log(sum(exp(beta*v)))/beta
        # dLSE/dv_j = w_j (where w_j = exp(beta*v_j)/sum(exp(beta*v)))
        # d(pair_val)/d(c_i) = (c_i - c_j) / (2*d) * ... wait
        # pair_val = dist/2, d(pair_val)/d(c_i) = (c_i - c_j)/(2*d)
        # So dLSE/d(c_i) += w_j * (c_i - c_j)/(2*d)
        # d(-LSE)/d(c_i) -= w_j * (c_i - c_j)/(2*d)
        
        u = (centres[i] - centres[j]) / d
        grad[i] -= w * u / 2.0
        grad[j] += w * u / 2.0
    
    # Wall contributions
    for idx in range(n):
        w = weights[len(pair_flat) + idx]
        i = idx
        c = centres[i]
        abs_c = np.abs(c)
        axis = np.argmax(abs_c)
        # d(wall_val)/d(c_ik) = -sign(c_ik) for the max axis
        g = np.zeros(3)
        g[axis] = -np.sign(c[axis])
        grad[i] -= w * g
    
    return grad


def _lbgfgs_polish(centres, beta=500.0, maxiter=200, time_deadline=None):
    """Final L-BFGS-B polish on the LSE surrogate."""
    def obj_and_grad(x):
        x = x.reshape(N, 3)
        val = _lse_surrogate(x, beta=beta)
        g = _lse_gradient(x, beta=beta).reshape(-1)
        return val, g
    
    x0 = centres.reshape(-1)
    
    result = minimize(
        obj_and_grad, x0, method='L-BFGS-B',
        jac=True,  # We provide gradient
        options={'maxiter': maxiter, 'gtol': 1e-10, 'ftol': 1e-12}
    )
    
    # Check time
    if time_deadline is not None and time.time() > time_deadline:
        return centres
    
    x_opt = result.x.reshape(N, 3)
    x_opt = np.clip(x_opt, -0.4999, 0.4999)
    
    return x_opt


def _generate_grid_start(m=4):
    """Generate a grid-based initial configuration."""
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    np.random.shuffle(grid)
    return grid[:N].copy()


def _generate_fcc_start():
    """Generate an FCC-like initial configuration."""
    # FCC lattice points in a cube
    a = 1.0 / 4.0  # lattice constant
    points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                x = (i + 0.5) * a - 0.5
                y = (j + 0.5) * a - 0.5
                z = (k + 0.5) * a - 0.5
                points.append([x, y, z])
                # FCC: add face-centered points
                if i < 3:
                    points.append([x + a/2, y + a/2, z])
                    points.append([x + a/2, y, z + a/2])
                    points.append([x, y + a/2, z + a/2])
    
    points = np.array(points[:N])
    np.random.shuffle(points)
    return points[:N].copy()


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    np.random.seed(42)
    
    # Generate multiple starting points
    starts = []
    starts.append(_generate_grid_start(4))
    starts.append(_generate_grid_start(5))
    starts.append(_generate_fcc_start())
    
    # Add a few random starts
    for _ in range(2):
        random_start = np.random.uniform(-0.45, 0.45, (N, 3))
        starts.append(random_start)
    
    best_centres = None
    best_r = 0.0
    
    # Phase 1: Nesterov ascent with continuation on k
    # Start with larger k (softer, more exploration) and reduce to smaller k (sharper)
    k_schedule = [10, 7, 5, 3, 2, 1]
    
    for start_idx, init in enumerate(starts):
        if time.time() > deadline - 30.0:
            break
        
        c = init.copy()
        current_r = _compute_radius(c)
        
        # Continuation over k values
        for k in k_schedule:
            if time.time() > deadline - 15.0:
                break
            
            # Adaptive learning rate: smaller for smaller k
            lr = 0.001 * (5.0 / (k + 1))
            momentum = 0.9 - 0.05 * (len(k_schedule) - k_schedule.index(k))
            
            # Run Nesterov ascent
            steps = max(200, int(2000 * (1.0 / (k + 1))))
            c, r = _nesterov_ascent(
                c, max_iter=steps, lr=lr, momentum=momentum,
                k=k, deadline=deadline - 10.0
            )
            
            if r > current_r:
                current_r = r
        
        # L-BFGS-B polish
        if time.time() < deadline - 5.0:
            c_polished = _lbgfgs_polish(c, beta=500.0, maxiter=100, time_deadline=deadline - 3.0)
            r_polished = _compute_radius(c_polished)
            if r_polished > current_r:
                current_r = r_polished
                c = c_polished
        
        if current_r > best_r:
            best_r = current_r
            best_centres = c.copy()
    
    # Final high-precision polish on the best solution
    if best_centres is not None and time.time() < deadline - 3.0:
        final = _lbgfgs_polish(best_centres, beta=1000.0, maxiter=200, time_deadline=deadline - 1.0)
        final_r = _compute_radius(final)
        if final_r > best_r:
            best_r = final_r
            best_centres = final
    
    if best_centres is None:
        best_centres = starts[0]
    
    return best_centres
