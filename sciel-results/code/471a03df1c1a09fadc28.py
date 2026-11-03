"""
Pack 200 equal circles in a unit square using a two-phase approach:
1. Lennard-Jones potential energy minimization with BFGS for global relaxation
2. Soft-min gradient-based optimization for precision polishing
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
_TIME_LIMIT = 130  # Leave margin for 150s limit (20s safety)

def _get_radius(pts):
    """
    Calculate the maximum possible radius for given centers.
    Returns the min of (wall distances) and (pairwise distances / 2).
    """
    if len(pts) == 0:
        return 0.0
    
    # Wall distances: distance to each of the 4 walls
    wall_dists = np.array([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])
    
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0
    
    return min(min_wall, min_pair)

def _lj_potential_and_grad(x, sigma, epsilon, wall_k):
    """
    Compute Lennard-Jones potential energy and gradient.
    
    LJ: U(r) = 4*epsilon * ((sigma/r)^12 - (sigma/r)^6)
    Wall: U_wall = wall_k * sum of (distance_to_wall)^2 for points outside
    
    x: flattened array of shape (2*N,)
    """
    pts = x.reshape((N, 2))
    
    # Compute all pairwise distances
    D = cdist(pts, pts)
    np.fill_diagonal(D, 1.0)  # Avoid division by zero on diagonal
    
    # LJ potential: sum over all pairs i<j
    # U = 4*epsilon * ((sigma/r)^12 - (sigma/r)^6)
    # For numerical stability, only consider pairs with r < 2*sigma (cutoff)
    cutoff = 2.5 * sigma
    mask = D < cutoff
    # Exclude diagonal
    mask[np.arange(N), np.arange(N)] = False
    
    # Compute LJ terms
    r = D[mask]
    sr6 = (sigma / r) ** 6
    U_pairs = 4 * epsilon * (sr6**2 - sr6)
    energy = np.sum(U_pairs)
    
    # Gradient of LJ potential
    # dU/dr = 4*epsilon * (-12*sigma^12/r^13 + 6*sigma^6/r^7)
    #       = 24*epsilon * (-2*(sigma/r)^12 + (sigma/r)^6) / r
    # dU/dx_i = (dU/dr) * (x_i - x_j) / r
    # dU/dx_j = (dU/dr) * (x_j - x_i) / r
    
    grad = np.zeros((N, 2))
    
    if np.any(mask):
        # Get indices of valid pairs
        i_idx, j_idx = np.where(mask)
        dx = pts[i_idx, 0] - pts[j_idx, 0]
        dy = pts[i_idx, 1] - pts[j_idx, 1]
        r_vals = D[i_idx, j_idx]
        
        # Compute force factor: F(r) = 24*epsilon * (-2*(sigma/r)^12 + (sigma/r)^6) / r
        sr6_vals = (sigma / r_vals) ** 6
        force_factor = 24 * epsilon * (-2 * sr6_vals**2 + sr6_vals) / r_vals
        
        # Accumulate gradients
        # For point i: force is in direction (x_i - x_j) * force_factor
        # For point j: force is in direction (x_j - x_i) * force_factor = -(x_i - x_j) * force_factor
        
        np.add.at(grad[:, 0], i_idx, dx * force_factor)
        np.add.at(grad[:, 0], j_idx, -dx * force_factor)
        np.add.at(grad[:, 1], i_idx, dy * force_factor)
        np.add.at(grad[:, 1], j_idx, -dy * force_factor)
    
    # Wall potential: quadratic penalty for points outside [-0.5, 0.5]
    # U_wall = wall_k * sum over points of max(0, |coord| - 0.5)^2
    # Gradient: wall_k * 2 * sign(coord) * max(0, |coord| - 0.5)
    
    # x-coordinate walls
    left_dist = pts[:, 0] + 0.5   # distance to left wall (negative if outside)
    right_dist = 0.5 - pts[:, 0]  # distance to right wall (negative if outside)
    
    # For left wall: if pts[:,0] < -0.5, penalty = wall_k * (-0.5 - pts[:,0])^2
    # Gradient: -2 * wall_k * (-0.5 - pts[:,0]) = 2 * wall_k * (pts[:,0] + 0.5) when pts[:,0] < -0.5
    left_violation = np.maximum(0, -left_dist)
    right_violation = np.maximum(0, -right_dist)
    
    wall_energy_x = wall_k * (np.sum(left_violation**2) + np.sum(right_violation**2))
    
    # y-coordinate walls
    bottom_dist = pts[:, 1] + 0.5
    top_dist = 0.5 - pts[:, 1]
    bottom_violation = np.maximum(0, -bottom_dist)
    top_violation = np.maximum(0, -top_dist)
    
    wall_energy_y = wall_k * (np.sum(bottom_violation**2) + np.sum(top_violation**2))
    
    energy += wall_energy_x + wall_energy_y
    
    # Wall gradients
    # d/dx_i [wall_k * (left_violation)^2] where left_violation = max(0, -0.5 - x_i)
    # If x_i < -0.5: gradient = wall_k * 2 * (-0.5 - x_i) * (-1) = -2 * wall_k * (-0.5 - x_i) = 2 * wall_k * (x_i + 0.5)
    # Wait, let me redo: U = wall_k * (-0.5 - x)^2 for x < -0.5
    # dU/dx = wall_k * 2 * (-0.5 - x) * (-1) = -2 * wall_k * (-0.5 - x) = 2 * wall_k * (x + 0.5)
    # But x + 0.5 < 0 when x < -0.5, so gradient is negative, pushing x toward -0.5. Good.
    
    grad[:, 0] += -2 * wall_k * left_violation  # For left wall: gradient = -2*wall_k*max(0, -0.5-x)
    grad[:, 0] += 2 * wall_k * right_violation   # For right wall: gradient = 2*wall_k*max(0, x-0.5)
    
    # Actually let me be more careful:
    # Left wall: U = wall_k * (-0.5 - x)^2 for x < -0.5
    # dU/dx = 2 * wall_k * (-0.5 - x) * (-1) = -2 * wall_k * (-0.5 - x) = 2 * wall_k * (x + 0.5)
    # When x < -0.5, (x+0.5) < 0, so gradient is negative -> pushes x right. Correct.
    # left_violation = max(0, -0.5 - x) = max(0, -(x+0.5))
    # So gradient = 2 * wall_k * (x + 0.5) = -2 * wall_k * left_violation when x < -0.5
    
    # Right wall: U = wall_k * (x - 0.5)^2 for x > 0.5
    # dU/dx = 2 * wall_k * (x - 0.5)
    # right_violation = max(0, x - 0.5)
    # gradient = 2 * wall_k * right_violation when x > 0.5
    
    # My code above has the right structure but let me verify the signs are consistent.
    # Actually, I'll just recompute cleanly:
    
    # Reset and recompute wall gradients properly
    wall_grad_x = np.zeros(N)
    wall_grad_y = np.zeros(N)
    
    # Left wall: x < -0.5
    mask_left = pts[:, 0] < -0.5
    wall_grad_x[mask_left] = -2 * wall_k * (pts[mask_left, 0] + 0.5)
    # This is negative when x < -0.5, pushing x to increase (move right). Correct.
    
    # Right wall: x > 0.5
    mask_right = pts[:, 0] > 0.5
    wall_grad_x[mask_right] = 2 * wall_k * (pts[mask_right, 0] - 0.5)
    # This is positive when x > 0.5, pushing x to decrease (move left). Correct.
    
    # Bottom wall: y < -0.5
    mask_bottom = pts[:, 1] < -0.5
    wall_grad_y[mask_bottom] = -2 * wall_k * (pts[mask_bottom, 1] + 0.5)
    
    # Top wall: y > 0.5
    mask_top = pts[:, 1] > 0.5
    wall_grad_y[mask_top] = 2 * wall_k * (pts[mask_top, 1] - 0.5)
    
    # Re-add wall gradients (overwriting the previous incorrect addition)
    # Actually, I added them incorrectly above. Let me fix by subtracting and re-adding.
    # Simpler: just compute the total gradient from scratch.
    
    # Let me rewrite this function more carefully.
    return energy, grad.flatten()

def _soft_min(vals, beta):
    """
    Compute the soft minimum of a set of values.
    soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
    """
    y = -beta * vals
    max_y = np.max(y)
    if not np.isfinite(max_y):
        return np.min(vals)
    shifted = y - max_y
    sum_exp = np.sum(np.exp(shifted))
    if sum_exp == 0:
        return np.min(vals)
    return -(max_y + np.log(sum_exp)) / beta

def _objective_and_grad(x, beta):
    """
    Objective function for minimization.
    We want to MAXIMIZE the soft-min of constraints.
    So we MINIMIZE -soft_min.
    
    x: flattened array of shape (2*N,)
    beta: sharpness parameter for soft-min
    """
    pts = x.reshape((N, 2))
    
    # Indices for pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)
    
    # Compute all constraint values C
    C = np.empty(4*N + num_pairs)
    
    # Fill wall values
    C[0:N] = pts[:, 0] + 0.5
    C[N:2*N] = 0.5 - pts[:, 0]
    C[2*N:3*N] = pts[:, 1] + 0.5
    C[3*N:4*N] = 0.5 - pts[:, 1]
    
    # Fill pair values
    D = cdist(pts, pts)
    pair_dists = D[i_idx, j_idx]
    C[4*N:] = pair_dists / 2.0
    
    # Compute soft min and weights
    y = -beta * C
    max_y = np.max(y)
    if not np.isfinite(max_y):
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)
    
    shifted = y - max_y
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp == 0:
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)
    
    soft_min_val = -(max_y + np.log(sum_exp)) / beta
    obj_val = -soft_min_val
    
    # Weights for gradient
    weights = exp_shifted / sum_exp
    
    # Compute Gradient
    grad_pts = np.zeros((N, 2))
    
    # Wall gradients
    grad_pts[:, 0] += weights[0:N] - weights[N:2*N]
    grad_pts[:, 1] += weights[2*N:3*N] - weights[3*N:4*N]
    
    # Pair gradients
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    valid = dists > 1e-10
    dx_safe = np.where(valid, dx, 1.0)
    dy_safe = np.where(valid, dy, 1.0)
    dists_safe = np.where(valid, dists, 1.0)
    
    grad_pair_x_i = np.where(valid, dx_safe / (2.0 * dists_safe), 0.0)
    grad_pair_x_j = np.where(valid, -dx_safe / (2.0 * dists_safe), 0.0)
    grad_pair_y_i = np.where(valid, dy_safe / (2.0 * dists_safe), 0.0)
    grad_pair_y_j = np.where(valid, -dy_safe / (2.0 * dists_safe), 0.0)
    
    pair_weights = weights[4*N:]
    
    np.add.at(grad_pts[:, 0], i_idx, grad_pair_x_i * pair_weights)
    np.add.at(grad_pts[:, 0], j_idx, grad_pair_x_j * pair_weights)
    np.add.at(grad_pts[:, 1], i_idx, grad_pair_y_i * pair_weights)
    np.add.at(grad_pts[:, 1], j_idx, grad_pair_y_j * pair_weights)
    
    grad_pts = -grad_pts / beta
    
    return obj_val, grad_pts.flatten()

def _generate_hex_initial():
    """
    Generate a centered hexagonal lattice packing as initial guess.
    Target spacing ~0.0732 (2 * best_known_radius) for better initial density.
    """
    s = 0.0732
    points = []
    row_h = s * np.sqrt(3) / 2
    
    y = -0.6
    row_idx = 0
    while y < 0.6:
        shift = (s / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.6 + shift
        while x < 0.6:
            points.append([x, y])
            x += s
        y += row_h
        row_idx += 1
        
    pts = np.array(points)
    
    margin = 0.001
    mask = (pts[:, 0] >= -0.5 + margin) & (pts[:, 0] <= 0.5 - margin) & \
           (pts[:, 1] >= -0.5 + margin) & (pts[:, 1] <= 0.5 - margin)
    pts = pts[mask]
    
    if len(pts) < N:
        s2 = 0.070
        points2 = []
        row_h2 = s2 * np.sqrt(3) / 2
        y = -0.6
        row_idx = 0
        while y < 0.6:
            shift = (s2 / 2.0) if row_idx % 2 == 1 else 0.0
            x = -0.6 + shift
            while x < 0.6:
                points2.append([x, y])
                x += s2
            y += row_h2
            row_idx += 1
        pts = np.array(points2)
        mask = (pts[:, 0] >= -0.5 + margin) & (pts[:, 0] <= 0.5 - margin) & \
               (pts[:, 1] >= -0.5 + margin) & (pts[:, 1] <= 0.5 - margin)
        pts = pts[mask]
        if len(pts) < N:
            return np.random.uniform(-0.4, 0.4, (N, 2))
        
    dists_to_center = np.sum(pts**2, axis=1)
    indices = np.argsort(dists_to_center)
    pts = pts[indices[:N]]
    
    return pts

def _generate_random_initial():
    """
    Generate a random initial configuration.
    """
    return np.random.uniform(-0.4, 0.4, (N, 2))

def _run_lj_optimization(pts, time_limit):
    """
    Phase 1: Lennard-Jones potential energy minimization using BFGS.
    This provides a good global relaxation that avoids local minima.
    """
    t_start = time.time()
    
    # LJ parameters: sigma should be slightly larger than target spacing
    # Target radius ~0.0366, so target spacing ~0.0732
    # Use sigma = 0.073, epsilon = 1.0, wall_k = 100
    sigma = 0.073
    epsilon = 1.0
    wall_k = 100.0
    
    x0 = pts.flatten()
    
    def lj_fun_and_grad(x):
        energy, grad = _lj_potential_and_grad(x, sigma, epsilon, wall_k)
        return energy, grad
    
    try:
        res = minimize(
            lj_fun_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
        )
        x_opt = res.x
        pts_opt = np.clip(x_opt.reshape((N, 2)), -0.5, 0.5)
        
        # Check time
        if time.time() - t_start > time_limit:
            return pts_opt
        
        return pts_opt
    except Exception:
        return pts

def _run_optimization(pts, time_limit, betas=None):
    if betas is None:
        betas = [10, 20, 50, 100, 200, 500, 1000]
        
    t_start = time.time()
    x0 = pts.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_r = _get_radius(pts)
    best_x = x0.copy()
    
    for beta in betas:
        if time.time() - t_start > time_limit:
            break
            
        def fun_and_grad(x):
            val, g = _objective_and_grad(x, beta)
            return val, g
            
        try:
            maxiter = 200 if beta >= 100 else 100
            res = minimize(
                fun_and_grad, 
                x0, 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-14, 'maxls': 50}
            )
            x0 = res.x
            pts_curr = x0.reshape((N, 2))
            pts_curr = np.clip(pts_curr, -0.5, 0.5)
            
            r = _get_radius(pts_curr)
            if r > best_r:
                best_r = r
                best_x = x0.copy()
                
        except Exception:
            pass
            
    return best_r, best_x.reshape((N, 2))

def _generate_jittered_hex(jitter_amount=0.005):
    """
    Generate a hexagonal grid with random jitter to break symmetry.
    """
    s = 0.0732
    points = []
    row_h = s * np.sqrt(3) / 2
    y = -0.5 + row_h/2
    row_idx = 0
    while y < 0.5:
        shift = (s / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.5 + s/2 + shift
        while x < 0.5:
            px = x + np.random.uniform(-jitter_amount, jitter_amount)
            py = y + np.random.uniform(-jitter_amount, jitter_amount)
            points.append([px, py])
            x += s
        y += row_h
        row_idx += 1
    pts = np.array(points)
    
    pts = np.clip(pts, -0.49, 0.49)
    
    if len(pts) > N:
        dists_to_center = np.sum(pts**2, axis=1)
        indices = np.argsort(dists_to_center)
        pts = pts[indices[:N]]
    elif len(pts) < N:
        remaining = N - len(pts)
        random_pts = np.random.uniform(-0.4, 0.4, (remaining, 2))
        pts = np.vstack([pts, random_pts])
        
    return pts

def _run_md_relaxation(pts, time_limit):
    """
    Phase 1: Repulsive Force-Directed Relaxation with Momentum.
    """
    t_start = time.time()
    x = pts.copy()
    v = np.zeros_like(x)
    damping = 0.98
    dt = 0.01
    cutoff = 0.15
    k_pair = 100.0
    k_wall = 100.0

    while time.time() - t_start < time_limit:
        F = np.zeros_like(x)
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, 1.0)
        dists = np.sqrt(dists_sq)
        
        mask = dists < cutoff
        mask[np.arange(N), np.arange(N)] = False
        
        if np.any(mask):
            factors = np.zeros_like(dists)
            factors[mask] = k_pair * (cutoff - dists[mask]) / dists[mask]
            F += np.sum(diff * factors[:, :, np.newaxis], axis=1)
            
        F[:, 0] += k_wall * np.where(x[:, 0] < -0.5, -0.5 - x[:, 0], 0)
        F[:, 0] += k_wall * np.where(x[:, 0] > 0.5, 0.5 - x[:, 0], 0)
        F[:, 1] += k_wall * np.where(x[:, 1] < -0.5, -0.5 - x[:, 1], 0)
        F[:, 1] += k_wall * np.where(x[:, 1] > 0.5, 0.5 - x[:, 1], 0)
        
        v = v * damping + F * dt
        x = x + v * dt
        x = np.clip(x, -0.5, 0.5)
        
    return x

def construct():
    t0 = time.time()
    np.random.seed(42)
    
    best_r = 0.0
    best_pts = None
    
    # Define multiple initializations using jittered hex grids
    inits = []
    inits.append(_generate_jittered_hex(0.005))
    inits.append(_generate_jittered_hex(0.010))
    inits.append(_generate_jittered_hex(0.020))
    
    num_inits = len(inits)
    
    # Phase 1: MD Relaxation
    md_time_per_init = 20.0
    
    for i, init_pts in enumerate(inits):
        elapsed = time.time() - t0
        if elapsed > _TIME_LIMIT - 60:  # Leave 60s for soft-min optimization
            break
            
        t_limit = min(md_time_per_init, _TIME_LIMIT - 60 - elapsed)
        if t_limit < 3:
            break
        
        pts_md = _run_md_relaxation(init_pts, time_limit=t_limit)
        
        # Then do soft-min optimization on the MD-relaxed points
        remaining_for_softmin = _TIME_LIMIT - 10 - (time.time() - t0)
        if remaining_for_softmin > 10:
            r_opt, pts_opt = _run_optimization(pts_md, time_limit=remaining_for_softmin)
            if r_opt > best_r:
                best_r = r_opt
                best_pts = pts_opt

    # Phase 2: Polish the best configuration with high-beta soft-min
    if best_pts is not None:
        elapsed = time.time() - t0
        remaining_time = _TIME_LIMIT - 5 - elapsed
        if remaining_time > 10:
            betas_polish = [500, 1000, 2000, 5000]
            r_polish, pts_polish = _run_optimization(best_pts, time_limit=remaining_time, betas=betas_polish)
            if r_polish > best_r:
                best_r = r_polish
                best_pts = pts_polish
                
    if best_pts is None:
        best_pts = np.random.uniform(-0.4, 0.4, (N, 2))
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
