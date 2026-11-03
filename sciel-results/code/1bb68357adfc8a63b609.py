"""
Pack 200 equal circles in a unit square using gradient-based optimization 
with a soft-min objective to handle the max-min constraint.
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
    # pts is (N, 2)
    # min_x = np.min(pts[:, 0]) -> dist to left wall = min_x - (-0.5) = min_x + 0.5
    # max_x = np.max(pts[:, 0]) -> dist to right wall = 0.5 - max_x
    # min_y = np.min(pts[:, 1]) -> dist to bottom wall = min_y + 0.5
    # max_y = np.max(pts[:, 1]) -> dist to top wall = 0.5 - max_y
    
    wall_dists = np.array([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])
    
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # cdist is O(N^2) which is 40000 for N=200. This is fast enough.
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0
    
    return min(min_wall, min_pair)

def _soft_min(vals, beta):
    """
    Compute the soft minimum of a set of values.
    soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
    """
    # Handle numerical stability
    # We want to minimize the negative of the soft min, so we maximize the soft min.
    # To avoid overflow in exp, we shift by the max value.
    # Note: for soft min, we use -beta * x. 
    # Let y_i = -beta * x_i.
    # soft_min = -1/beta * log(sum(exp(y_i)))
    # max_y = max(y_i) = -beta * min(x_i)
    # sum(exp(y_i)) = exp(max_y) * sum(exp(y_i - max_y))
    # log(sum) = max_y + log(sum(exp(y_i - max_y)))
    # soft_min = -1/beta * (max_y + log(sum(exp(y_i - max_y))))
    
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
    
    # Calculate constraints
    # 1. Wall distances
    # dist to left: x_i + 0.5
    # dist to right: 0.5 - x_i
    # dist to bottom: y_i + 0.5
    # dist to top: 0.5 - y_i
    
    # We can compute these vectorially
    # But for gradient, it's easier to think about individual terms.
    
    # Let's compute the values for the soft min
    # There are 4*N wall terms and N*(N-1)/2 pair terms.
    
    # Wall terms
    wall_vals = np.concatenate([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ])
    
    # Pair terms
    # We need distances between all pairs i < j
    # Using cdist is convenient for values, but for gradients, we need to be careful.
    # However, L-BFGS-B uses finite differences or analytical gradients if provided.
    # Providing analytical gradients is much faster and more stable.
    
    # Let's compute the soft min value and its gradient w.r.t x.
    
    # Combine all constraint values into a single array C
    # C has size 4N + N(N-1)/2
    # For N=200, this is 800 + 19900 = 20700 values. This is manageable.
    
    # We will compute the values and the Jacobian (gradient of C w.r.t x)
    # Then grad_obj = -1/beta * sum( w_i * grad_C_i ) where w_i are weights from log-sum-exp.
    
    # Step 1: Compute C
    # Wall parts are easy.
    # Pair parts:
    # D = cdist(pts, pts)
    # We need upper triangle indices.
    
    # Instead of building the full C array explicitly which might be memory heavy or slow to index,
    # let's compute the log-sum-exp components directly.
    
    # log_sum_exp(-beta * C) = max(-beta*C) + log(sum(exp(-beta*C - max(-beta*C))))
    
    # Let's identify the critical terms.
    # Actually, for N=200, building the full C array is fine (20k floats).
    
    # Indices for pairs
    # i < j
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)
    
    # Pair distances
    # diff = pts[i] - pts[j]
    # dist = norm(diff)
    # val = dist / 2
    
    # Let's compute all values C
    C = np.empty(4*N + num_pairs)
    
    # Fill wall values
    C[0:N] = pts[:, 0] + 0.5
    C[N:2*N] = 0.5 - pts[:, 0]
    C[2*N:3*N] = pts[:, 1] + 0.5
    C[3*N:4*N] = 0.5 - pts[:, 1]
    
    # Fill pair values
    # Vectorized distance calculation for pairs
    # This is a bit tricky to do purely vectorized without loops for the norm, 
    # but we can use cdist and extract the upper triangle.
    D = cdist(pts, pts)
    # Extract upper triangle
    pair_dists = D[i_idx, j_idx]
    C[4*N:] = pair_dists / 2.0
    
    # Step 2: Compute soft min and weights
    y = -beta * C
    max_y = np.max(y)
    if not np.isfinite(max_y):
        # Fallback to hard min if numerical issues
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
    # w_i = exp(y_i) / sum(exp(y_j)) = exp(shifted_i) / sum_exp
    weights = exp_shifted / sum_exp
    
    # Step 3: Compute Gradient
    # grad_obj = - (1/beta) * sum( w_i * grad_C_i )
    # Note: grad_C_i is the gradient of the i-th constraint value w.r.t x.
    
    grad = np.zeros_like(x)
    
    # Wall gradients
    # C[0:N] = x_0 + 0.5 -> grad is [1, 0] for each point's x
    # C[N:2*N] = 0.5 - x_0 -> grad is [-1, 0]
    # C[2*N:3*N] = x_1 + 0.5 -> grad is [0, 1]
    # C[3*N:4*N] = 0.5 - x_1 -> grad is [0, -1]
    
    # We can accumulate these efficiently
    # x-coord contributions
    grad_x_left = weights[0:N] * 1.0
    grad_x_right = weights[N:2*N] * (-1.0)
    grad_y_bottom = weights[2*N:3*N] * 1.0
    grad_y_top = weights[3*N:4*N] * (-1.0)
    
    # Add to grad array
    # x is (2N,). Indices 0, 2, 4... are x-coords. 1, 3, 5... are y-coords.
    # Let's reshape to (N, 2) for easier handling, then flatten back.
    grad_pts = np.zeros((N, 2))
    grad_pts[:, 0] += grad_x_left + grad_x_right
    grad_pts[:, 1] += grad_y_bottom + grad_y_top
    
    # Pair gradients
    # C[4*N + k] = dist(i, j) / 2
    # dist = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
    # grad_x_i = (x_i - x_j) / (2 * dist)
    # grad_x_j = (x_j - x_i) / (2 * dist)
    # Similarly for y.
    
    # We need to iterate over pairs or use vectorized accumulation.
    # With ~20k pairs, a loop might be slow in Python. 
    # Let's try to vectorize the accumulation using np.add.at
    
    # Get the coordinates of the pairs
    # i_idx, j_idx are the indices of points.
    # x_i = pts[i_idx, 0], etc.
    
    # Calculate differences
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    # Avoid division by zero
    valid = dists > 1e-10
    dx_safe = np.where(valid, dx, 1.0)
    dy_safe = np.where(valid, dy, 1.0)
    dists_safe = np.where(valid, dists, 1.0)
    
    # Gradients for each pair term
    # d(dist/2)/dx_i = dx / (2 * dist)
    # d(dist/2)/dx_j = -dx / (2 * dist)
    
    grad_pair_x_i = dx_safe / (2.0 * dists_safe)
    grad_pair_x_j = -dx_safe / (2.0 * dists_safe)
    grad_pair_y_i = dy_safe / (2.0 * dists_safe)
    grad_pair_y_j = -dy_safe / (2.0 * dists_safe)
    
    # Mask out invalid pairs (dist ~ 0)
    grad_pair_x_i = np.where(valid, grad_pair_x_i, 0.0)
    grad_pair_x_j = np.where(valid, grad_pair_x_j, 0.0)
    grad_pair_y_i = np.where(valid, grad_pair_y_i, 0.0)
    grad_pair_y_j = np.where(valid, grad_pair_y_j, 0.0)
    
    # Accumulate weights
    pair_weights = weights[4*N:]
    
    # np.add.at for accumulating gradients into the correct point indices
    np.add.at(grad_pts[:, 0], i_idx, grad_pair_x_i * pair_weights)
    np.add.at(grad_pts[:, 0], j_idx, grad_pair_x_j * pair_weights)
    np.add.at(grad_pts[:, 1], i_idx, grad_pair_y_i * pair_weights)
    np.add.at(grad_pts[:, 1], j_idx, grad_pair_y_j * pair_weights)
    
    # Total gradient
    # Remember: we are minimizing -soft_min.
    # d(-soft_min)/dx = - (1/beta) * sum(w_i * dC_i/dx)
    # The sum calculated above is sum(w_i * dC_i/dx).
    # So we need to negate and divide by beta.
    
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
    
    # Generate a grid larger than the square to ensure we have enough points
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
    
    # Filter points inside the square (with small margin for safety)
    margin = 0.001
    mask = (pts[:, 0] >= -0.5 + margin) & (pts[:, 0] <= 0.5 - margin) & \
           (pts[:, 1] >= -0.5 + margin) & (pts[:, 1] <= 0.5 - margin)
    pts = pts[mask]
    
    if len(pts) < N:
        # Fallback: generate more points with slightly larger spacing
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
        
    # Select N points closest to the center for balanced distribution
    dists_to_center = np.sum(pts**2, axis=1)
    indices = np.argsort(dists_to_center)
    pts = pts[indices[:N]]
    
    return pts

def _generate_random_initial():
    """
    Generate a random initial configuration.
    """
    return np.random.uniform(-0.4, 0.4, (N, 2))

def _run_optimization(pts, time_limit, betas=None):
    if betas is None:
        betas = [5, 10, 20, 50, 100, 200, 400]
        
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
            # Use more iterations for higher betas (finer optimization)
            maxiter = 150 if beta >= 100 else 100
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

def _generate_hex_initial_with_spacing(s):
    points = []
    row_h = s * np.sqrt(3) / 2
    y = -0.5 + row_h/2
    row_idx = 0
    while y < 0.5:
        shift = (s / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.5 + s/2 + shift
        while x < 0.5:
            points.append([x, y])
            x += s
        y += row_h
        row_idx += 1
    pts = np.array(points)
    if len(pts) >= N:
        return pts[:N]
    else:
        # Fallback to random if hex grid fails
        return np.random.uniform(-0.4, 0.4, (N, 2))

def construct():
    t0 = time.time()
    np.random.seed(42)
    
    best_r = 0.0
    best_pts = None
    
    # Phase 1: Hex Initial (primary approach)
    pts_hex = _generate_hex_initial()
    if pts_hex is not None and len(pts_hex) == N:
        remaining = _TIME_LIMIT - (time.time() - t0)
        if remaining > 30:
            r_hex, pts_hex_opt = _run_optimization(pts_hex, time_limit=min(80, remaining - 20))
            if r_hex > best_r:
                best_r = r_hex
                best_pts = pts_hex_opt
    
    # Phase 2: Second hex with different offset
    if best_pts is not None and time.time() - t0 < _TIME_LIMIT - 30:
        # Generate a shifted hex grid
        s = 0.0732
        points = []
        row_h = s * np.sqrt(3) / 2
        y = -0.6 + row_h / 2
        row_idx = 0
        while y < 0.6:
            shift = (s / 2.0) if row_idx % 2 == 0 else 0.0
            x = -0.6 + shift
            while x < 0.6:
                points.append([x, y])
                x += s
            y += row_h
            row_idx += 1
        pts_shifted = np.array(points)
        margin = 0.001
        mask = (pts_shifted[:, 0] >= -0.5 + margin) & (pts_shifted[:, 0] <= 0.5 - margin) & \
               (pts_shifted[:, 1] >= -0.5 + margin) & (pts_shifted[:, 1] <= 0.5 - margin)
        pts_shifted = pts_shifted[mask]
        if len(pts_shifted) >= N:
            dists_to_center = np.sum(pts_shifted**2, axis=1)
            indices = np.argsort(dists_to_center)
            pts_shifted = pts_shifted[indices[:N]]
            remaining = _TIME_LIMIT - (time.time() - t0)
            if remaining > 20:
                r_shift, pts_shift_opt = _run_optimization(pts_shifted, time_limit=min(40, remaining - 15))
                if r_shift > best_r:
                    best_r = r_shift
                    best_pts = pts_shift_opt
    
    # Phase 3: High-precision polish on best configuration
    if best_pts is not None and time.time() - t0 < _TIME_LIMIT - 10:
        betas_polish = [500, 1000, 2000]
        remaining_time = _TIME_LIMIT - (time.time() - t0) - 5
        if remaining_time > 0:
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
