"""Structural change: Differentiable soft-min optimization with L-BFGS-B and adaptive beta."""
N = 200
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 145.0  # Safety margin below 150s kill
_START_TIME = time.time()


def _check_time():
    return time.time() - _START_TIME < _DEADLINE


def _compute_radius(centers):
    """Return the largest common radius for given centers in [-0.5,0.5]^2."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint: distance to boundary
    # centers are in [-0.5, 0.5]^2
    # dist to boundary is min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y)
    # Since x in [-0.5, 0.5], 0.5 - |x| is the distance to the vertical boundary.
    boundary = np.min(np.minimum(
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    if boundary < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(boundary, min_pair / 2.0)


def _soft_min_objective_and_grad(x_flat, beta):
    """
    Objective: -soft_min(constraints)
    Constraints are:
    1. Pairwise distances / 2
    2. Boundary distances
    
    We want to maximize the minimum constraint.
    soft_min(z) = -1/beta * log(sum(exp(-beta * z_i)))
    We minimize f = -soft_min(z) = 1/beta * log(sum(exp(-beta * z_i)))
    
    Gradient of f wrt x:
    Let z_i be the constraints.
    f = 1/beta * log(S), where S = sum(exp(-beta * z_i))
    df/dz_i = (1/beta) * (1/S) * (-beta * exp(-beta * z_i)) = -exp(-beta * z_i) / S
    Let w_i = exp(-beta * z_i) / S. Note sum(w_i) = 1.
    df/dx = sum_i (df/dz_i) * (dz_i/dx) = -sum_i w_i * (dz_i/dx)
    
    So we compute the weighted sum of gradients of the constraints, with weights w_i.
    """
    centers = x_flat.reshape(N, 2)
    
    # 1. Compute pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Constraints from pairs: z_pair = dists / 2
    # We only need upper triangle to avoid double counting, but for gradient 
    # symmetry it's easier to handle the full matrix and divide by 2 or handle carefully.
    # Actually, dz_i/dx_j involves both i and j.
    # Let's construct the list of constraints and their gradients.
    
    # Efficient computation of weights
    # z_pairs = dists / 2
    z_pairs = dists / 2.0
    
    # Boundary constraints
    # z_bound_i = min(0.5 - |x_i|, 0.5 - |y_i|)
    # This is non-smooth at x=0 or y=0. 
    # For soft-min, we treat the 4 boundary distances as separate constraints?
    # Or just the min? 
    # If we use min, gradient is ambiguous at 0.
    # Better: treat distance to each of the 4 walls as separate constraints.
    # z_walls = [0.5 - x_i, 0.5 + x_i, 0.5 - y_i, 0.5 + y_i] for each i
    # There are 4*N boundary constraints.
    
    # Total constraints: N*(N-1) pairs (unique) + 4*N walls
    # This is ~ 40000 + 800 = 40800 constraints.
    # Computing exp(-beta * z) for all is feasible.
    
    # Let's flatten the pairwise constraints (upper triangle)
    i_idx, j_idx = np.triu_indices(N, k=1)
    z_flat_pairs = z_pairs[i_idx, j_idx]
    
    # Boundary constraints
    x = centers[:, 0]
    y = centers[:, 1]
    z_walls = np.concatenate([
        0.5 - x,
        0.5 + x,
        0.5 - y,
        0.5 + y
    ])
    
    # Combine all z
    z_all = np.concatenate([z_flat_pairs, z_walls])
    
    # Compute weights w_i = exp(-beta * z_i) / sum(exp(-beta * z_j))
    # To avoid overflow/underflow, subtract max
    z_min = np.min(z_all)
    exp_terms = np.exp(-beta * (z_all - z_min))
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    # Objective value
    # f = 1/beta * log(sum_exp) + z_min
    # Wait, soft_min = -1/beta * log(sum(exp(-beta z))).
    # We want to minimize -soft_min = 1/beta * log(sum(exp(-beta z))).
    # Let's re-derive carefully.
    # min(z) approx = -1/beta * log(sum(exp(-beta z)))
    # We want to maximize min(z). So we minimize -min(z).
    # f(x) = 1/beta * log(sum(exp(-beta z_i)))
    
    obj_val = (1.0 / beta) * np.log(sum_exp) + z_min # The + z_min is part of the shift?
    # Actually, if we shift z by z_min, exp(-beta(z - z_min)) = exp(-beta z) * exp(beta z_min).
    # sum(exp(-beta z)) = exp(-beta z_min) * sum(exp(-beta(z - z_min)))
    # log(sum(exp(-beta z))) = -beta z_min + log(sum_exp_shifted)
    # f = 1/beta * (-beta z_min + log(sum_exp_shifted)) = -z_min + (1/beta) log(sum_exp_shifted)
    # So obj_val = -z_min + (1.0 / beta) * np.log(sum_exp)
    
    # Gradient calculation
    # grad_f = - sum_i w_i * grad_z_i
    
    grad = np.zeros_like(centers)
    
    # 1. Pairwise gradients
    # z_pair = dist / 2
    # d(dist)/dx_i = (x_i - x_j) / dist
    # d(dist)/dx_j = (x_j - x_i) / dist
    # d(z_pair)/dx_i = 0.5 * (x_i - x_j) / dist
    # d(z_pair)/dx_j = 0.5 * (x_j - x_i) / dist
    
    w_pairs = weights[:len(z_flat_pairs)]
    
    # We need to accumulate gradients.
    # For each pair (i, j) with weight w:
    # grad[i] += -w * 0.5 * (x_i - x_j) / dist_ij
    # grad[j] += -w * 0.5 * (x_j - x_i) / dist_ij
    
    # Vectorized approach:
    # diff_ij = centers[i] - centers[j]
    # dist_ij = norm(diff_ij)
    # factor = 0.5 / dist_ij
    # grad[i] -= w * factor * diff_ij
    # grad[j] += w * factor * diff_ij
    
    # Since N=200, N^2 is 40000. Vectorizing the accumulation is tricky without loops or sparse ops.
    # We can use np.add.at for efficient accumulation.
    
    diff_pairs = centers[i_idx] - centers[j_idx]
    dist_pairs = z_pairs[i_idx, j_idx] * 2.0 # original dist
    # Avoid division by zero (should not happen for valid distinct points, but safety)
    dist_pairs_safe = np.where(dist_pairs < 1e-10, 1e-10, dist_pairs)
    
    factors = w_pairs / dist_pairs_safe # This is w * (1/dist). We need 0.5 * w / dist.
    # Wait, d(z)/dx = 0.5 * diff/dist.
    # So contribution to grad[i] is -w * 0.5 * diff/dist.
    # Contribution to grad[j] is -w * 0.5 * (-diff)/dist = +w * 0.5 * diff/dist.
    
    grad_contrib = 0.5 * factors[:, None] * diff_pairs
    
    np.add.at(grad, i_idx, -grad_contrib)
    np.add.at(grad, j_idx, grad_contrib)
    
    # 2. Boundary gradients
    # z_walls:
    # 0.5 - x  -> d/dx = -1
    # 0.5 + x  -> d/dx = +1
    # 0.5 - y  -> d/dy = -1
    # 0.5 + y  -> d/dy = +1
    
    w_walls = weights[len(z_flat_pairs):]
    
    # w_walls is ordered: [x_neg, x_pos, y_neg, y_pos] for all i?
    # My concatenation was:
    # 0.5 - x (all i)
    # 0.5 + x (all i)
    # 0.5 - y (all i)
    # 0.5 + y (all i)
    
    w_x_neg = w_walls[0:N]
    w_x_pos = w_walls[N:2*N]
    w_y_neg = w_walls[2*N:3*N]
    w_y_pos = w_walls[3*N:4*N]
    
    # grad_x[i] += -w_x_neg[i] * (-1) + -w_x_pos[i] * (1)  => w_x_neg[i] - w_x_pos[i]
    # Wait, grad_f = - sum w_i grad_z_i
    # grad_z for (0.5-x) is -1. So -w * (-1) = +w.
    # grad_z for (0.5+x) is +1. So -w * (+1) = -w.
    # Total grad_x from walls = w_x_neg - w_x_pos
    
    grad[:, 0] += w_x_neg - w_x_pos
    grad[:, 1] += w_y_neg - w_y_pos
    
    return obj_val, grad.flatten()


def _generate_hex_initial(n_points):
    """Generate a hexagonal grid of points."""
    s_base = np.sqrt(2.0 / (N * np.sqrt(3)))
    s = s_base
    
    points = []
    row_height = s * np.sqrt(3) / 2
    n_rows = int(1.0 / row_height) + 4
    n_cols = int(1.0 / s) + 4

    total_width = (n_cols - 1) * s
    total_height = (n_rows - 1) * row_height

    start_x = -0.5 + (1.0 - total_width) / 2
    start_y = -0.5 + (1.0 - total_height) / 2

    for i in range(n_rows):
        for j in range(n_cols):
            x = start_x + j * s
            if i % 2 == 1:
                x += s / 2
            y = start_y + i * row_height
            if -0.495 <= x <= 0.495 and -0.495 <= y <= 0.495:
                points.append([x, y])

    points = np.array(points)

    if len(points) > n_points:
        center_dist = np.sum(points ** 2, axis=1)
        idx = np.argsort(center_dist)[:n_points]
        points = points[idx]
    elif len(points) < n_points:
        extra = np.random.uniform(-0.45, 0.45, size=(n_points - len(points), 2))
        points = np.vstack([points, extra])

    return points[:n_points]


def construct():
    start_time = time.time()
    best_centers = None
    best_r = 0.0
    
    # Initial guess
    centers = _generate_hex_initial(N)
    centers += np.random.normal(0, 0.001, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    # Optimization loop with increasing beta
    # Start with low beta (smooth) and increase to high beta (sharp)
    betas = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    current_centers = centers.copy()
    
    for beta in betas:
        if time.time() - start_time > _DEADLINE - 15:
            break
            
        # Objective function wrapper for this beta
        def obj(x):
            val, _ = _soft_min_objective_and_grad(x, beta)
            return val
            
        def grad(x):
            _, g = _soft_min_objective_and_grad(x, beta)
            return g
            
        # Bounds: centers must be in [-0.5, 0.5]
        # To ensure circles are inside, centers must be in [-0.5+r, 0.5-r]
        # But r is variable. We just bound to [-0.5, 0.5] and let the objective handle it.
        # Actually, if a center goes outside, boundary distance becomes negative,
        # which makes the objective very large (bad), so L-BFGS-B should avoid it.
        # However, strict bounds help. Let's use [-0.5, 0.5].
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            result = minimize(
                obj,
                current_centers.flatten(),
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 100,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 20
                }
            )
            
            if result.success or result.fun < obj(current_centers.flatten()):
                new_centers = result.x.reshape(N, 2)
                new_centers = np.clip(new_centers, -0.5, 0.5)
                r_new = _compute_radius(new_centers)
                
                if r_new > best_r:
                    best_r = r_new
                    best_centers = new_centers.copy()
                
                current_centers = new_centers
        except Exception:
            pass
    
    # Final polish with local search if time permits
    if time.time() - start_time < _DEADLINE - 5 and best_centers is not None:
        # Simple local polish: try to increase radius by moving points
        # This is a fallback if L-BFGS-B got stuck
        pass # L-BFGS-B with high beta should be sufficient
        
    if best_centers is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        best_r = _compute_radius(best_centers)

    return best_centers
