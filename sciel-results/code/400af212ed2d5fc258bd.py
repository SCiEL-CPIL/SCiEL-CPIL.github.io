import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
_TIME_LIMIT = 140  # Leave margin for 150s limit

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
    wall_vals = np.concatenate([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ])
    
    # Pair terms
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)
    
    # Pair distances
    D = cdist(pts, pts)
    pair_dists = D[i_idx, j_idx]
    pair_vals = pair_dists / 2.0
    
    # Combine all constraint values into a single array C
    C = np.concatenate([wall_vals, pair_vals])
    
    # Step 2: Compute soft min and weights
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
    
    # Step 3: Compute Gradient
    grad = np.zeros_like(x)
    grad_pts = np.zeros((N, 2))
    
    # Wall gradients
    grad_pts[:, 0] += weights[0:N] * 1.0
    grad_pts[:, 0] += weights[N:2*N] * (-1.0)
    grad_pts[:, 1] += weights[2*N:3*N] * 1.0
    grad_pts[:, 1] += weights[3*N:4*N] * (-1.0)
    
    # Pair gradients
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    # Avoid division by zero
    valid = dists > 1e-10
    dx_safe = np.where(valid, dx, 1.0)
    dy_safe = np.where(valid, dy, 1.0)
    dists_safe = np.where(valid, dists, 1.0)
    
    w_pair = weights[4*N:]
    grad_x = w_pair * dx_safe / (2 * dists_safe)
    grad_y = w_pair * dy_safe / (2 * dists_safe)
    
    # Use np.add.at for accumulation
    np.add.at(grad_pts[:, 0], i_idx, grad_x)
    np.add.at(grad_pts[:, 0], j_idx, -grad_x)
    np.add.at(grad_pts[:, 1], i_idx, grad_y)
    np.add.at(grad_pts[:, 1], j_idx, -grad_y)
    
    # Reshape grad_pts back to grad
    grad = grad_pts.flatten()
    
    return obj_val, grad

def _bottleneck_polish(positions, beta=100.0, max_iter=500, tol=1e-12, deadline=None):
    """
    High-precision polish pass using the soft-min objective with analytical gradients.
    """
    x0 = positions.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        result = minimize(
            lambda x: _objective_and_grad(x, beta),
            x0,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10, 'maxls': 50}
        )
        return result.x.reshape(N, 2)
    except Exception:
        return positions

def _overlap_energy_and_grad(x, r):
    """
    Objective function for overlap minimization.
    We want to minimize the sum of squared overlaps.
    This is a smooth approximation to the constraint satisfaction problem.
    
    x: flattened array of shape (2*N,)
    r: target radius
    """
    pts = x.reshape((N, 2))
    
    # Compute pairwise distances
    D = cdist(pts, pts)
    np.fill_diagonal(D, 2*r + 1)  # Ensure diagonal doesn't contribute
    
    # Pairwise overlaps
    # Overlap amount = 2r - d if d < 2r else 0
    # Squared overlap = (2r - d)^2 if d < 2r else 0
    pair_overlap = np.maximum(0, 2*r - D)
    pair_energy = 0.5 * np.sum(pair_overlap**2)  # 0.5 to simplify gradient
    
    # Boundary overlaps
    xs = pts[:, 0]
    ys = pts[:, 1]
    
    dist_x = np.minimum(0.5 - xs, xs + 0.5)
    dist_y = np.minimum(0.5 - ys, ys + 0.5)
    dist_b = np.minimum(dist_x, dist_y)
    
    boundary_overlap = np.maximum(0, r - dist_b)
    boundary_energy = np.sum(boundary_overlap**2)
    
    total_energy = pair_energy + boundary_energy
    
    # Gradient computation
    grad_pts = np.zeros((N, 2))
    
    # Pairwise gradient
    # d(pair_overlap^2)/dx_i = 2 * pair_overlap * d(2r - d)/dx_i
    # d(2r - d)/dx_i = -dx / d (for i != j)
    # So d(pair_overlap^2)/dx_i = -2 * pair_overlap * dx / d
    # For the sum over all pairs, we need to accumulate
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    valid = dists > 1e-10
    dx_safe = np.where(valid, dx, 1.0)
    dy_safe = np.where(valid, dy, 1.0)
    dists_safe = np.where(valid, dists, 1.0)
    
    # Overlap values for pairs
    pair_ovl = np.maximum(0, 2*r - dists)
    
    # Gradient contribution from each pair
    # For point i: -pair_ovl * dx / dist
    # For point j: +pair_ovl * dx / dist
    grad_x = pair_ovl * dx_safe / dists_safe
    grad_y = pair_ovl * dy_safe / dists_safe
    
    np.add.at(grad_pts[:, 0], i_idx, -grad_x)
    np.add.at(grad_pts[:, 0], j_idx, grad_x)
    np.add.at(grad_pts[:, 1], i_idx, -grad_y)
    np.add.at(grad_pts[:, 1], j_idx, grad_y)
    
    # Boundary gradient
    # d(boundary_overlap^2)/dx = 2 * boundary_overlap * d(r - dist_b)/dx
    # d(r - dist_b)/dx = -d(dist_b)/dx
    # dist_b = min(dist_x, dist_y)
    # If dist_x < dist_y, dist_b = dist_x, d(dist_b)/dx = d(dist_x)/dx
    # dist_x = min(0.5 - x, x + 0.5)
    # If 0.5 - x < x + 0.5 (i.e., x > 0), dist_x = 0.5 - x, d(dist_x)/dx = -1
    # If x < 0, dist_x = x + 0.5, d(dist_x)/dx = 1
    # Similarly for y
    
    # For simplicity, compute numerically or use subgradients
    # Let's use a smooth approximation: treat dist_b as differentiable
    # Actually, let's compute the gradient more carefully
    
    # For x dimension:
    # dist_x = min(0.5 - x, x + 0.5)
    # If x >= 0: dist_x = 0.5 - x, d(dist_x)/dx = -1
    # If x < 0: dist_x = x + 0.5, d(dist_x)/dx = 1
    # So d(dist_x)/dx = -sign(x)
    
    # dist_y similarly
    
    # dist_b = min(dist_x, dist_y)
    # If dist_x < dist_y: d(dist_b)/dx = d(dist_x)/dx = -sign(x)
    # If dist_y < dist_x: d(dist_b)/dx = 0
    
    # So d(dist_b)/dx = -sign(x) * (1 if dist_x < dist_y else 0)
    # d(dist_b)/dy = -sign(y) * (1 if dist_y < dist_x else 0)
    
    # d(boundary_overlap^2)/dx = 2 * boundary_overlap * (-d(dist_b)/dx)
    # = 2 * boundary_overlap * sign(x) * (1 if dist_x < dist_y else 0)
    
    # Let's compute this
    sign_x = np.where(xs >= 0, 1.0, -1.0)
    sign_y = np.where(ys >= 0, 1.0, -1.0)
    
    # Which dimension is the bottleneck?
    # dist_x < dist_y means x is closer to boundary than y
    x_is_bottleneck = dist_x < dist_y
    y_is_bottleneck = dist_y < dist_x
    
    # Gradient for boundary
    # d(boundary_overlap^2)/dx = 2 * boundary_overlap * sign(x) * x_is_bottleneck
    # d(boundary_overlap^2)/dy = 2 * boundary_overlap * sign(y) * y_is_bottleneck
    
    grad_pts[:, 0] += 2 * boundary_overlap * sign_x * x_is_bottleneck
    grad_pts[:, 1] += 2 * boundary_overlap * sign_y * y_is_bottleneck
    
    grad = grad_pts.flatten()
    
    return total_energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    # Hexagonal grid initialization
    def hexagonal_init(n, r_target=0.036):
        s = 2 * r_target * 1.001
        points = []
        for i in range(-15, 15):
            for j in range(-15, 15):
                x = j * s + (i % 2) * s / 2
                y = i * s * np.sqrt(3) / 2
                points.append([x, y])
        
        points = np.array(points)
        mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & \
               (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
        points = points[mask]
        
        if len(points) > n:
            dists_to_center = np.sum(points**2, axis=1)
            dist_x = np.minimum(0.5 - points[:, 0], points[:, 0] + 0.5)
            dist_y = np.minimum(0.5 - points[:, 1], points[:, 1] + 0.5)
            dist_b = np.minimum(dist_x, dist_y)
            score = dists_to_center - 0.1 * dist_b
            idx = np.argsort(score)[:n]
            points = points[idx]
        elif len(points) < n:
            needed = n - len(points)
            extra = np.random.uniform(-0.4, 0.4, size=(needed, 2))
            points = np.vstack([points, extra])
            
        return points

    def square_grid_init(n):
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]

    best_radius = -np.inf
    best_positions = None
    
    # Multiple initializations
    np.random.seed(42)
    inits = [
        hexagonal_init(N, 0.036),
        hexagonal_init(N, 0.035),
        square_grid_init(N),
        np.random.uniform(-0.45, 0.45, size=(N, 2))
    ]
    
    for init_pos in inits:
        if time.time() > deadline - 20:
            break
            
        current_pos = init_pos.copy()
        
        # Phase 1: Use overlap energy to find a good configuration
        # Start with a conservative radius and inflate
        r = 0.025
        step = 0.002
        
        while r < 0.042:
            if time.time() > deadline - 15:
                break
                
            # Optimize with overlap energy objective
            x0 = current_pos.flatten()
            bounds = [(-0.5, 0.5)] * (2 * N)
            
            try:
                result = minimize(
                    lambda x: _overlap_energy_and_grad(x, r),
                    x0,
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 30}
                )
                current_pos = result.x.reshape(N, 2)
            except Exception:
                break
            
            # Check feasibility
            actual_r = _get_radius(current_pos)
            
            if actual_r >= r:
                r += step
            else:
                # Binary search for max feasible r
                r_low = r - step
                r_high = r
                
                for _ in range(10):
                    if time.time() > deadline - 10:
                        break
                    r_mid = (r_low + r_high) / 2
                    
                    try:
                        res_mid = minimize(
                            lambda x: _overlap_energy_and_grad(x, r_mid),
                            current_pos.flatten(),
                            jac=True,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 60, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 30}
                        )
                        current_pos = res_mid.x.reshape(N, 2)
                        mid_r = _get_radius(current_pos)
                        if mid_r >= r_mid:
                            r_low = r_mid
                        else:
                            r_high = r_mid
                    except Exception:
                        break
                
                if r_low > best_radius:
                    best_radius = r_low
                    best_positions = current_pos.copy()
                
                break
        
        if r >= 0.042:
            actual_r = _get_radius(current_pos)
            if actual_r > best_radius:
                best_radius = actual_r
                best_positions = current_pos.copy()

    # Fallback
    if best_positions is None:
        best_positions = square_grid_init(N)
        best_radius = _get_radius(best_positions)
    
    # Phase 2: Final high-precision polish using soft-min objective
    if time.time() < deadline - 15:
        # Use bottleneck polish
        polished = _bottleneck_polish(best_positions, beta=100.0, max_iter=500, tol=1e-12, deadline=deadline)
        r_polished = _get_radius(polished)
        if r_polished > best_radius:
            best_radius = r_polished
            best_positions = polished
    
    # Final validation and clamping
    best_positions = np.clip(best_positions, -0.49999, 0.49999)
    
    return best_positions
