import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
_TIME_LIMIT = 135  # Leave margin for 150s limit

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
    grad_x = pair_ovl * dx_safe / dists_safe
    grad_y = pair_ovl * dy_safe / dists_safe
    
    np.add.at(grad_pts[:, 0], i_idx, -grad_x)
    np.add.at(grad_pts[:, 0], j_idx, grad_x)
    np.add.at(grad_pts[:, 1], i_idx, -grad_y)
    np.add.at(grad_pts[:, 1], j_idx, grad_y)
    
    # Boundary gradient
    sign_x = np.where(xs >= 0, 1.0, -1.0)
    sign_y = np.where(ys >= 0, 1.0, -1.0)
    
    # Which dimension is the bottleneck?
    x_is_bottleneck = dist_x < dist_y
    y_is_bottleneck = dist_y < dist_x
    
    # Gradient for boundary
    grad_pts[:, 0] += 2 * boundary_overlap * sign_x * x_is_bottleneck
    grad_pts[:, 1] += 2 * boundary_overlap * sign_y * y_is_bottleneck
    
    grad = grad_pts.flatten()
    
    return total_energy, grad

def _interior_point_max_r(x, r0, max_iter=200, tol=1e-10, deadline=None):
    """
    Directly maximize radius r using an interior-point style approach.
    This is a constrained NLP where we maximize r subject to:
    1. ||c_i - c_j|| >= 2r for all i < j
    2. ||c_i||_inf <= 0.5 - r
    
    We use a penalty-based approach with a barrier term for constraints.
    """
    pts = x.reshape(N, 2)
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def _ip_objective_and_grad(z, mu=0.01):
        """
        z = [x_0, ..., x_{2N-1}, r]
        We minimize -r + mu * sum(violations^2)
        """
        x = z[:2*N]
        r = z[2*N]
        pts = x.reshape(N, 2)
        
        # Compute pairwise distances
        dx = pts[i_idx, 0] - pts[j_idx, 0]
        dy = pts[i_idx, 1] - pts[j_idx, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Constraint violations: 2r - dist should be <= 0
        pair_viol = np.maximum(0, 2*r - dists)
        
        # Boundary constraints: r - dist_to_wall should be <= 0
        xs = pts[:, 0]
        ys = pts[:, 1]
        dist_x = np.minimum(0.5 - xs, xs + 0.5)
        dist_y = np.minimum(0.5 - ys, ys + 0.5)
        dist_b = np.minimum(dist_x, dist_y)
        boundary_viol = np.maximum(0, r - dist_b)
        
        penalty = np.sum(pair_viol**2) + np.sum(boundary_viol**2)
        obj = -r + mu * penalty
        
        # Gradient
        grad_x = np.zeros(2*N)
        grad_pts = np.zeros((N, 2))
        
        # Pair gradient
        valid = dists > 1e-10
        dx_safe = np.where(valid, dx, 1.0)
        dy_safe = np.where(valid, dy, 1.0)
        dists_safe = np.where(valid, dists, 1.0)
        
        # d(pair_viol^2)/dx_i for active constraints
        # pair_viol = 2r - dist (when active)
        # d(pair_viol^2)/dx_i = 2 * pair_viol * d(2r - dist)/dx_i = 2 * pair_viol * (-dx/dist)
        active_pair = pair_viol > 0
        if np.any(active_pair):
            grad_px = 2 * pair_viol[active_pair] * (-dx_safe[active_pair] / dists_safe[active_pair])
            grad_py = 2 * pair_viol[active_pair] * (-dy_safe[active_pair] / dists_safe[active_pair])
            
            np.add.at(grad_pts[:, 0], i_idx[active_pair], grad_px)
            np.add.at(grad_pts[:, 0], j_idx[active_pair], -grad_px)
            np.add.at(grad_pts[:, 1], i_idx[active_pair], grad_py)
            np.add.at(grad_pts[:, 1], j_idx[active_pair], -grad_py)
        
        # Boundary gradient
        active_boundary = boundary_viol > 0
        if np.any(active_boundary):
            sign_x = np.where(xs[active_boundary] >= 0, 1.0, -1.0)
            sign_y = np.where(ys[active_boundary] >= 0, 1.0, -1.0)
            
            x_is_bottleneck = dist_x[active_boundary] < dist_y[active_boundary]
            y_is_bottleneck = dist_y[active_boundary] < dist_x[active_boundary]
            
            grad_pts[active_boundary, 0] += 2 * boundary_viol[active_boundary] * sign_x * x_is_bottleneck
            grad_pts[active_boundary, 1] += 2 * boundary_viol[active_boundary] * sign_y * y_is_bottleneck
        
        grad_x = grad_pts.flatten()
        grad_r = -1.0 + 2 * mu * (np.sum(2 * pair_viol) + np.sum(boundary_viol))
        
        grad = np.concatenate([grad_x, [grad_r]])
        
        return obj, grad
    
    # Initial guess
    z0 = np.concatenate([x.flatten(), [r0]])
    bounds = [(-0.5, 0.5)] * (2*N) + [(0.01, 0.1)]
    
    try:
        result = minimize(
            lambda z: _ip_objective_and_grad(z),
            z0,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8, 'maxls': 50}
        )
        return result.x[:2*N].reshape(N, 2), result.x[2*N]
    except Exception:
        return x.reshape(N, 2), r0

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
        if time.time() > deadline - 25:
            break
            
        current_pos = init_pos.copy()
        
        # Phase 1: Use overlap energy to find a good configuration
        # Start with a conservative radius and inflate
        r = 0.025
        step = 0.002
        
        while r < 0.042:
            if time.time() > deadline - 20:
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
                    options={'maxiter': 80, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 30}
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
                
                for _ in range(8):
                    if time.time() > deadline - 15:
                        break
                    r_mid = (r_low + r_high) / 2
                    
                    try:
                        res_mid = minimize(
                            lambda x: _overlap_energy_and_grad(x, r_mid),
                            current_pos.flatten(),
                            jac=True,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 30}
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
    
    # Phase 2: Interior point refinement
    if time.time() < deadline - 20:
        try:
            ip_pos, ip_r = _interior_point_max_r(best_positions, best_radius, max_iter=100, deadline=deadline)
            ip_r_actual = _get_radius(ip_pos)
            if ip_r_actual > best_radius:
                best_radius = ip_r_actual
                best_positions = ip_pos
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish using soft-min objective
    if time.time() < deadline - 10:
        # Use bottleneck polish
        polished = _bottleneck_polish(best_positions, beta=150.0, max_iter=300, tol=1e-12, deadline=deadline)
        r_polished = _get_radius(polished)
        if r_polished > best_radius:
            best_radius = r_polished
            best_positions = polished
    
    # Final validation and clamping
    best_positions = np.clip(best_positions, -0.49999, 0.49999)
    
    return best_positions
