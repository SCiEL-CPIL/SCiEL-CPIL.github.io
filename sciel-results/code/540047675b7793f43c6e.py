"""
Optimized packing of 50 equal spheres in a unit cube using Nesterov-accelerated
gradient ascent on the exact min-distance objective with active-set gradients.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 120.0  # seconds, safety margin from 150s limit
TIME_MARGIN = 5.0  # return this many seconds before deadline


def _compute_radius(centers):
    """Compute the exact maximum common radius for given centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Boundary constraint: min distance from center to cube face
    # Cube is [-0.5, 0.5]^3
    dist_to_face = np.min(0.5 - np.abs(centers), axis=1)
    r_boundary = np.min(dist_to_face)
    
    # Pairwise constraint: min distance between any two centers / 2
    # Use vectorized calculation for speed
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    
    return min(r_boundary, r_pair)


def _compute_radius_and_active_set(centers, tol=1e-6):
    """
    Compute radius and identify active constraints (pairs and walls) that limit the radius.
    Returns (radius, active_pairs, active_walls) where:
      active_pairs: list of (i, j) for pairs with dist/2 <= radius + tol
      active_walls: list of (i, axis) for wall constraints with dist <= radius + tol
    """
    if centers.shape != (N, 3):
        return 0.0, [], []
    if not np.all(np.isfinite(centers)):
        return 0.0, [], []
    
    # Boundary constraint
    wall_dists = 0.5 - np.abs(centers)  # shape (N, 3)
    r_boundary = np.min(wall_dists)
    
    # Pairwise constraint
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    
    radius = min(r_boundary, r_pair)
    
    # Identify active constraints
    active_pairs = []
    active_walls = []
    
    # Active pairs: dist/2 <= radius + tol
    pair_radius = dists / 2.0
    for i in range(N):
        for j in range(i + 1, N):
            if pair_radius[i, j] <= radius + tol:
                active_pairs.append((i, j))
    
    # Active walls: wall_dist <= radius + tol
    for i in range(N):
        for axis in range(3):
            if wall_dists[i, axis] <= radius + tol:
                active_walls.append((i, axis))
    
    return radius, active_pairs, active_walls


def _gradient_exact_min_radius(centers, active_pairs, active_walls):
    """
    Compute subgradient of the min-radius function at the given centers.
    The function is r(centers) = min(min_{i<j} |c_i-c_j|/2, min_{i,k} (0.5-|c_ik|)).
    
    The subgradient is the normalized sum of gradients of all active constraints.
    
    Returns gradient of shape (N, 3).
    """
    grad = np.zeros_like(centers)
    
    n_active = len(active_pairs) + len(active_walls)
    if n_active == 0:
        return grad
    
    # For each active pair (i, j), the constraint is |c_i - c_j| / 2 >= r
    # Gradient w.r.t. c_i: -(c_i - c_j) / |c_i - c_j|
    # Gradient w.r.t. c_j: +(c_i - c_j) / |c_i - c_j|
    # (normalized so that the gradient of the distance is unit length)
    
    for (i, j) in active_pairs:
        diff = centers[i] - centers[j]
        dist = np.linalg.norm(diff)
        if dist > 1e-12:
            unit_vec = diff / dist
            grad[i] -= unit_vec / 2.0  # gradient of |c_i-c_j|/2 w.r.t. c_i
            grad[j] += unit_vec / 2.0  # gradient of |c_i-c_j|/2 w.r.t. c_j
    
    # For each active wall (i, axis), the constraint is 0.5 - |c_ik| >= r
    # Gradient w.r.t. c_ik: -sign(c_ik)
    for (i, axis) in active_walls:
        if centers[i, axis] > 0:
            grad[i, axis] -= 1.0
        elif centers[i, axis] < 0:
            grad[i, axis] += 1.0
        # If exactly 0, subgradient is [-1, 1], we pick 0 (or could randomize)
    
    # Normalize the subgradient to have unit norm (for steepest ascent direction)
    norm = np.linalg.norm(grad)
    if norm > 1e-12:
        grad /= norm
    
    return grad


def _nesterov_gradient_ascent(centers, max_time, start_time, lr_init=0.01, momentum=0.9):
    """
    Nesterov-accelerated gradient ascent on the exact min-radius objective.
    Uses active-set gradient computation for efficiency.
    """
    x = centers.copy().flatten()
    v = np.zeros_like(x)  # velocity for Nesterov
    momentum = momentum
    
    best_x = x.copy()
    best_r = _compute_radius(x.reshape(N, 3))
    
    lr = lr_init
    gamma = 0.999  # learning rate decay
    
    for it in range(5000):
        # Check time
        if time.time() - start_time > max_time:
            break
        
        # Nesterov lookahead
        x_lookahead = x + momentum * v
        
        centers_la = np.clip(x_lookahead.reshape(N, 3), -0.4999, 0.4999)
        
        # Compute active set and gradient at lookahead point
        r_la, active_pairs, active_walls = _compute_radius_and_active_set(centers_la, tol=1e-4)
        grad = _gradient_exact_min_radius(centers_la, active_pairs, active_walls)
        grad_flat = grad.flatten()
        
        # Nesterov update
        v = momentum * v + lr * grad_flat
        x_new = x + v
        
        # Project back to feasible region
        x_new = np.clip(x_new, -0.5, 0.5)
        centers_new = x_new.reshape(N, 3)
        
        r_new = _compute_radius(centers_new)
        
        if r_new > best_r:
            best_r = r_new
            best_x = x_new.copy()
        
        x = x_new
        lr *= gamma
        
        # Early stopping if convergence
        if np.linalg.norm(v) < 1e-10:
            break
    
    return best_x.reshape(N, 3)


def _init_grid(n):
    """Generate an initial grid-based configuration with jitter."""
    # 4x4x4 = 64 points, pick 50
    # Spacing is 1/4 = 0.25. Radius approx 0.125.
    points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                points.append([
                    -0.5 + (i + 0.5) * 0.25,
                    -0.5 + (j + 0.5) * 0.25,
                    -0.5 + (k + 0.5) * 0.25
                ])
    points = np.array(points)
    
    # Randomly select 50 to break symmetry
    idx = np.random.choice(len(points), n, replace=False)
    points = points[idx]
    
    # Add small random jitter to break symmetry and help escape local minima
    points += np.random.randn(n, 3) * 0.005
    points = np.clip(points, -0.49, 0.49)
    return points


def _init_random(n):
    """Generate a random initial configuration."""
    return np.random.uniform(-0.45, 0.45, (n, 3))


def _smooth_objective(x, beta=50.0):
    """
    Smooth approximation of the negative minimum radius (for L-BFGS-B fallback).
    """
    centers = x.reshape(N, 3)
    
    wall_dists = 0.5 - np.abs(centers)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 1e10)
    dists = np.sqrt(dists_sq)
    
    triu_idx = np.triu_indices(N, k=1)
    pair_dists = dists[triu_idx] / 2.0
    
    wall_vals = (0.5 - np.abs(centers)).flatten()
    
    all_vals = np.concatenate([pair_dists, wall_vals])
    
    max_val = np.max(all_vals)
    smooth_min = max_val - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (all_vals - max_val))))
    
    return -smooth_min


def _optimize_lbfgsb(points, max_time, start_time):
    """
    Optimize using L-BFGS-B on the smooth objective (fallback/comparison).
    """
    x0 = points.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_x = x0.copy()
    best_f = _smooth_objective(x0, beta=50.0)
    
    try:
        res = minimize(
            _smooth_objective,
            x0,
            args=(50.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
    except Exception:
        pass
    
    if time.time() - start_time > max_time:
        return best_x.reshape(N, 3)
    
    def _obj_fine(x):
        return _smooth_objective(x, beta=200.0)
    
    try:
        res2 = minimize(
            _obj_fine,
            best_x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res2.fun < best_f:
            best_f = res2.fun
            best_x = res2.x
    except Exception:
        pass
        
    return best_x.reshape(N, 3)


def _polish(centers, max_time, start_time):
    """
    Final high-precision polish using alternating L-BFGS-B and Nesterov steps.
    """
    best = centers.copy()
    best_r = _compute_radius(best)
    
    for _ in range(3):
        if time.time() - start_time > max_time:
            break
        
        # Try L-BFGS-B polish
        polished = _optimize_lbfgsb(best, min(5.0, max_time - (time.time() - start_time)), start_time)
        polished = np.clip(polished, -0.5, 0.5)
        r = _compute_radius(polished)
        if r > best_r:
            best_r = r
            best = polished
        
        if time.time() - start_time > max_time:
            break
        
        # Try Nesterov polish
        polished2 = _nesterov_gradient_ascent(best, min(3.0, max_time - (time.time() - start_time)), start_time, lr_init=0.005, momentum=0.8)
        polished2 = np.clip(polished2, -0.5, 0.5)
        r2 = _compute_radius(polished2)
        if r2 > best_r:
            best_r = r2
            best = polished2
    
    return best


def construct():
    start_time = time.time()
    
    best_centers = None
    best_radius = 0.0
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return DEADLINE - TIME_MARGIN - elapsed()
    
    # Define initializations
    np.random.seed(42)
    inits = []
    
    # 1. Grid based
    inits.append(_init_grid(N))
    
    # 2. Random
    inits.append(_init_random(N))
    
    # 3. Layered structure
    layers = []
    for layer_idx in range(5):
        z_pos = (layer_idx + 0.5) / 5.0 - 0.5
        for i in range(10):
            row = i // 5
            col = i % 5
            x = (col + 0.5) / 5.0 - 0.5
            y = (row + 0.5) / 2.0 - 0.5
            layers.append([x, y, z_pos])
    inits.append(np.array(layers[:N]))
    
    # 4. Another random with different seed
    np.random.seed(123)
    inits.append(_init_random(N))
    
    # 5. BCC-inspired initialization
    bcc_points = []
    # 4x4x4 grid with BCC offset
    for i in range(4):
        for j in range(4):
            for k in range(4):
                bcc_points.append([
                    -0.5 + (i + 0.5) * 0.25,
                    -0.5 + (j + 0.5) * 0.25,
                    -0.5 + (k + 0.5) * 0.25
                ])
                # Add BCC offset point
                bcc_points.append([
                    -0.5 + (i + 0.5) * 0.25 + 0.125,
                    -0.5 + (j + 0.5) * 0.25 + 0.125,
                    -0.5 + (k + 0.5) * 0.25 + 0.125
                ])
    bcc_array = np.array(bcc_points)
    idx = np.random.choice(len(bcc_array), N, replace=False)
    bcc_array = bcc_array[idx]
    bcc_array += np.random.randn(N, 3) * 0.003
    bcc_array = np.clip(bcc_array, -0.49, 0.49)
    inits.append(bcc_array)
    
    for init_idx, init in enumerate(inits):
        if remaining() < 10.0:
            break
        
        # Allocate time for this initialization
        max_opt_time = min(15.0, remaining())
        
        # First try L-BFGS-B (fast, good for coarse optimization)
        opt_centers = _optimize_lbfgsb(init, min(8.0, max_opt_time), start_time)
        opt_centers = np.clip(opt_centers, -0.5, 0.5)
        
        r = _compute_radius(opt_centers)
        
        if r > best_radius:
            best_radius = r
            best_centers = opt_centers.copy()
        
        # Then try Nesterov gradient ascent from the L-BFGS-B result
        if remaining() > 5.0:
            nesterov_centers = _nesterov_gradient_ascent(
                opt_centers, 
                min(7.0, remaining()), 
                start_time,
                lr_init=0.008,
                momentum=0.9
            )
            nesterov_centers = np.clip(nesterov_centers, -0.5, 0.5)
            r_nest = _compute_radius(nesterov_centers)
            
            if r_nest > best_radius:
                best_radius = r_nest
                best_centers = nesterov_centers.copy()
    
    # Final polish phase
    if best_centers is not None and remaining() > 5.0:
        polished = _polish(best_centers, min(20.0, remaining()), start_time)
        polished = np.clip(polished, -0.5, 0.5)
        r_pol = _compute_radius(polished)
        if r_pol > best_radius:
            best_radius = r_pol
            best_centers = polished
    
    if best_centers is None:
        best_centers = _init_grid(N)
    
    return best_centers


if __name__ == "__main__":
    centers = construct()
    print(f"Score: {_compute_radius(centers)}")
    print(f"Shape: {centers.shape}")
