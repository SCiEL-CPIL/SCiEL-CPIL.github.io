"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    wall_dist = 0.5 - np.abs(centres)
    min_wall_dist = np.min(wall_dist)
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def _neg_radius(centres):
    """Negative of the radius for minimization."""
    return -_compute_radius(centres)


def _neg_radius_grad(centres):
    """
    Gradient of the negative radius.
    We want to maximize min(pair_dist/2, wall_dist).
    The gradient of min is in the direction of the constraint that is tightest.
    For simplicity, we approximate the gradient by summing gradients of all constraints
    weighted by how close they are to the minimum (soft-min gradient approximation).
    """
    n = len(centres)
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    
    wall_dist = 0.5 - np.abs(centres)
    
    # All constraints: pair distances / 2 and wall distances
    # We want gradient of -min(constraints)
    # Approximation: gradient = - sum_j w_j * grad(constraint_j) / sum(w_j)
    # where w_j = exp(-beta * constraint_j)
    
    beta = 50.0
    
    # Pairwise part
    # constraint_ij = dist_ij / 2
    # grad_i(constraint_ij) = -0.5 * (c_i - c_j) / dist_ij  (since d(dist)/dc_i = (c_i-c_j)/dist)
    # Wait, dist = |c_i - c_j|. d(dist)/dc_i = (c_i - c_j)/|c_i - c_j|.
    # So grad_i(dist/2) = 0.5 * (c_i - c_j) / dist_ij
    # We want to increase the minimum, so we move in direction of gradient of the min.
    # For minimization of -min, we move in direction of -grad(min).
    # But scipy minimize minimizes f. If f = -min, then grad f = -grad(min).
    # So we want to return -grad(min).
    
    # Let's compute the gradient of the soft-min surrogate for the radius.
    # R_soft = -1/beta * log(sum(exp(-beta * x_j)))
    # grad_i R_soft = sum_j (w_j / sum(w_j)) * grad_i x_j
    # where w_j = exp(-beta * x_j)
    
    # x_j are the constraints: dist_ij/2 and wall_dist_ik
    # grad_i (dist_ij/2) = 0.5 * (c_i - c_j) / dist_ij
    # grad_i (wall_dist_ik) = -sign(c_ik)
    
    # We want grad_i (-R_soft) = - sum_j (w_j / sum(w_j)) * grad_i x_j
    
    # Initialize gradient
    grad = np.zeros_like(centres)
    
    # Pairwise contributions
    mask = (dist > 1e-10)
    weights = np.zeros_like(dist)
    weights[mask] = np.exp(-beta * dist[mask] / 2.0)
    
    # Normalize weights? The soft-min gradient formula has sum(w_j) in denominator.
    # Since we are just using this as a direction for gradient descent, scaling doesn't matter much,
    # but let's keep it somewhat normalized to avoid huge gradients.
    
    # grad_i (dist_ij/2) = 0.5 * diff[i,j] / dist[i,j]
    # Contribution to grad_i (-R_soft) from pair (i,j) is:
    # - (w_ij / sum_all_w) * 0.5 * diff[i,j] / dist[i,j]
    
    # Let's compute the unnormalized sum first
    # sum_j w_ij * 0.5 * diff[i,j] / dist[i,j]
    
    inv_dist = np.zeros_like(dist)
    inv_dist[mask] = 1.0 / dist[mask]
    
    # weight_matrix[i,j] = w_ij * 0.5 * inv_dist[i,j]
    w_mat = weights * 0.5 * inv_dist
    np.fill_diagonal(w_mat, 0)
    
    # grad_pair[i] = - sum_j w_mat[i,j] * diff[i,j]
    grad_pair = -np.sum(w_mat[:, :, np.newaxis] * diff, axis=1)
    
    # Wall contributions
    # x_k = wall_dist_k = 0.5 - |c_k|
    # grad_i (x_k) = -sign(c_ik)
    # Contribution to grad_i (-R_soft) from wall k:
    # - (w_ik / sum_all_w) * (-sign(c_ik)) = (w_ik / sum_all_w) * sign(c_ik)
    
    wall_weights = np.exp(-beta * wall_dist)
    
    # grad_wall[i] = sum_k (wall_weights[i,k] * sign(c[i,k]))
    # Note: the normalization factor sum_all_w is common to all terms, so we can ignore it for direction.
    
    grad_wall = np.sum(wall_weights * np.sign(centres), axis=1, keepdims=True)
    # Wait, np.sign(centres) is shape (N,3). wall_weights is (N,3).
    # grad_wall[i,k] = wall_weights[i,k] * sign(c[i,k])
    # So grad_wall = wall_weights * np.sign(centres)
    
    grad_wall = wall_weights * np.sign(centres)
    
    # Total gradient of -R_soft
    # We need to combine grad_pair and grad_wall.
    # The soft-min is over ALL constraints. So the weights should be normalized over the full set.
    # sum_all_w = sum(pair_weights) + sum(wall_weights)
    
    total_w = np.sum(weights) + np.sum(wall_weights)
    
    if total_w > 0:
        grad_pair_normalized = grad_pair / total_w
        grad_wall_normalized = grad_wall / total_w
    else:
        grad_pair_normalized = grad_pair
        grad_wall_normalized = grad_wall
    
    grad = grad_pair_normalized + grad_wall_normalized
    
    return grad


def _fibonacci_lattice_init(n_points, n_dims=3):
    """
    Generate points on a Fibonacci lattice (spherical code) scaled to fit in the cube.
    This provides good initial separation.
    """
    # Use the golden angle spiral on a sphere
    phi = np.pi * (3.0 - np.sqrt(5.0))
    z = np.linspace(-1, 1, n_points)
    radius = np.sqrt(1 - z**2)
    theta = phi * np.arange(n_points)
    
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    points = np.column_stack([x, y, z])
    
    # Scale to fit in [-0.5, 0.5]^3
    # Max coordinate is 1. Scale by 0.45 to stay inside with margin
    points *= 0.45
    
    return points


def _fcc_init():
    """Create an FCC-inspired packing for 50 spheres."""
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ])
    
    nx, ny, nz = 3, 3, 4
    points = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                for b in basis:
                    p = np.array([ix, iy, iz]) + b
                    p = (p - 1.25) / 2.5
                    points.append(p)
    
    points = np.array(points)
    
    selected = [0]
    remaining = list(range(1, len(points)))
    
    while len(selected) < N and remaining:
        best_idx = -1
        best_min_dist = -1
        for idx in remaining:
            min_d = np.min(np.linalg.norm(points[selected] - points[idx], axis=1))
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx
        
        if best_idx == -1:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    centres = points[selected]
    
    rng = np.random.RandomState(42)
    centres += rng.normal(0, 0.003, size=centres.shape)
    centres = np.clip(centres, -0.49, 0.49)
    
    return centres


def _random_init(seed):
    """Create a random initial packing."""
    rng = np.random.RandomState(seed)
    centres = rng.uniform(-0.45, 0.45, size=(N, 3))
    return centres


def _scipy_optimize(centres, time_limit=20.0):
    """
    Refine positions using scipy.optimize.minimize with BFGS.
    """
    start_time = time.time()
    
    # Flatten centres
    x0 = centres.flatten()
    
    def objective(x):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        c = x.reshape(N, 3)
        # Ensure valid bounds
        if np.any(c < -0.5) or np.any(c > 0.5):
            return 1e6
        return _neg_radius(c)
    
    def gradient(x):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        c = x.reshape(N, 3)
        return _neg_radius_grad(c).flatten()
    
    try:
        res = minimize(objective, x0, method='BFGS', jac=gradient,
                       options={'maxiter': 500, 'gtol': 1e-6})
        if res.success or res.fun < _neg_radius(centres):
            best_c = res.x.reshape(N, 3)
            # Clip to ensure validity
            best_c = np.clip(best_c, -0.5 + 1e-9, 0.5 - 1e-9)
            return best_c
    except Exception:
        pass
    
    return centres


def _force_directed(centres, time_limit=20.0):
    """
    Fast force-directed refinement.
    """
    start_time = time.time()
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    c = centres.copy()
    step = 0.002
    
    while time.time() - start_time < time_limit:
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dist, np.inf)
        
        threshold = 0.35
        mask = (dist < threshold) & (dist > 1e-10)
        
        inv_d3 = np.zeros_like(dist)
        inv_d3[mask] = 1.0 / (dist[mask] ** 3)
        
        forces = np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)
        
        for k in range(3):
            above = c[:, k] > 0.45
            below = c[:, k] < -0.45
            forces[above, k] -= (c[above, k] - 0.45) * 50.0
            forces[below, k] += (-0.45 - c[below, k]) * 50.0
        
        forces -= 0.005 * c
        
        force_mags = np.linalg.norm(forces, axis=1, keepdims=True)
        force_mags = np.maximum(force_mags, 1e-10)
        forces = forces / force_mags
        
        c += step * forces
        c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)
        
        current_radius = _compute_radius(c)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centres = c.copy()
    
    return best_centres, best_radius


def construct():
    """Construct centres of N equal spheres in unit cube."""
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_centres = None
    best_radius = 0.0
    
    # Strategy 1: Fibonacci lattice initialization + Scipy optimization
    if time.time() < deadline - 30:
        try:
            init = _fibonacci_lattice_init(N)
            # Quick force-directed refinement first
            fd_centres, _ = _force_directed(init, time_limit=min(10, deadline - time.time() - 10))
            # Then scipy polish
            opt_centres = _scipy_optimize(fd_centres, time_limit=min(20, deadline - time.time()))
            r = _compute_radius(opt_centres)
            if r > best_radius:
                best_radius = r
                best_centres = opt_centres
        except Exception:
            pass
    
    # Strategy 2: FCC initialization + Scipy optimization
    if time.time() < deadline - 30 and best_centres is not None:
        try:
            init = _fcc_init()
            fd_centres, _ = _force_directed(init, time_limit=min(10, deadline - time.time() - 10))
            opt_centres = _scipy_optimize(fd_centres, time_limit=min(20, deadline - time.time()))
            r = _compute_radius(opt_centres)
            if r > best_radius:
                best_radius = r
                best_centres = opt_centres
        except Exception:
            pass
    
    # Strategy 3: Random restarts with force-directed + short scipy
    if time.time() < deadline - 10:
        seeds = [1, 7, 13, 42, 99, 123, 456, 789, 1000, 2024]
        for seed in seeds:
            if time.time() > deadline - 5:
                break
            try:
                init = _random_init(seed)
                time_left = deadline - time.time() - 2
                if time_left < 5:
                    break
                fd_centres, _ = _force_directed(init, time_limit=min(5, time_left * 0.6))
                opt_centres = _scipy_optimize(fd_centres, time_limit=min(3, time_left * 0.4))
                r = _compute_radius(opt_centres)
                if r > best_radius:
                    best_radius = r
                    best_centres = opt_centres
            except Exception:
                continue
    
    # Final polish with force-directed
    if best_centres is not None and time.time() < deadline - 1:
        try:
            polished, pr = _force_directed(best_centres, time_limit=min(10, deadline - time.time() - 0.5))
            if pr > best_radius:
                best_radius = pr
                best_centres = polished
        except Exception:
            pass
    
    # Fallback
    if best_centres is None:
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centres = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    # Validation
    if not np.all(np.isfinite(best_centres)) or np.any(np.abs(best_centres) > 0.5):
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centres = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    return best_centres
# EVOLVE-BLOCK-END
