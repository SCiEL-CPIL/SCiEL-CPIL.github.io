"""Optimized packing of 50 equal circles in a unit circle using L-BFGS-B."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50


def _score(positions):
    """Compute the packing radius."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    radii = np.hypot(positions[:, 0], positions[:, 1])
    min_boundary = 1.0 - np.max(radii)
    return min(min_pair, min_boundary)


def _soft_min(values, beta):
    """Soft minimum approximation."""
    log_sum = np.log(np.sum(np.exp(-beta * values))) / beta
    return -log_sum


def _objective_and_gradient(x, beta=100.0):
    """
    Objective: maximize the soft-min of (pairwise_dists/2, boundary_dists).
    We minimize the negative.
    """
    n = N
    positions = x.reshape(n, 2)
    
    # Compute pairwise distances (n x n)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Distance to boundary for each point
    radii = np.hypot(positions[:, 0], positions[:, 1])
    boundary_dists = 1.0 - radii
    
    # Collect all constraint values: pairwise/2 (upper triangle) and boundary
    # We want to maximize the minimum of these
    # Pairwise: dists[i,j]/2 for i<j
    # Boundary: boundary_dists[i]
    
    # Build array of all values
    # For efficiency, compute soft-min separately and combine
    # Actually, let's just compute the soft-min of all relevant values
    
    # Get upper triangle indices
    iu = np.triu_indices(n, k=1)
    pair_vals = dists[iu] / 2.0
    all_vals = np.concatenate([pair_vals, boundary_dists])
    
    # Soft minimum
    soft_min_val = _soft_min(all_vals, beta)
    obj = -soft_min_val
    
    # Gradient computation
    # d(soft_min)/dx = -sum(exp(-beta*v_i) / sum(exp(-beta*v_j))) * dv_i/dx
    # So d(-soft_min)/dx = sum(exp(-beta*v_i) / sum(exp(-beta*v_j))) * dv_i/dx
    
    exp_vals = np.exp(-beta * all_vals)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(x)
    
    # Gradient of pairwise distances
    # For each pair (i,j), v = dists[i,j]/2
    # d(dist)/dx_i = (x_i - x_j) / dist
    # d(dist)/dx_j = (x_j - x_i) / dist
    
    # Vectorized: for each pair, accumulate gradient
    pair_dists = dists[iu]
    # Avoid division by zero
    safe_dists = np.where(pair_dists < 1e-10, 1e-10, pair_dists)
    
    # weights for pairs: first len(iu[0]) elements of weights
    pair_weights = weights[:len(iu[0])]
    
    # For each pair (i,j), contribution to grad[i] = pair_weights[k] * (x_i - x_j) / dist
    # contribution to grad[j] = -pair_weights[k] * (x_i - x_j) / dist
    
    i_idx = iu[0]
    j_idx = iu[1]
    
    # diff[i,j] = x_i - x_j
    pair_diffs = positions[i_idx] - positions[j_idx]  # (num_pairs, 2)
    pair_grad_mags = pair_weights / safe_dists  # (num_pairs,)
    
    # Accumulate
    pair_grads = pair_diffs * pair_grad_mags[:, np.newaxis]
    
    # Use np.add.at for accumulation
    np.add.at(grad, i_idx, pair_grads)
    np.add.at(grad, j_idx, -pair_grads)
    
    # Gradient of boundary distances
    # v_i = 1 - |x_i|, dv_i/dx_i = -x_i / |x_i|
    boundary_weights = weights[len(iu[0]):]
    safe_radii = np.where(radii < 1e-10, 1e-10, radii)
    boundary_grads = -positions / safe_radii[:, np.newaxis] * boundary_weights[:, np.newaxis]
    
    grad += boundary_grads
    
    return obj, grad


def _get_hex_lattice_init(n):
    """Generate initial positions on a hexagonal lattice."""
    spacing = 0.2
    points = []
    k = 12
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r < 0.95:
                points.append((x, y))
    
    points = np.array(points)
    dists = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(dists, kind='stable')[:n]
    return points[idx].copy()


def _get_ring_init(n):
    """Generate initial positions in concentric rings."""
    positions = []
    # Center
    positions.append([0.0, 0.0])
    
    # Ring 1: 6 points
    r1 = 0.25
    for k in range(6):
        angle = k * np.pi / 3
        positions.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    # Ring 2: 12 points
    r2 = 0.5
    for k in range(12):
        angle = k * np.pi / 6
        positions.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    
    # Ring 3: 18 points
    r3 = 0.75
    for k in range(18):
        angle = k * 2 * np.pi / 18
        positions.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    
    # Ring 4: 15 points
    r4 = 0.9
    for k in range(15):
        angle = k * 2 * np.pi / 15
        positions.append([r4 * np.cos(angle), r4 * np.sin(angle)])
    
    positions = np.array(positions[:n])
    return positions


def _get_random_init(n, seed=42):
    """Generate random initial positions inside the unit circle."""
    rng = np.random.RandomState(seed)
    positions = []
    for _ in range(n):
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            if x*x + y*y <= 1.0:
                positions.append([x, y])
                break
    return np.array(positions)


def _optimize_lbfgs(positions, max_time=60.0, beta_start=50.0, beta_end=500.0):
    """Optimize using L-BFGS-B with increasing beta."""
    start_time = time.time()
    n = len(positions)
    x0 = positions.flatten().copy()
    
    best_r = _score(positions)
    best_x = x0.copy()
    
    # Progressive beta schedule
    betas = [50, 100, 200, 500, 1000, 2000]
    
    for beta in betas:
        if time.time() - start_time > max_time:
            break
        
        def obj_grad(x):
            return _objective_and_gradient(x, beta=beta)
        
        result = minimize(
            obj_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
        )
        
        x_new = result.x
        r_new = _score(x_new.reshape(n, 2))
        
        if r_new > best_r:
            best_r = r_new
            best_x = x_new.copy()
        
        x0 = x_new
    
    return best_x.reshape(n, 2), best_r


def _polish(positions, max_time=30.0):
    """Final polish with high beta for precision."""
    start_time = time.time()
    n = len(positions)
    x0 = positions.flatten().copy()
    best_r = _score(positions)
    best_x = x0.copy()
    
    betas = [1000, 2000, 5000, 10000]
    
    for beta in betas:
        if time.time() - start_time > max_time:
            break
        
        def obj_grad(x):
            return _objective_and_gradient(x, beta=beta)
        
        result = minimize(
            obj_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 500}
        )
        
        x_new = result.x
        r_new = _score(x_new.reshape(n, 2))
        
        if r_new > best_r:
            best_r = r_new
            best_x = x_new.copy()
        
        x0 = x_new
    
    return best_x.reshape(n, 2), best_r


def construct():
    """Construct positions for N circles in the unit disk."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Try multiple initializations and pick the best
    inits = [
        _get_hex_lattice_init(N),
        _get_ring_init(N),
        _get_random_init(N, seed=42),
        _get_random_init(N, seed=123),
    ]
    
    best_positions = None
    best_r = 0.0
    
    for init_pos in inits:
        if time.time() > deadline - 30.0:
            break
        
        # Optimize
        positions, r = _optimize_lbfgs(init_pos, max_time=25.0)
        
        if r > best_r:
            best_r = r
            best_positions = positions
    
    if best_positions is None:
        # Fallback
        best_positions = _get_hex_lattice_init(N)
        best_r = _score(best_positions)
    
    # Polish if time permits
    if time.time() < deadline - 10.0:
        best_positions, best_r = _polish(best_positions, max_time=25.0)
    
    # Final validation: ensure all circles are inside
    radii = np.hypot(best_positions[:, 0], best_positions[:, 1])
    current_r = _score(best_positions)
    max_allowed = 1.0 - current_r
    mask = radii > max_allowed
    if np.any(mask):
        best_positions[mask] *= max_allowed / radii[mask]
    
    return best_positions
