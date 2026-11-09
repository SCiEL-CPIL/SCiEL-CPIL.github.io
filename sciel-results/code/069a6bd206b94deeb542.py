"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    # Pairwise distances using vectorized computation
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Distance to walls
    wall_dist = 0.5 - np.abs(centres)
    min_wall_dist = np.min(wall_dist)
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def _neg_soft_min_radius(centres, beta=50.0):
    """
    Negative of the soft-min radius (for minimization).
    Uses log-sum-exp trick for numerical stability.
    """
    n = len(centres)
    
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    
    # Flatten pairwise distances (upper triangle only to avoid double counting)
    mask = ~np.eye(n, dtype=bool)
    pair_dists = dist[mask] / 2.0
    
    # Wall distances
    wall_dist = 0.5 - np.abs(centres)
    wall_dists = wall_dist.flatten()
    
    # Combine all constraints
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Soft min: -1/beta * log(sum(exp(-beta * x)))
    # For numerical stability, subtract max
    x_max = np.max(all_dists)
    log_sum = np.log(np.sum(np.exp(-beta * (all_dists - x_max))))
    soft_min_val = x_max - log_sum / beta
    
    return -soft_min_val


def _neg_soft_min_radius_grad(centres, beta=50.0):
    """
    Gradient of the negative soft-min radius.
    """
    n = len(centres)
    
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    
    # Weights for soft min: w_i = exp(-beta * x_i) / sum(exp(-beta * x_j))
    mask = ~np.eye(n, dtype=bool)
    pair_dists = dist[mask] / 2.0
    
    wall_dist = 0.5 - np.abs(centres)
    wall_dists = wall_dist.flatten()
    
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Compute weights
    x_max = np.max(all_dists)
    exp_vals = np.exp(-beta * (all_dists - x_max))
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient of soft_min w.r.t. each constraint is -weights
    # So gradient of -soft_min is +weights
    
    # Build gradient for centres
    grad = np.zeros_like(centres)
    
    # Pairwise part
    # For each pair (i,j), the contribution to grad[i] is weights_pair * (c_i - c_j) / dist_ij * beta
    # But we need to be careful with the indexing
    
    # Let's reconstruct the pair distances matrix
    # pair_dists has length n*(n-1), corresponding to all (i,j) with i!=j
    
    # Actually, let's just compute it directly
    pair_weights = weights[:n*(n-1)]
    
    # Reshape to matrix
    pw_matrix = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                pw_matrix[i, j] = pair_weights[idx]
                idx += 1
    
    # Gradient from pairwise terms
    # d(soft_min)/dc_i = -sum_j pw_matrix[i,j] * (c_i - c_j) / dist_ij * beta
    # So d(-soft_min)/dc_i = sum_j pw_matrix[i,j] * (c_i - c_j) / dist_ij * beta
    
    for i in range(n):
        for j in range(n):
            if i != j and dist[i, j] > 1e-10:
                grad[i] += pw_matrix[i, j] * (centres[i] - centres[j]) / dist[i, j] * beta
    
    # Wall part
    wall_weights = weights[n*(n-1):]
    
    for k in range(3):
        for i in range(n):
            # wall_dist[i,k] = 0.5 - |c_i,k|
            # d(soft_min)/dc_i,k = -wall_weights[i*3+k] * d(0.5-|c_i,k|)/dc_i,k
            # = -wall_weights[i*3+k] * (-sign(c_i,k))
            # = wall_weights[i*3+k] * sign(c_i,k)
            # So d(-soft_min)/dc_i,k = -wall_weights[i*3+k] * sign(c_i,k)
            
            if centres[i, k] > 1e-10:
                grad[i, k] -= wall_weights[i*3+k] * beta
            elif centres[i, k] < -1e-10:
                grad[i, k] += wall_weights[i*3+k] * beta
    
    return grad


def _make_fibonacci_lattice(n_points):
    """
    Generate points on a Fibonacci sphere/lattice for good initial separation.
    We'll use a 3D generalization: distribute points in a cube using a
    space-filling curve approach or golden ratio spiral in 3D.
    """
    # Use a simple approach: generate points using golden ratio in 3D
    # This gives reasonably uniform distribution
    
    golden_ratio = (1 + np.sqrt(5)) / 2
    inv_phi = 1.0 / golden_ratio
    
    points = []
    for i in range(n_points):
        # Use a 3D spiral based on golden ratio
        t = i / n_points
        
        # Map to [-0.5, 0.5]^3 using a space-filling pattern
        x = (t * inv_phi) % 1.0 - 0.5
        y = (t * inv_phi * inv_phi) % 1.0 - 0.5
        z = (t * inv_phi * inv_phi * inv_phi) % 1.0 - 0.5
        
        points.append([x, y, z])
    
    return np.array(points)


def _make_fibonacci_packing():
    """
    Create a Fibonacci-based initial packing with good separation.
    """
    # Generate more points than needed, then select best N
    n_candidates = N * 3
    points = _make_fibonacci_lattice(n_candidates)
    
    # Scale to fit in cube with some margin
    points = points * 0.9  # Keep away from walls initially
    
    # Select N points with maximum minimum distance using greedy
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
    
    # Add small random perturbation to break symmetry
    rng = np.random.RandomState(42)
    centres += rng.normal(0, 0.002, size=centres.shape)
    centres = np.clip(centres, -0.48, 0.48)
    
    return centres


def _optimize_with_lbfgsb(centres, time_limit=30.0, beta=50.0):
    """
    Optimize using L-BFGS-B with smooth surrogate objective.
    """
    start_time = time.time()
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    def objective(x):
        c = x.reshape(-1, 3)
        return _neg_soft_min_radius(c, beta)
    
    def gradient(x):
        c = x.reshape(-1, 3)
        g = _neg_soft_min_radius_grad(c, beta)
        return g.flatten()
    
    # Flatten centres
    x0 = centres.flatten()
    
    # Bounds: keep within cube
    bounds = [(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3)
    
    # Time-limited optimization
    elapsed = time.time() - start_time
    if elapsed < time_limit:
        try:
            result = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
            )
            
            if result.success or result.fun < objective(x0):
                new_centres = result.x.reshape(-1, 3)
                new_radius = _compute_radius(new_centres)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centres = new_centres
        except Exception:
            pass
    
    return best_centres, best_radius


def _make_fcc_packing():
    """
    Create an FCC-inspired packing for 50 spheres.
    """
    # FCC basis
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ])
    
    # Generate FCC lattice points in a 3x3x4 supercell
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
    
    # Select 50 with maximum minimum distance using greedy algorithm
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
    
    # Add small random perturbation to break symmetry
    rng = np.random.RandomState(42)
    centres += rng.normal(0, 0.003, size=centres.shape)
    centres = np.clip(centres, -0.49, 0.49)
    
    return centres


def _make_random_packing(seed):
    """Create a random initial packing."""
    rng = np.random.RandomState(seed)
    centres = rng.uniform(-0.45, 0.45, size=(N, 3))
    return centres


def _make_grid_packing():
    """Create a grid-based initial packing."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    selected = [0]
    remaining = list(range(1, len(grid)))
    
    while len(selected) < N and remaining:
        best_idx = -1
        best_min_dist = -1
        for idx in remaining:
            min_d = np.min(np.linalg.norm(grid[selected] - grid[idx], axis=1))
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx
        
        if best_idx == -1:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    centres = grid[selected]
    
    rng = np.random.RandomState(123)
    centres += rng.normal(0, 0.002, size=centres.shape)
    centres = np.clip(centres, -0.49, 0.49)
    
    return centres


def _polish(centres, time_limit=10.0):
    """
    High-precision polish pass with very small steps and tighter convergence.
    """
    start_time = time.time()
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    n = len(centres)
    c = centres.copy()
    
    step = 0.0002
    min_step = 1e-12
    patience = 0
    max_patience = 1000
    
    while time.time() - start_time < time_limit:
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        dist = np.sqrt(dist_sq)
        np.fill_diagonal(dist, np.inf)
        
        threshold = 0.15
        mask = (dist < threshold) & (dist > 1e-10)
        
        # Use inverse square for stronger repulsion at close range
        inv_d2 = np.zeros_like(dist)
        inv_d2[mask] = 1.0 / (dist[mask] ** 2)
        
        forces = np.sum(diff * inv_d2[:, :, np.newaxis] / dist[:, :, np.newaxis], axis=1)
        # Simplify: diff / d^3
        forces = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1) / 1.0  # This is wrong, let me fix
        
        # Correct: force = diff / d^3
        inv_d3 = np.zeros_like(dist)
        inv_d3[mask] = 1.0 / (dist[mask] ** 3)
        forces = np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)
        
        for k in range(3):
            above = c[:, k] > 0.49
            below = c[:, k] < -0.49
            forces[above, k] -= (c[above, k] - 0.49) * 200.0
            forces[below, k] += (-0.49 - c[below, k]) * 200.0
        
        forces -= 0.0005 * c
        
        force_mags = np.linalg.norm(forces, axis=1, keepdims=True)
        force_mags = np.maximum(force_mags, 1e-10)
        forces = forces / force_mags
        
        c += step * forces
        c = np.clip(c, -0.5 + 1e-10, 0.5 - 1e-10)
        
        current_radius = _compute_radius(c)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centres = c.copy()
            patience = 0
        else:
            patience += 1
            step = max(step * 0.97, min_step)
        
        if patience > max_patience:
            break
    
    return best_centres, best_radius


def construct():
    """Construct centres of N equal spheres in unit cube."""
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for safety
    
    best_centres = None
    best_radius = 0.0
    
    # Strategy 1: Fibonacci lattice initialization + L-BFGS-B optimization
    if time.time() < deadline - 30:
        try:
            init = _make_fibonacci_packing()
            centres, radius = _optimize_with_lbfgsb(init, time_limit=min(40, deadline - time.time()))
            if radius > best_radius:
                best_radius = radius
                best_centres = centres
        except Exception:
            pass
    
    # Strategy 2: FCC-inspired packing + L-BFGS-B
    if time.time() < deadline - 25:
        try:
            init = _make_fcc_packing()
            centres, radius = _optimize_with_lbfgsb(init, time_limit=min(30, deadline - time.time()))
            if radius > best_radius:
                best_radius = radius
                best_centres = centres
        except Exception:
            pass
    
    # Strategy 3: Grid-based packing + L-BFGS-B
    if time.time() < deadline - 20:
        try:
            init = _make_grid_packing()
            centres, radius = _optimize_with_lbfgsb(init, time_limit=min(25, deadline - time.time()))
            if radius > best_radius:
                best_radius = radius
                best_centres = centres
        except Exception:
            pass
    
    # Strategy 4: Multiple random restarts with L-BFGS-B
    if time.time() < deadline - 15:
        for seed in [1, 7, 13, 42, 99, 123, 456, 789, 1000, 2024]:
            if time.time() > deadline - 8:
                break
            try:
                init = _make_random_packing(seed)
                time_left = deadline - time.time() - 3
                if time_left < 5:
                    break
                centres, radius = _optimize_with_lbfgsb(init, time_limit=min(8, time_left))
                if radius > best_radius:
                    best_radius = radius
                    best_centres = centres
            except Exception:
                continue
    
    # Final polish pass - use remaining time
    if best_centres is not None and time.time() < deadline - 1:
        try:
            polish_time = min(10, deadline - time.time() - 0.5)
            polished, pr = _polish(best_centres, time_limit=polish_time)
            if pr > best_radius:
                best_radius = pr
                best_centres = polished
        except Exception:
            pass
    
    # If no valid result, fallback to simple grid
    if best_centres is None:
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centres = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    # Final validation
    if not np.all(np.isfinite(best_centres)) or np.any(np.abs(best_centres) > 0.5):
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centres = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    return best_centres
# EVOLVE-BLOCK-END
