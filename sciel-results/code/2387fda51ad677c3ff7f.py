"""Pack 100 equal spheres in a unit cube [-0.5, 0.5]^3 to maximize minimum radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall constraint
    wall_dist = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_dist)
    if r_wall < 0:
        return 0.0
    # Pairwise distances
    # Use efficient computation: for each pair, compute distance
    # For N=100, we can use broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    return min(r_wall, r_pair)

def _energy_and_grad(centers):
    """
    Smooth surrogate: minimize a soft-min of all constraints.
    We want to maximize r = min(min_pair_dist/2, min_wall_dist).
    Equivalent to minimizing -r, but that's nonsmooth.
    
    Use: E = -softmin(logsumexp of -log(constraints))
    Or simpler: use a penalty-based approach.
    
    Actually, let's use a direct approach:
    Define f(x) = min over all constraints of (constraint value)
    We want to maximize f.
    
    For smooth optimization, use:
    E(x) = -logsumexp(beta * log(constraints)) / beta  (as beta -> inf, this -> min)
    
    But constraints can be zero or negative, so log is problematic.
    
    Alternative: use a soft-min via:
    softmin(a_1, ..., a_m) = (1/beta) * log(sum exp(-beta * a_i))
    Then we minimize softmin of all constraint violations... 
    
    Let me think differently. Define:
    d_ij = |c_i - c_j| for i < j
    w_i = 0.5 - max_k |c_ik|
    
    We want to maximize min(min_ij d_ij / 2, min_i w_i)
    
    Let's define the objective as:
    E = -softmin(all pairwise d_ij / 2, all wall distances w_i)
    
    Using softmin: softmin(x_1,...,x_m) = -(1/beta) * log(sum exp(-beta * x_i))
    As beta -> large, softmin -> min.
    
    So we minimize E = (1/beta) * log(sum exp(-beta * x_i)) where x_i are all the "radius" values.
    
    Actually, let's just use a simpler approach: 
    Minimize a sum of squared violations + a large penalty for the minimum.
    
    Better: Use the fact that we can optimize with L-BFGS-B by defining:
    objective = -softmin of all constraints
    
    Let me use beta that increases over iterations.
    """
    centers = centers.reshape(N, 3)
    
    # Compute all pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Get all unique pairs (i < j)
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0  # radius from pair constraint
    
    # Wall distances
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    # All constraint values (radii)
    all_r = np.concatenate([pair_dists, wall_dists])
    
    # Soft-min: we want to maximize the minimum, so minimize -softmin
    # softmin(x) = -(1/beta) * log(sum(exp(-beta * x)))
    # For numerical stability, use logsumexp
    beta = 50.0  # temperature parameter
    
    # To avoid overflow, shift by max
    x = -beta * all_r
    x_max = np.max(x)
    if x_max > 700:
        # Overflow protection
        log_sum = x_max + np.log(np.sum(np.exp(x - x_max)))
    else:
        log_sum = np.log(np.sum(np.exp(x)))
    
    softmin_val = -log_sum / beta
    return softmin_val  # We minimize this, which maximizes softmin(all_r)

def _grad_energy(centers, beta=50.0):
    """Gradient of the soft-min objective."""
    centers = centers.reshape(N, 3)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    all_r = np.concatenate([pair_dists, wall_dists])
    
    # Soft-min gradient
    # softmin(x) = -(1/beta) * log(sum(exp(-beta * x)))
    # d(softmin)/dx_i = exp(-beta * x_i) / sum(exp(-beta * x_j))
    
    x = -beta * all_r
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    sum_exp = np.sum(exp_x)
    weights = exp_x / sum_exp  # These are the softmax weights
    
    # Gradient with respect to centers
    grad = np.zeros_like(centers)
    
    n_pairs = len(i_idx)
    
    # Pairwise contributions
    for k in range(n_pairs):
        i, j = i_idx[k], j_idx[k]
        w = weights[k]
        # d(pair_dist_k)/d(centers[i]) = -(centers[i] - centers[j]) / |centers[i] - centers[j]|
        # pair_dist_k = |c_i - c_j| / 2
        # d(pair_dist_k)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|)
        # d(pair_dist_k)/d(c_j) = (c_j - c_i) / (2 * |c_i - c_j|)
        
        d = centers[i] - centers[j]
        dist = np.sqrt(np.sum(d * d))
        if dist < 1e-12:
            continue
        grad[i] -= w * d / (2.0 * dist)
        grad[j] += w * d / (2.0 * dist)
    
    # Wall contributions
    for i in range(N):
        w = weights[n_pairs + i]
        # wall_dist_i = 0.5 - max_k |c_ik|
        # d(wall_dist_i)/d(c_ik) = -sign(c_ik) if k is argmax
        # Find which coordinate has max absolute value
        abs_c = np.abs(centers[i])
        k_max = np.argmax(abs_c)
        grad[i, k_max] += w * np.sign(centers[i, k_max])
        # Note: d(0.5 - max|c|)/dc = -sign(c_k) for k=argmax
        # So d(softmin)/dc = weight * (-sign(c_k)) * ... 
        # Wait, let me redo this.
        # softmin = -(1/beta) log sum exp(-beta * r_m)
        # d(softmin)/dr_m = exp(-beta*r_m) / sum(exp(-beta*r_n)) = weights[m]
        # d(softmin)/dc_i = sum_m weights[m] * dr_m/dc_i
        # For wall: r_m = 0.5 - max_k |c_mk|
        # dr_m/dc_mk = -sign(c_mk) if k = argmax
        # So d(softmin)/dc_mk = weights[m] * (-sign(c_mk))
        # We're minimizing softmin, so gradient of objective is this.
        # grad[i, k_max] += weights[n_pairs+i] * (-sign(centers[i, k_max]))
    
    # Redo wall gradient correctly
    for i in range(N):
        w = weights[n_pairs + i]
        abs_c = np.abs(centers[i])
        k_max = np.argmax(abs_c)
        # dr/dc = -sign(c_k) for k=argmax
        grad[i, k_max] += w * (-np.sign(centers[i, k_max]))
    
    return grad

def _local_optimize(centers, max_time=10.0, beta=50.0):
    """Run L-BFGS-B local optimization from given centers."""
    start_time = time.time()
    
    def objective(x):
        return _energy_and_grad(x.reshape(N, 3))
    
    def grad(x):
        return _grad_energy(x.reshape(N, 3), beta).flatten()
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    result = minimize(
        objective,
        centers.flatten(),
        jac=grad,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    elapsed = time.time() - start_time
    best_centers = result.x.reshape(N, 3)
    best_radius = _compute_radius(best_centers)
    
    return best_centers, best_radius, elapsed

def _generate_initial_points(method='grid', rng=None):
    """Generate initial center positions."""
    if rng is None:
        rng = np.random.default_rng()
    
    if method == 'grid':
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()
    elif method == 'random':
        return rng.uniform(-0.45, 0.45, size=(N, 3))
    elif method == 'bcc':
        # Body-centered cubic arrangement
        # Generate a BCC lattice and select N points
        # BCC has 2 atoms per unit cell
        # For 100 spheres, we need about 50 cells
        # Let's use a 5x5x4 grid of BCC cells
        points = []
        for ix in range(5):
            for iy in range(5):
                for iz in range(4):
                    # Corner
                    x = (ix + 0.5) / 5.0 - 0.5
                    y = (iy + 0.5) / 5.0 - 0.5
                    z = (iz + 0.5) / 4.0 - 0.5
                    points.append([x, y, z])
                    # Center
                    xc = (ix + 1.0) / 5.0 - 0.5
                    yc = (iy + 1.0) / 5.0 - 0.5
                    zc = (iz + 1.0) / 4.0 - 0.5
                    points.append([xc, yc, zc])
        points = np.array(points[:N])
        return points
    else:
        return rng.uniform(-0.45, 0.45, size=(N, 3))

def construct():
    """Construct centers of 100 equal spheres in [-0.5, 0.5]^3 to maximize radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_centers = None
    best_radius = 0.0
    
    # Try multiple initializations and local optimizations
    rng = np.random.default_rng(42)
    
    # First, try the simple grid as a baseline
    grid_centers = _generate_initial_points('grid', rng)
    centers, radius, elapsed = _local_optimize(grid_centers, max_time=15.0)
    if radius > best_radius:
        best_radius = radius
        best_centers = centers.copy()
    
    # Try BCC initialization
    if time.time() < deadline - 20.0:
        bcc_centers = _generate_initial_points('bcc', rng)
        centers, radius, elapsed = _local_optimize(bcc_centers, max_time=15.0)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    # Try random initializations
    n_random_tries = 5
    for i in range(n_random_tries):
        if time.time() > deadline - 20.0:
            break
        
        # Add some perturbation to grid
        base = _generate_initial_points('grid', rng)
        noise_scale = 0.05 + 0.02 * i
        random_centers = base + rng.normal(0, noise_scale, size=(N, 3))
        # Clip to stay in bounds
        random_centers = np.clip(random_centers, -0.48, 0.48)
        
        centers, radius, elapsed = _local_optimize(random_centers, max_time=15.0)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    # Final polish: run one more optimization with higher beta for precision
    if time.time() < deadline - 10.0 and best_centers is not None:
        def objective_high_beta(x):
            return _energy_and_grad(x.reshape(N, 3))
        
        def grad_high_beta(x):
            return _grad_energy(x.reshape(N, 3), beta=100.0).flatten()
        
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        result = minimize(
            objective_high_beta,
            best_centers.flatten(),
            jac=grad_high_beta,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'maxfun': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        polished_centers = result.x.reshape(N, 3)
        polished_radius = _compute_radius(polished_centers)
        if polished_radius > best_radius:
            best_radius = polished_radius
            best_centers = polished_centers.copy()
    
    # Ensure validity: clip to bounds and verify
    if best_centers is not None:
        best_centers = np.clip(best_centers, -0.5, 0.5)
        final_radius = _compute_radius(best_centers)
        if final_radius > 0:
            return best_centers
    
    # Fallback to grid if something went wrong
    return _generate_initial_points('grid', rng)
