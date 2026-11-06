"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt


def _radius(centers):
    """Compute the maximum common radius for given centers (vectorized)."""
    n = len(centers)
    if n < 2:
        wall = 0.5 - np.max(np.abs(centers), axis=1)
        return np.min(wall)
    # Pairwise distances (upper triangle)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair_dist = np.sqrt(np.min(d2))
    # Distance to walls
    wall = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall)
    return min(min_pair_dist / 2.0, min_wall)


def _soft_radius(centers, beta):
    """
    Compute a smooth approximation of the radius using log-sum-exp.
    We want to maximize min(pairwise_dist/2, wall_dist).
    soft_min(d) = -(1/beta) * log(sum(exp(-beta * d_i)))
    We want to maximize soft_min(d).
    So we minimize -soft_min(d).
    """
    n = len(centers)
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    pair_dists = np.sqrt(d2) / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    
    # Combine all distances into one array
    # pair_dists is (n, n), wall_dists is (n, 3)
    # We need to flatten and concatenate
    pair_flat = pair_dists[np.triu_indices(n, k=1)]
    wall_flat = wall_dists.flatten()
    all_dists = np.concatenate([pair_flat, wall_flat])
    
    # soft min
    max_d = np.max(all_dists)
    log_sum = np.log(np.sum(np.exp(-beta * (all_dists - max_d))))
    soft_min_val = max_d - np.log(np.sum(np.exp(-beta * (all_dists - max_d)))) / beta
    
    return soft_min_val


def _grad_soft_radius(centers, beta):
    """
    Gradient of -soft_radius with respect to centers.
    soft_min(d) = max_d - (1/beta) * log(sum(exp(-beta * d_i)))
    d(soft_min)/d(centers) = -sum_i w_i * d(d_i)/d(centers)
    where w_i = exp(-beta * d_i) / sum_j exp(-beta * d_j)
    
    For pairwise distance d_ij = |c_i - c_j| / 2:
    d(d_ij)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|)
    d(d_ij)/d(c_j) = (c_j - c_i) / (2 * |c_i - c_j|)
    
    For wall distance d_ik = 0.5 - |c_ik|:
    d(d_ik)/d(c_ik) = -sign(c_ik)
    """
    n = len(centers)
    grad = np.zeros_like(centers)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    dists = np.sqrt(d2) / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    
    # Combine all distances
    pair_flat = dists[np.triu_indices(n, k=1)]
    wall_flat = wall_dists.flatten()
    all_dists = np.concatenate([pair_flat, wall_flat])
    
    # Compute weights
    max_d = np.max(all_dists)
    exp_vals = np.exp(-beta * (all_dists - max_d))
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    # Pairwise gradients
    pair_weights = weights[:len(pair_flat)]
    # Reconstruct pairwise weight matrix
    pair_w_matrix = np.zeros((n, n))
    iu = np.triu_indices(n, k=1)
    pair_w_matrix[iu] = pair_weights
    pair_w_matrix = pair_w_matrix + pair_w_matrix.T
    
    # d(d_ij)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|)
    # = diff[i,j] / (2 * |diff[i,j]|) = diff[i,j] / (2 * 2 * d_ij) since d_ij = |diff|/2
    # Actually: |diff[i,j]| = 2 * d_ij
    # d(d_ij)/d(c_i) = diff[i,j] / |diff[i,j]| = diff[i,j] / (2 * d_ij)
    
    # Compute gradient contribution from pairs
    # grad[i] += sum_j w_ij * (c_i - c_j) / (2 * d_ij)
    # But we need to be careful: d_ij = |c_i - c_j| / 2
    # d(d_ij)/d(c_i) = (c_i - c_j) / |c_i - c_j| = (c_i - c_j) / (2 * d_ij)
    
    # Vectorized computation
    # For each pair (i,j), contribution to grad[i] is w_ij * (c_i - c_j) / (2 * d_ij)
    # Contribution to grad[j] is w_ij * (c_j - c_i) / (2 * d_ij)
    
    # Let's compute this efficiently
    # diff[i,j] = c_i - c_j
    # d_ij = |diff[i,j]| / 2
    # unit vector = diff[i,j] / |diff[i,j]|
    # d(d_ij)/d(c_i) = unit_vector / 2 ... wait
    # d_ij = |c_i - c_j| / 2
    # d(d_ij)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|)
    
    # So grad[i] += sum_j w_ij * (c_i - c_j) / (2 * |c_i - c_j|)
    
    # |diff[i,j]| = 2 * d_ij
    # So (c_i - c_j) / (2 * |c_i - c_j|) = diff[i,j] / (2 * 2 * d_ij) = diff[i,j] / (4 * d_ij)
    
    # Let's just compute directly
    abs_diff = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(abs_diff, 1.0)  # Avoid division by zero
    
    # unit direction
    dir_vec = diff / abs_diff[:, :, np.newaxis]
    
    # grad[i] += sum_j w_ij * dir_vec[i,j] / 2
    # Because d(d_ij)/d(c_i) = dir_vec[i,j] / 2
    
    # Weighted sum
    weighted_dir = pair_w_matrix[:, :, np.newaxis] * dir_vec / 2.0
    pair_grad = np.sum(weighted_dir, axis=1)
    
    # Wall gradients
    wall_weights = weights[len(pair_flat):]
    wall_w_matrix = wall_weights.reshape(n, 3)
    
    # d(d_ik)/d(c_ik) = -sign(c_ik)
    signs = np.sign(centers)
    wall_grad = -wall_w_matrix * signs
    
    # Total gradient of soft_min
    soft_min_grad = pair_grad + wall_grad
    
    # We want to minimize -soft_min, so gradient of objective is -soft_min_grad
    return -soft_min_grad


def _bcc_init(n_points):
    """Generate initial points on a distorted BCC lattice."""
    points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ox = -0.5 + i * (1.0/3.0)
                oy = -0.5 + j * (1.0/3.0)
                oz = -0.5 + k * (1.0/3.0)
                
                points.append([ox, oy, oz])
                points.append([ox + 1.0/6.0, oy + 1.0/6.0, oz + 1.0/6.0])
                
    points = np.array(points)
    return points[:n_points].copy()


def _fcc_init(n_points):
    """Generate initial points on an FCC-like lattice."""
    points = []
    # FCC: corners + face centers
    # In a unit cell, FCC has 4 points: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    # Let's use a 3x3x3 grid of FCC cells
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ox = -0.5 + i * (1.0/3.0)
                oy = -0.5 + j * (1.0/3.0)
                oz = -0.5 + k * (1.0/3.0)
                cell = 1.0/3.0
                
                # 4 FCC points per cell
                points.append([ox, oy, oz])
                points.append([ox + cell/2, oy + cell/2, oz])
                points.append([ox + cell/2, oy, oz + cell/2])
                points.append([ox, oy + cell/2, oz + cell/2])
    
    points = np.array(points)
    return points[:n_points].copy()


def _random_init(n_points, seed=None):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n_points, 3))


def _optimize_smooth(centers, time_budget, max_beta=5000, initial_beta=100):
    """
    Optimize using smooth surrogate (soft-min) with gradient-based method.
    Gradually increase beta for sharper approximation.
    """
    start = time.time()
    centers = centers.copy()
    n = len(centers)
    
    best_r = _radius(centers)
    best_centers = centers.copy()
    
    beta = initial_beta
    while time.time() - start < time_budget:
        # L-BFGS-B on smooth objective
        def objective(x):
            c = x.reshape(n, 3)
            # Clip to keep in bounds
            c = np.clip(c, -0.5, 0.5)
            return -_soft_radius(c, beta)
        
        def gradient(x):
            c = x.reshape(n, 3)
            c = np.clip(c, -0.5, 0.5)
            return _grad_soft_radius(c, beta).flatten()
        
        x0 = centers.flatten()
        
        try:
            result = opt.minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                                  bounds=[(-0.5, 0.5)] * (n * 3),
                                  options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12})
            new_centers = np.clip(result.x.reshape(n, 3), -0.5, 0.5)
            new_r = _radius(new_centers)
            
            if new_r > best_r:
                best_r = new_r
                best_centers = new_centers.copy()
                centers = new_centers.copy()
        except Exception:
            pass
        
        # Increase beta
        beta = min(beta * 2, max_beta)
        
        # Check time
        if time.time() - start > time_budget - 5:
            break
    
    return best_centers, best_r


def _sa_optimize(centers, time_budget):
    """Simulated Annealing optimization."""
    start = time.time()
    centers = centers.copy()
    n = len(centers)
    current_r = _radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    T = 0.005
    T_min = 1e-5
    step_size = 0.01
    decay = 0.9995
    
    while time.time() - start < time_budget:
        i = np.random.randint(n)
        
        delta = np.random.normal(0, step_size, 3)
        new_center = centers[i].copy()
        new_center += delta
        new_center = np.clip(new_center, -0.5, 0.5)
        
        new_centers = centers.copy()
        new_centers[i] = new_center
        new_r = _radius(new_centers)
        
        if new_r >= current_r:
            centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        else:
            if np.random.rand() < np.exp((new_r - current_r) / T):
                centers = new_centers
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
        
        T *= decay
        step_size *= decay
        if T < T_min:
            T = T_min
            step_size = max(step_size, 0.001)
            
    return best_centers, best_r


def _coordinate_descent(centers, time_budget):
    """Coordinate Descent polish."""
    start = time.time()
    centers = centers.copy()
    n = len(centers)
    current_r = _radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    while time.time() - start < time_budget:
        improved = False
        for i in range(n):
            for d in range(3):
                val = centers[i, d]
                
                # Use golden-section search or fine grid
                candidates = np.linspace(-0.5, 0.5, 50)
                best_val = val
                best_r_local = current_r
                
                for c_val in candidates:
                    new_centers = centers.copy()
                    new_centers[i, d] = c_val
                    r = _radius(new_centers)
                    if r > best_r_local:
                        best_r_local = r
                        best_val = c_val
                
                if abs(best_val - val) > 1e-10:
                    centers[i, d] = best_val
                    current_r = best_r_local
                    improved = True
                    if current_r > best_r:
                        best_r = current_r
                        best_centers = centers.copy()
        
        if not improved:
            break
            
    return best_centers, best_r


def _high_precision_polish(centers, time_budget):
    """Final high-precision polish using L-BFGS-B on the exact (non-smooth) objective."""
    start = time.time()
    centers = centers.copy()
    n = len(centers)
    
    best_r = _radius(centers)
    best_centers = centers.copy()
    
    # Use L-BFGS-B with a penalty-based approach
    def objective(x):
        c = x.reshape(n, 3)
        c = np.clip(c, -0.5, 0.5)
        return -_radius(c)
    
    x0 = centers.flatten()
    
    try:
        result = opt.minimize(objective, x0, method='L-BFGS-B',
                              bounds=[(-0.5, 0.5)] * (n * 3),
                              options={'maxiter': 1000, 'gtol': 1e-10, 'ftol': 1e-15})
        new_centers = np.clip(result.x.reshape(n, 3), -0.5, 0.5)
        new_r = _radius(new_centers)
        
        if new_r > best_r:
            best_r = new_r
            best_centers = new_centers.copy()
    except Exception:
        pass
    
    # Final coordinate descent
    if time.time() - start < time_budget - 2:
        cd_centers, cd_r = _coordinate_descent(best_centers, time_budget - (time.time() - start) - 2)
        if cd_r > best_r:
            best_r = cd_r
            best_centers = cd_centers.copy()
    
    return best_centers, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Multiple initializations
    np.random.seed(42)
    
    inits = []
    # BCC
    inits.append(_bcc_init(N))
    # FCC
    inits.append(_fcc_init(N))
    # Random with different seeds
    for s in range(5):
        inits.append(_random_init(N, seed=s))
    
    # Sort by initial radius
    init_radii = [_radius(c) for c in inits]
    sorted_indices = np.argsort(init_radii)[::-1]
    inits = [inits[i] for i in sorted_indices]
    
    best_centers = inits[0].copy()
    best_r = init_radii[sorted_indices[0]]
    
    # Phase 1: Smooth optimization on top initializations
    for idx in range(min(4, len(inits))):
        if time.time() - start_time > deadline - 60:
            break
        budget = min(15.0, deadline - 60 - (time.time() - start_time))
        if budget < 5:
            break
        
        smooth_centers, smooth_r = _optimize_smooth(inits[idx], budget, max_beta=5000, initial_beta=100)
        if smooth_r > best_r:
            best_r = smooth_r
            best_centers = smooth_centers.copy()
    
    # Phase 2: Simulated Annealing on best
    if time.time() - start_time < deadline - 30:
        budget = min(20.0, deadline - 30 - (time.time() - start_time))
        if budget > 5:
            sa_centers, sa_r = _sa_optimize(best_centers, budget)
            if sa_r > best_r:
                best_r = sa_r
                best_centers = sa_centers.copy()
    
    # Phase 3: High-precision polish
    if time.time() - start_time < deadline - 10:
        budget = min(30.0, deadline - 10 - (time.time() - start_time))
        if budget > 5:
            polish_centers, polish_r = _high_precision_polish(best_centers, budget)
            if polish_r > best_r:
                best_r = polish_r
                best_centers = polish_centers.copy()
    
    # Final validity check
    if not np.all(np.isfinite(best_centers)) or np.any(np.abs(best_centers) > 0.5):
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].copy()
    
    return best_centers
# EVOLVE-BLOCK-END
