"""Combined approach: force-directed + full-pairwise soft-min optimization for 50 circles."""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

N = 50
DEADLINE = 145.0

def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    if min_wall < 0:
        return 0.0
    
    tree = cKDTree(centers)
    dists, idx = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_pair, min_wall)

def _force_directed_layout(centers, time_budget=30.0):
    """
    Force-directed layout using repulsive forces between all circles
    and attractive forces toward the square center.
    Uses full pairwise distances for better global exploration.
    """
    start_time = time.time()
    centers = centers.copy()
    n = len(centers)
    
    r_est = _compute_radius(centers)
    
    velocities = np.zeros_like(centers)
    damping = 0.85
    max_step = 0.03
    
    repulsion_strength = 1.0
    attraction_strength = 0.05
    center = np.array([0.0, 0.0])
    
    for it in range(50000):
        if time.time() - start_time > time_budget:
            break
        
        forces = np.zeros_like(centers)
        
        # Repulsive forces between all pairs
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    forces[i] += np.random.normal(size=2) * 0.01
                    forces[j] -= np.random.normal(size=2) * 0.01
                    continue
                
                # Strong repulsion when close, weak when far
                force_mag = repulsion_strength / (d * d)
                if d < 0.15:
                    force_mag *= 2.0  # Extra push when very close
                
                unit = diff / d
                forces[i] += force_mag * unit
                forces[j] -= force_mag * unit
        
        # Attractive force toward center (keeps circles in square)
        for i in range(n):
            to_center = center - centers[i]
            forces[i] += attraction_strength * to_center
        
        # Wall repulsion: push away from boundaries
        for i in range(n):
            x, y = centers[i]
            # Left wall
            if x < -0.4:
                forces[i, 0] += (0.4 + x) * 2.0
            # Right wall
            if x > 0.4:
                forces[i, 0] -= (x - 0.4) * 2.0
            # Bottom wall
            if y < -0.4:
                forces[i, 1] += (0.4 + y) * 2.0
            # Top wall
            if y > 0.4:
                forces[i, 1] -= (y - 0.4) * 2.0
        
        # Update velocities and positions
        velocities = damping * velocities + forces
        step = np.clip(velocities, -max_step, max_step)
        centers += step
        centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _soft_min_full(x, tau=0.008):
    """
    Soft-min objective using ALL pairwise distances and wall distances.
    x: flattened array of 2*N coordinates
    Returns negative soft-min (we minimize this, so maximize soft-min).
    """
    centers = x.reshape(-1, 2)
    n = N
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # All pairwise distances (upper triangle)
    iu, ju = np.triu_indices(n, k=1)
    diff = centers[iu] - centers[ju]
    pair_dists = np.sqrt(np.sum(diff**2, axis=1)) / 2.0
    
    all_dists = np.concatenate([walls, pair_dists])
    
    # Soft-min via log-sum-exp
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    log_sum = np.log(np.sum(np.exp(-shifted / tau)))
    soft_min = min_d - tau * log_sum
    
    return -soft_min

def _grad_soft_min_full(x, tau=0.008):
    """
    Analytical gradient of -soft_min using all pairwise distances.
    """
    centers = x.reshape(-1, 2)
    n = N
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # All pairwise distances
    iu, ju = np.triu_indices(n, k=1)
    diff = centers[iu] - centers[ju]
    pair_dists = np.sqrt(np.sum(diff**2, axis=1)) / 2.0
    
    all_dists = np.concatenate([walls, pair_dists])
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-shifted / tau)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(x)
    
    # 1. Wall contributions
    for i in range(n):
        w_wall = weights[i]
        if w_wall < 1e-10:
            continue
        if wall_dists_x[i] < wall_dists_y[i]:
            grad[2*i] += w_wall * np.sign(centers[i, 0])
        elif wall_dists_y[i] < wall_dists_x[i]:
            grad[2*i+1] += w_wall * np.sign(centers[i, 1])
        else:
            grad[2*i] += 0.5 * w_wall * np.sign(centers[i, 0])
            grad[2*i+1] += 0.5 * w_wall * np.sign(centers[i, 1])
    
    # 2. Pair contributions
    for k in range(len(iu)):
        w_pair = weights[n + k]
        if w_pair < 1e-10:
            continue
        
        i, j = iu[k], ju[k]
        d = np.linalg.norm(centers[i] - centers[j])
        if d < 1e-12:
            continue
        
        diff_ij = centers[i] - centers[j]
        factor = w_pair / (2.0 * d)
        
        grad[2*i:2*i+2] -= factor * diff_ij
        grad[2*j:2*j+2] += factor * diff_ij
    
    return grad

def _lbfgs_full(x0, tau=0.008, time_budget=20.0):
    """Run L-BFGS-B on the full-pairwise soft-min objective."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f = _soft_min_full(x, tau)
        g = _grad_soft_min_full(x, tau)
        return f, g
    
    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12,
                     'maxfun': 2000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _high_precision_polish(centers, time_budget):
    """
    High-precision polish using progressively smaller tau.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    for tau in [0.003, 0.0015, 0.0008, 0.0004]:
        if time.time() - start_time > time_budget:
            break
        
        x0 = best.flatten().copy()
        trial = _lbfgs_full(x0, tau=tau, time_budget=min(15, deadline - time.time() - 10))
        trial_r = _compute_radius(trial)
        if trial_r > best_r:
            best_r = trial_r
            best = trial
    
    return best, best_r

def _initial_hexagonal():
    """Create initial hexagonal-like layout."""
    # Hexagonal packing arrangement
    # For 50 circles, use a roughly hexagonal grid
    # Rows with alternating offsets
    centers = []
    
    # Try different grid configurations
    # 7x7 grid with some removed, or 8x7, etc.
    # Use a hexagonal pattern
    
    # Approximate hexagonal packing
    dx = 0.14  # horizontal spacing
    dy = 0.12  # vertical spacing (slightly less for hexagonal)
    
    # Start from a grid and add hexagonal offset
    m = 7  # 7x7 = 49, need 50
    ticks = (np.arange(m) + 0.5) / m - 0.5
    
    for row in range(m):
        for col in range(m):
            if len(centers) >= N:
                break
            x = ticks[col]
            y = ticks[row]
            # Hexagonal offset for odd rows
            if row % 2 == 1:
                x += 0.5 * dx * m
            centers.append([x, y])
    
    centers = np.array(centers[:N])
    
    # Add slight jitter
    rng = np.random.default_rng(42)
    centers = centers + rng.normal(0, 0.003, centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = np.random.default_rng(777)
    
    # Initial configuration: hexagonal layout
    best_centers = _initial_hexagonal()
    best_radius = _compute_radius(best_centers)
    
    # Phase 1: Force-directed layout for global exploration
    fd_budget = min(40.0, (deadline - time.time()) * 0.35)
    if fd_budget > 5:
        fd_centers = _force_directed_layout(best_centers, time_budget=fd_budget)
        fd_radius = _compute_radius(fd_centers)
        if fd_radius > best_radius:
            best_radius = fd_radius
            best_centers = fd_centers
    
    # Phase 2: L-BFGS with full-pairwise soft-min, multiple tau values
    if time.time() < deadline - 50:
        x0 = best_centers.flatten()
        for tau in [0.01, 0.008, 0.005, 0.003]:
            if time.time() > deadline - 35:
                break
            trial = _lbfgs_full(x0, tau=tau, time_budget=min(12, deadline - time.time() - 20))
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
            x0 = best_centers.flatten()
    
    # Phase 3: High-precision polish with smaller tau
    if time.time() < deadline - 25:
        best_centers, best_radius = _high_precision_polish(
            best_centers, time_budget=min(25, deadline - time.time() - 15)
        )
    
    # Phase 4: Final local refinement with very small tau
    if time.time() < deadline - 10:
        x0 = best_centers.flatten()
        trial = _lbfgs_full(x0, tau=0.0002, time_budget=min(10, deadline - time.time() - 5))
        trial_r = _compute_radius(trial)
        if trial_r > best_radius:
            best_radius = trial_r
            best_centers = trial
    
    # Final validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_hexagonal()
    
    return best_centers
