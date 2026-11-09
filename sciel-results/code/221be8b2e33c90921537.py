"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree


def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Distance to walls
    min_dist_to_wall = np.min(0.5 - np.abs(centers), axis=(0, 1))
    
    # Minimum pairwise distance / 2
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=1.0)  # query pairs within distance 1
    if len(pairs) == 0:
        # Fallback: compute all pairs
        min_pair_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                min_pair_dist = min(min_pair_dist, d)
        min_pair_dist = min_pair_dist
    else:
        min_pair_dist = np.inf
        for i, j in pairs:
            d = np.linalg.norm(centers[i] - centers[j])
            min_pair_dist = min(min_pair_dist, d)
    
    return min(min_dist_to_wall, min_pair_dist / 2.0)


def _soft_min_radius(centers, beta=50.0):
    """Compute a smooth approximation of the minimum radius using log-sum-exp."""
    centers = np.asarray(centers, dtype=np.float64)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)  # (N, 3)
    
    # Pairwise distances / 2
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=1.0)
    
    if len(pairs) == 0:
        pair_dists = np.array([np.inf])
    else:
        pair_dists = np.array([np.linalg.norm(centers[i] - centers[j]) / 2.0 for i, j in pairs])
    
    # Combine all constraints: wall distances and pair distances
    all_dists = np.concatenate([wall_dists.flatten(), pair_dists])
    all_dists = all_dists[all_dists > 1e-15]  # filter out non-positive
    
    if len(all_dists) == 0:
        return 0.0
    
    # Soft min via log-sum-exp: -log(sum(exp(-x/beta))) * beta
    # This approximates min(x) as beta -> 0
    scaled = -all_dists / beta
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum * beta
    
    return soft_min


def _compute_gradients(centers, beta=50.0):
    """Compute gradients of the soft-min radius with respect to center positions."""
    centers = np.asarray(centers, dtype=np.float64)
    grads = np.zeros_like(centers)
    
    # Wall distances: d_i = 0.5 - |c_i|
    wall_dists = 0.5 - np.abs(centers)  # (N, 3)
    
    # Pairwise distances / 2
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=1.0)
    
    if len(pairs) == 0:
        pair_dists = np.array([np.inf])
        pair_indices = []
    else:
        pair_dists = np.array([np.linalg.norm(centers[i] - centers[j]) / 2.0 for i, j in pairs])
        pair_indices = list(pairs)
    
    # Combine all constraints
    all_dists = np.concatenate([wall_dists.flatten(), pair_dists])
    valid_mask = all_dists > 1e-15
    all_dists_valid = all_dists[valid_mask]
    
    if len(all_dists_valid) == 0:
        return grads
    
    # Compute soft-min weights: w_k = exp(-d_k / beta) / sum(exp(-d_j / beta))
    scaled = -all_dists_valid / beta
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    weights = exp_scaled / np.sum(exp_scaled)
    
    # Wall gradient contributions
    n_wall = N * 3
    wall_weights = weights[:n_wall]
    
    for i in range(N):
        for k in range(3):
            w = wall_weights[i * 3 + k]
            # d/d(c_ik) (0.5 - |c_ik|) = -sign(c_ik)
            grads[i, k] += -w * np.sign(centers[i, k])
    
    # Pair gradient contributions
    for idx, (i, j) in enumerate(pair_indices):
        w = weights[n_wall + idx]
        diff = centers[i] - centers[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-15:
            continue
        # d/d(c_i) (|c_i - c_j|/2) = (c_i - c_j) / (2 * |c_i - c_j|)
        # d/d(c_j) (|c_i - c_j|/2) = -(c_i - c_j) / (2 * |c_i - c_j|)
        grad_vec = diff / (2.0 * dist)
        grads[i] += w * grad_vec
        grads[j] -= w * grad_vec
    
    return grads


def _initial_cubic_grid():
    """Create initial cubic grid placement."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _random_initial():
    """Create random initial placement."""
    return np.random.uniform(-0.5, 0.5, (N, 3))


def _gradient_optimization(centers, max_iters=5000, lr0=0.02, lr_decay=0.999, beta0=100.0, beta_decay=0.995):
    """Gradient-based optimization using soft-min surrogate."""
    centers = centers.copy()
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    lr = lr0
    beta = beta0
    
    for it in range(max_iters):
        if it % 100 == 0 and time.time() > 140:
            break
        
        # Compute gradient of soft-min radius
        grads = _compute_gradients(centers, beta)
        
        # Update positions (maximize radius = follow gradient)
        new_centers = centers + lr * grads
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_radius = _compute_radius(new_centers)
        
        if new_radius >= best_radius - 1e-12:
            centers = new_centers
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = centers.copy()
        else:
            # Reduce learning rate if we made things worse
            lr *= 0.9
        
        # Decay learning rate and increase beta (sharper soft-min)
        lr *= lr_decay
        beta = max(10.0, beta * beta_decay)
        
        if lr < 1e-7:
            break
    
    return best_centers, best_radius


def _local_optimization(centers, max_iters=2000, lr=0.01, cooling=0.999):
    """Local optimization using gradient-free approach with repulsive forces."""
    centers = centers.copy()
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Use a simple force-based approach
    current_radius = best_radius
    step_size = 0.05
    
    for it in range(max_iters):
        if it % 100 == 0 and time.time() > 140:
            break
            
        # Compute forces: repulsion between spheres and attraction to center
        forces = np.zeros_like(centers)
        
        # Repulsive forces between pairs
        for i in range(N):
            for j in range(i+1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    diff = np.random.uniform(-0.01, 0.01, 3)
                    dist = np.linalg.norm(diff)
                # Force proportional to 1/dist^2 for close spheres
                force_mag = 1.0 / (dist * dist + 1e-6)
                forces[i] += force_mag * diff / dist
                forces[j] -= force_mag * diff / dist
        
        # Wall repulsion
        for i in range(3):
            # Attract away from walls
            forces[:, i] += (0.5 - np.abs(centers[:, i])) * 0.1 * np.sign(centers[:, i])
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.where(force_norms < 1e-10, 1.0, force_norms)
        forces = forces / force_norms
        
        # Update positions
        new_centers = centers + step_size * forces
        # Clamp to cube
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_radius = _compute_radius(new_centers)
        
        if new_radius > current_radius:
            centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
        else:
            # Reduce step size on failure
            step_size *= 0.95
        
        if step_size < 1e-6:
            break
    
    return best_centers, best_radius


def _simulated_annealing(centers, max_iters=5000, T0=0.05, T_end=0.001):
    """Simulated annealing optimization."""
    centers = centers.copy()
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_radius = best_radius
    
    T = T0
    cooling_rate = (T_end / T0) ** (1.0 / max_iters)
    
    for it in range(max_iters):
        if it % 50 == 0 and time.time() > 140:
            break
            
        # Random perturbation
        perturbation = np.random.normal(0, 0.02, (N, 3))
        new_centers = centers + perturbation
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_radius = _compute_radius(new_centers)
        delta = new_radius - current_radius
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
        
        T *= cooling_rate
    
    return best_centers, best_radius


def _polish(centers, max_iters=2000):
    """High-precision polish pass with very small steps."""
    centers = centers.copy()
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    step_size = 0.001
    
    for it in range(max_iters):
        if it % 50 == 0 and time.time() > 145:
            break
        
        # Small random perturbations
        for trial in range(10):
            perturbation = np.random.normal(0, step_size, (N, 3))
            new_centers = centers + perturbation
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            new_radius = _compute_radius(new_centers)
            
            if new_radius > best_radius:
                centers = new_centers
                best_radius = new_radius
                best_centers = centers.copy()
        
        # Decay step size
        step_size *= 0.999
        
        if step_size < 1e-8:
            break
    
    return best_centers, best_radius


def construct():
    """Construct 50 sphere centers in unit cube to maximize minimum radius."""
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Start with cubic grid
    best_centers = _initial_cubic_grid()
    best_radius = _compute_radius(best_centers)
    
    # Phase 1: Gradient-based optimization from grid
    if time.time() < deadline - 30:
        grad_centers, grad_radius = _gradient_optimization(best_centers, max_iters=3000)
        if grad_radius > best_radius:
            best_centers = grad_centers
            best_radius = grad_radius
    
    # Phase 2: Local force-based optimization
    if time.time() < deadline - 20:
        opt_centers, opt_radius = _local_optimization(best_centers, max_iters=2000)
        if opt_radius > best_radius:
            best_centers = opt_centers
            best_radius = opt_radius
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 15:
        sa_centers, sa_radius = _simulated_annealing(best_centers, max_iters=8000, T0=0.03, T_end=0.0005)
        if sa_radius > best_radius:
            best_centers = sa_centers
            best_radius = sa_radius
    
    # Phase 4: Additional gradient refinement
    if time.time() < deadline - 10:
        grad_centers, grad_radius = _gradient_optimization(best_centers, max_iters=2000)
        if grad_radius > best_radius:
            best_centers = grad_centers
            best_radius = grad_radius
    
    # Phase 5: Random restarts with gradient optimization
    if time.time() < deadline - 20:
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            random_centers = _random_initial()
            # Gradient optimization from random start
            opt_c, opt_r = _gradient_optimization(random_centers, max_iters=1500)
            if opt_r > best_radius:
                best_centers = opt_c
                best_radius = opt_r
                # Polish
                if time.time() < deadline - 10:
                    pol_c, pol_r = _polish(best_centers, max_iters=500)
                    if pol_r > best_radius:
                        best_centers = pol_c
                        best_radius = pol_r
    
    # Final polish pass
    if time.time() < deadline - 5:
        pol_centers, pol_radius = _polish(best_centers, max_iters=1000)
        if pol_radius > best_radius:
            best_centers = pol_centers
            best_radius = pol_radius
    
    # Final verification
    final_radius = _compute_radius(best_centers)
    if final_radius <= 0:
        # Fallback to grid
        return _initial_cubic_grid()
    
    return best_centers
# EVOLVE-BLOCK-END
