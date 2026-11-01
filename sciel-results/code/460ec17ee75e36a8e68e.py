"""Packing of 50 equal circles in a unit square via soft-min optimization + SA + L-BFGS polish."""
import numpy as np
import time
import scipy.optimize as opt

N = 50

def _compute_r(centers):
    """Compute the true maximum radius for a given configuration."""
    # Wall distances
    wall_dists = np.minimum(0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1]))
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_wall)

def _soft_min_obj(x, tau):
    """
    Objective: minimize negative soft-min of all constraints.
    Constraints: pairwise distances >= 2r, wall distances >= r.
    We encode this as: for each pair (i,j), constraint value = dist_ij - 2*r_target
    For walls: constraint value = wall_dist_i - r_target
    We want all constraints >= 0. We minimize -soft_min(constraints).
    """
    r_target = x[-1]  # last element is the radius
    centers = x[:-1].reshape(-1, 2)
    
    # Wall distances
    wall_dists = np.minimum(0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1]))
    wall_constraints = wall_dists - r_target
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_constraints = dists - 2 * r_target
    
    # Combine all constraints
    all_constraints = np.concatenate([wall_constraints, pair_constraints.ravel()])
    
    # Soft-min: -tau * log(sum(exp(-c/tau))) approximates min(c)
    # We want to maximize the min constraint, so minimize -soft_min
    # soft_min = -tau * log(sum(exp(-c/tau)))
    # To avoid overflow, use log-sum-exp trick
    shifted = -all_constraints / tau
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    soft_min = -tau * log_sum_exp
    
    return -soft_min

def _soft_min_grad(x, tau):
    """Gradient of the soft-min objective."""
    r_target = x[-1]
    centers = x[:-1].reshape(-1, 2)
    
    # Wall distances
    wall_dists = np.minimum(0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1]))
    wall_constraints = wall_dists - r_target
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_constraints = dists - 2 * r_target
    
    all_constraints = np.concatenate([wall_constraints, pair_constraints.ravel()])
    
    # Soft-min weights
    shifted = -all_constraints / tau
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    grad = np.zeros_like(x)
    
    # Wall constraint gradients
    # wall_dists = min(0.5 - |x|, 0.5 - |y|)
    # d(wall_dists)/d(centers) depends on which is min
    abs_x = np.abs(centers[:, 0])
    abs_y = np.abs(centers[:, 1])
    wall_x = 0.5 - abs_x
    wall_y = 0.5 - abs_y
    
    # Gradient of wall_x w.r.t. x
    grad_wall_x = -np.sign(centers[:, 0])
    grad_wall_y = -np.sign(centers[:, 1])
    
    # For each wall constraint, the gradient is the min of the two wall distance gradients
    # We need to handle the min carefully
    for i in range(N):
        w = weights[i]  # weight for wall constraint i
        if wall_x[i] < wall_y[i]:
            grad[i * 2] += w * grad_wall_x[i]
            grad[i * 2 + 1] += 0
        else:
            grad[i * 2] += 0
            grad[i * 2 + 1] += w * grad_wall_y[i]
        grad[-1] += -w  # d(wall_constraint)/dr = -1
    
    # Pairwise constraint gradients
    # pair_constraints[k] = dists[i,j] - 2*r for pair (i,j)
    # d(dist_ij)/d(centers[i]) = (centers[i] - centers[j]) / dist_ij
    # d(pair_constraint)/dr = -2
    
    n_pairs = N * (N - 1) // 2
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            w = weights[N + idx]
            diff_ij = centers[i] - centers[j]
            dist_ij = np.linalg.norm(diff_ij)
            if dist_ij > 1e-10:
                grad_dir = diff_ij / dist_ij
                grad[i * 2:i * 2 + 2] += w * grad_dir
                grad[j * 2:j * 2 + 2] -= w * grad_dir
            grad[-1] += -2 * w
            idx += 1
    
    return grad

def _violation(centers, r):
    """Check if configuration is feasible: all circles inside and non-overlapping."""
    wall_dists = np.minimum(0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1]))
    if np.min(wall_dists) < r - 1e-10:
        return False
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 2 * r - 1e-10:
        return False
    return True

def _simulated_annealing(centers, r_target, max_steps=50000, T_start=0.05, T_end=0.001):
    """
    Simulated annealing to find a configuration with radius >= r_target.
    """
    rng = np.random.default_rng(42)
    current = centers.copy()
    current_feasible = _violation(current, r_target)
    
    alpha = (T_end / T_start) ** (1.0 / max_steps)
    
    for step in range(max_steps):
        if time.time() > deadline - 30.0:
            break
        
        T = T_start * (alpha ** step)
        
        # Choose 1-3 circles to move
        num_movers = rng.integers(1, 4)
        indices = rng.choice(N, size=num_movers, replace=False)
        
        step_size = 0.03 * (0.3 + 0.7 * T / T_start)
        
        new_centers = current.copy()
        new_centers[indices] += rng.normal(0, step_size, (num_movers, 2))
        new_centers = np.clip(new_centers, -0.4999, 0.4999)
        
        new_feasible = _violation(new_centers, r_target)
        
        if new_feasible and not current_feasible:
            current = new_centers
            current_feasible = True
        elif new_feasible and current_feasible:
            # Both feasible, prefer one with more margin
            # Compute margin
            def _margin(c):
                wall_dists = np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1]))
                min_wall = np.min(wall_dists)
                diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                np.fill_diagonal(dists, np.inf)
                min_pair = np.min(dists) / 2.0
                return min(min_pair, min_wall) - r_target
            
            if _margin(new_centers) > _margin(current):
                current = new_centers
        else:
            # Accept with probability
            if not new_feasible and current_feasible:
                prob = 0.01
            else:
                prob = np.exp(-1.0 / T)
            if rng.random() < prob:
                current = new_centers
                current_feasible = new_feasible
    
    return current, current_feasible

def _lbfgs_polish(centers, r_target, max_iter=200):
    """
    Use L-BFGS to polish a near-feasible configuration.
    """
    x0 = np.concatenate([centers.ravel(), [r_target]])
    
    try:
        result = opt.minimize(
            _soft_min_obj,
            x0,
            args=(0.01,),
            method='L-BFGS-B',
            jac=_soft_min_grad,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.success:
            new_centers = result.x[:-1].reshape(-1, 2)
            new_r = result.x[-1]
            if _violation(new_centers, new_r):
                return new_centers, new_r
    except Exception:
        pass
    
    return centers, r_target

def _hex_init():
    """Hexagonal lattice initialization."""
    points = []
    spacing = 0.18
    for i in range(12):
        for j in range(12):
            x = -0.5 + j * spacing + (i % 2) * spacing / 2
            y = -0.5 + i * spacing * 0.866
            if -0.49 <= x <= 0.49 and -0.49 <= y <= 0.49:
                points.append([x, y])
    rng = np.random.default_rng(42)
    rng.shuffle(points)
    return np.array(points[:N])

def _grid_init():
    """Grid initialization with jitter."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    init = grid[:N].copy()
    rng = np.random.default_rng(42)
    init += rng.normal(0, 0.02, init.shape)
    return np.clip(init, -0.49, 0.49)

def _random_init():
    """Random initialization."""
    rng = np.random.default_rng(42)
    return rng.uniform(-0.45, 0.45, (N, 2))

def _binary_search_radius(centers, r_low, r_high, max_iterations=10):
    """
    Binary search for the maximum feasible radius.
    """
    best_centers = centers.copy()
    best_r = r_low
    
    for _ in range(max_iterations):
        if time.time() > deadline - 10.0:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a feasible configuration at radius r_mid
        # Use SA to try to reach feasibility
        sa_centers, feasible = _simulated_annealing(
            centers, r_mid, 
            max_steps=10000, 
            T_start=0.03, 
            T_end=0.001
        )
        
        if feasible:
            # Try to polish
            polished, polished_r = _lbfgs_polish(sa_centers, r_mid, max_iter=100)
            if _violation(polished, polished_r):
                best_centers = polished
                best_r = polished_r
                r_low = polished_r
            else:
                best_centers = sa_centers
                best_r = r_mid
                r_low = r_mid
        else:
            r_high = r_mid
    
    return best_centers, best_r

def _sdp_relaxation_init():
    """
    SDP relaxation: lift centers into Gram matrix.
    For N circles with centers c_i, define G = [c_i c_j^T] (N x N matrix).
    Actually, we use the squared distances: d_ij = |c_i - c_j|^2 = |c_i|^2 + |c_j|^2 - 2 c_i^T c_j.
    
    A simpler approach: use the fact that for a feasible packing with radius r,
    we need |c_i - c_j| >= 2r for all i != j, and |c_i|_inf <= 0.5 - r.
    
    We can formulate this as: find c_i such that the minimum pairwise distance is maximized
    subject to wall constraints. This is equivalent to maximizing r such that:
    - (c_i - c_j)^T (c_i - c_j) >= (2r)^2 for all i != j
    - (c_i,k)^2 <= (0.5 - r)^2 for all i, k
    
    The SDP relaxation lifts this to a Gram matrix problem.
    """
    # For simplicity, we'll use a heuristic based on the known optimal packing structure.
    # The best known packing has a specific arrangement. We'll start from a hexagonal
    # lattice and refine.
    
    # Start from hexagonal
    centers = _hex_init()
    
    # Apply a few L-BFGS steps to improve
    r_initial = _compute_r(centers)
    x0 = np.concatenate([centers.ravel(), [r_initial]])
    
    try:
        result = opt.minimize(
            _soft_min_obj,
            x0,
            args=(0.02,),
            method='L-BFGS-B',
            jac=_soft_min_grad,
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        if result.success:
            new_centers = result.x[:-1].reshape(-1, 2)
            if _violation(new_centers, result.x[-1]):
                return new_centers
    except Exception:
        pass
    
    return centers

def construct():
    global deadline
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate multiple initializations
    inits = [
        _hex_init(),
        _grid_init(),
        _random_init(),
        _sdp_relaxation_init()
    ]
    
    # Evaluate all inits
    radii = [_compute_r(c) for c in inits]
    best_init_idx = np.argmax(radii)
    
    best_centers = inits[best_init_idx].copy()
    best_radius = radii[best_init_idx]
    
    # Binary search for better radius
    if time.time() < deadline - 20.0:
        # Start with a conservative lower bound
        r_low = best_radius * 0.9
        r_high = 0.075  # Slightly above known best
        
        refined_centers, refined_radius = _binary_search_radius(
            best_centers, r_low, r_high, max_iterations=8
        )
        
        if refined_radius > best_radius:
            best_radius = refined_radius
            best_centers = refined_centers
    
    # Final L-BFGS polish
    if time.time() < deadline - 5.0:
        polished, polished_r = _lbfgs_polish(best_centers, best_radius, max_iter=200)
        if _violation(polished, polished_r) and polished_r > best_radius:
            best_radius = polished_r
            best_centers = polished
    
    return best_centers
