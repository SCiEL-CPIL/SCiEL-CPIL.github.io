import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the largest common radius for the given centers in [-0.5, 0.5]^3."""
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(x, beta=50.0, target_r=0.111):
    """
    Soft-min objective: minimize the negative of a smooth approximation of the minimum
    pairwise distance and wall distance. Lower is better.
    """
    centers = x.reshape(-1, 3)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    
    # Pairwise distances
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists[dists < np.inf]
    
    # Combine all distances (pair distances contribute half to radius)
    # We want to maximize min(wall_dists, pair_dists/2)
    # So minimize -min(...)
    
    # Soft min of wall distances
    min_wall = -np.log(np.sum(np.exp(-beta * wall_dists))) / beta
    
    # Soft min of pair distances / 2
    min_pair = -np.log(np.sum(np.exp(-beta * pair_dists / 2.0))) / beta
    
    # Soft min of the two
    # Use a log-sum-exp trick: softmin(a, b) = -log(exp(-beta*a) + exp(-beta*b))/beta
    soft_min = -np.log(np.exp(-beta * min_wall) + np.exp(-beta * min_pair)) / beta
    
    return -soft_min  # We want to maximize soft_min, so minimize its negative

def _soft_min_gradient(x, beta=50.0):
    """
    Analytical gradient of the soft-min objective.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    # Gradient of wall dist w.r.t. center[i,k] = -sign(centers[i,k])
    wall_grad = -np.sign(centers)
    wall_grad[np.abs(centers) < 1e-12] = 0.0
    
    # Soft min of wall distances
    exp_wall = np.exp(-beta * wall_dists)
    sum_exp_wall = np.sum(exp_wall)
    weights_wall = exp_wall / sum_exp_wall  # shape (n, 3)
    
    # Pairwise distances
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    
    # For pair distances, gradient of dist(i,j) w.r.t. center[i] = (center[i] - center[j]) / dist(i,j)
    # and w.r.t. center[j] = (center[j] - center[i]) / dist(i,j)
    
    # Soft min of pair_dists/2
    pair_dists_half = dists / 2.0
    exp_pair = np.exp(-beta * pair_dists_half)
    exp_pair[np.isinf(pair_dists_half)] = 0.0
    sum_exp_pair = np.sum(exp_pair)
    weights_pair = exp_pair / sum_exp_pair  # shape (n, n)
    
    # Gradient of pair soft min w.r.t. centers
    pair_grad = np.zeros_like(centers)
    for i in range(n):
        for j in range(n):
            if i == j or not np.isfinite(dists[i, j]):
                continue
            w = weights_pair[i, j]
            diff = centers[i] - centers[j]
            d = dists[i, j]
            if d > 1e-12:
                pair_grad[i] += w * diff / d
                pair_grad[j] += w * (-diff) / d
    
    # Combine gradients
    # d(soft_min)/d(x) = weights_wall * wall_grad + pair_grad
    # But we're minimizing -soft_min, so gradient is negative of that
    grad = -(weights_wall * wall_grad + pair_grad)
    
    return grad.flatten()

def _build_initial_grid():
    """Build a grid-based initialization."""
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    init = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))[:N]
    return init

def _build_bcc_init():
    """Build a BCC-like initialization."""
    points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                points.append([i*0.25 - 0.45, j*0.25 - 0.45, k*0.25 - 0.45])
                points.append([i*0.25 + 0.125 - 0.45, j*0.25 + 0.125 - 0.45, k*0.25 + 0.125 - 0.45])
    points = np.array(points)
    mask = np.all(np.abs(points) < 0.45, axis=1)
    points = points[mask]
    if len(points) > N:
        rng = np.random.default_rng(43)
        idx = rng.choice(len(points), N, replace=False)
        points = points[idx]
    if len(points) < N:
        rng = np.random.default_rng(43)
        pad = rng.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, pad])
    return points

def _build_halton_init():
    """Build a Halton sequence initialization for quasi-random points."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, N + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        z = halton(i, 5)
        # Map from [0,1] to [-0.45, 0.45]
        points.append([x * 0.9 - 0.45, y * 0.9 - 0.45, z * 0.9 - 0.45])
    return np.array(points)

def _polish_repulsion(centers, deadline, rng):
    """
    Final polish using repulsion forces and small random steps.
    """
    c = centers.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    step_size = 0.005
    min_step = 0.0005
    decay = 0.9995
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        # Compute forces
        dists = cdist(c, c, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        
        # Find closest pairs
        min_dist = np.min(dists)
        if min_dist > 0.25:
            break
        
        # Apply repulsive forces to all close pairs
        forces = np.zeros_like(c)
        for i in range(N):
            for j in range(i + 1, N):
                d = dists[i, j]
                if d < 0.25 and d > 1e-12:
                    diff = c[i] - c[j]
                    # Force magnitude inversely proportional to distance
                    f_mag = step_size * 0.5 / (d + 0.01)
                    forces[i] += diff / d * f_mag
                    forces[j] -= diff / d * f_mag
        
        # Wall repulsion
        for k in range(3):
            close_left = c[:, k] < -0.4
            close_right = c[:, k] > 0.4
            forces[close_left, k] += step_size * 0.3
            forces[close_right, k] -= step_size * 0.3
        
        c_new = c + forces
        c_new += rng.normal(0, step_size * 0.05, c_new.shape)
        c_new = np.clip(c_new, -0.49, 0.49)
        
        r_new = _compute_radius(c_new)
        
        if r_new > best_r + 1e-12:
            c = c_new
            best_c = c_new.copy()
            best_r = r_new
            no_improve = 0
        else:
            no_improve += 1
        
        step_size = max(min_step, step_size * decay)
        
        if no_improve > max_no_improve:
            kick_idx = rng.choice(N, size=10, replace=False)
            c[kick_idx] += rng.normal(0, 0.05, (10, 3))
            c = np.clip(c, -0.49, 0.49)
            no_improve = 0
            step_size = 0.005
    
    return best_c, best_r

def _optimize_from_initial(init, deadline, rng):
    """
    Optimize from an initial configuration using L-BFGS-B with soft-min objective,
    followed by repulsion polish.
    """
    x0 = init.flatten()
    best_x = x0.copy()
    best_r = _compute_radius(init)
    
    # Phase 1: L-BFGS-B optimization with increasing beta
    betas = [10.0, 20.0, 40.0, 80.0]
    
    for beta in betas:
        if time.time() > deadline - 5.0:
            break
        
        # Time allocation for this beta phase
        phase_deadline = deadline - 5.0
        if len(betas) > 1:
            remaining_betas = betas[betas.index(beta) + 1:]
            phase_deadline = min(phase_deadline, time.time() + 30.0)
        
        result = minimize(
            _soft_min_objective,
            x0,
            args=(beta,),
            method='L-BFGS-B',
            jac=_soft_min_gradient,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        
        x_opt = np.clip(result.x, -0.49, 0.49).reshape(-1, 3)
        r_opt = _compute_radius(x_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_x = x_opt.flatten()
            x0 = best_x.copy()
    
    # Phase 2: Repulsion polish
    if time.time() < deadline - 2.0:
        c_final, r_final = _polish_repulsion(best_x.reshape(-1, 3), deadline - 0.5, rng)
        if r_final > best_r:
            best_r = r_final
            best_x = c_final.flatten()
    
    return best_x.reshape(-1, 3), best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_centers = None
    best_radius = -np.inf
    
    # Multiple diverse initializations
    inits = [
        _build_initial_grid(),
        _build_bcc_init(),
        _build_halton_init(),
    ]
    
    # Add perturbed versions for diversity
    for i in range(3):
        perturbed = inits[i] + rng.normal(0, 0.03, inits[i].shape)
        perturbed = np.clip(perturbed, -0.49, 0.49)
        inits.append(perturbed)
    
    # Sort by initial radius to prioritize good starts
    init_radii = [_compute_radius(i) for i in inits]
    sorted_idx = np.argsort(init_radii)[::-1]
    inits = [inits[i] for i in sorted_idx]
    
    # Time allocation: give more time to better initializations
    n_inits = len(inits)
    weights = np.array([0.35, 0.25, 0.2, 0.1, 0.1])
    weights = weights / weights.sum()
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
        
        # Allocate time proportionally
        time_left = deadline - time.time()
        alloc = time_left * weights[i]
        alloc = min(alloc, time_left - 5.0)  # Leave buffer
        slice_deadline = time.time() + alloc
        
        c_opt, r_opt = _optimize_from_initial(init, slice_deadline, rng)
        
        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = c_opt.copy()
    
    # Final global polish on best found
    if best_centers is not None and time.time() < deadline - 3.0:
        c_final, r_final = _polish_repulsion(best_centers, deadline - 0.5, rng)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = c_final.copy()
    
    if best_centers is None:
        best_centers = inits[0]
    
    return best_centers

# EVOLVE-BLOCK-END
