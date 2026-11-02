import numpy as np
import time
import scipy.optimize as opt

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

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
    
    # Pairwise distances
    # Using broadcasting for simplicity, N=50 is small enough
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_wall)

def _soft_min_obj_and_grad(x, tau, lam_dirichlet=0.0):
    """
    Objective: minimize -soft_min(minimum distance to constraint boundary) + lam * dirichlet_energy.
    x: flattened array of 2*N coordinates
    Returns (value, gradient)
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Combine all constraint distances into a flat array
    iu, ju = np.triu_indices(N, k=1)
    n_pairs = len(iu)
    
    all_dists = np.empty(n_pairs + N)
    all_dists[:n_pairs] = pair_dists[iu, ju]
    all_dists[n_pairs:] = walls
    
    # Soft min calculation
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    # Avoid overflow in exp
    exp_vals = np.exp(-shifted / tau)
    sum_exp = np.sum(exp_vals)
    log_sum = np.log(sum_exp)
    soft_min = min_d - tau * log_sum
    
    value = -soft_min
    grad = np.zeros_like(x)
    
    # Weights for gradient
    weights = exp_vals / sum_exp
    
    # 1. Wall contributions
    for i in range(N):
        w_wall = weights[n_pairs + i]
        if w_wall < 1e-12: 
            continue
        if wall_dists_x[i] < wall_dists_y[i]:
            grad[2*i] += w_wall * np.sign(centers[i, 0])
        elif wall_dists_y[i] < wall_dists_x[i]:
            grad[2*i+1] += w_wall * np.sign(centers[i, 1])
        else:
            # Both are limiting
            grad[2*i] += 0.5 * w_wall * np.sign(centers[i, 0])
            grad[2*i+1] += 0.5 * w_wall * np.sign(centers[i, 1])

    # 2. Pair contributions
    for k in range(n_pairs):
        w_pair = weights[k]
        if w_pair < 1e-12:
            continue
        i = iu[k]
        j = ju[k]
        d = dists[i, j]
        if d < 1e-12:
            continue
        
        diff_ij = centers[i] - centers[j]
        factor = w_pair / (2.0 * d)
        
        grad[2*i:2*i+2] -= factor * diff_ij
        grad[2*j:2*j+2] += factor * diff_ij
        
    # 3. Dirichlet energy regularization (uniform spacing force)
    # E_dir = sum_{i,j} w_ij * |c_i - c_j|^2 where w_ij is based on distance
    # This creates a "membrane" effect that resists clustering
    if lam_dirichlet > 0:
        # Use a simple potential: sum of squared distances to all other points
        # This encourages uniform distribution
        for i in range(N):
            # Sum of squared distances to all other points
            d2 = np.sum((centers - centers[i])**2, axis=1)
            d2[i] = 0
            # Gradient of sum_j |c_i - c_j|^2 w.r.t c_i is 2 * sum_j (c_i - c_j)
            # = 2 * N * c_i - 2 * sum_j c_j
            # But we want to minimize this, so we add a repulsive force
            # Actually, let's use a simpler form: Laplacian-like force
            # Force on i = -sum_j (c_i - c_j) / |c_i - c_j|^3 (Coulomb-like)
            # Or just: Force on i = sum_j (c_j - c_i) / |c_j - c_i|^2 (softened)
            
            # Simpler: use the fact that for uniform distribution, the Laplacian should be zero
            # Approximate Laplacian using nearest neighbors
            # For simplicity, use: grad += lam * sum_j (c_j - c_i) / (|c_j - c_i|^2 + eps)
            
            eps = 1e-4
            diffs = centers - centers[i]
            d2_all = np.sum(diffs**2, axis=1) + eps
            d2_all[i] = 1e10  # Exclude self
            weights_dir = 1.0 / d2_all
            force = np.sum(diffs * weights_dir[:, np.newaxis], axis=0)
            grad[2*i:2*i+2] += lam_dirichlet * force
    
    return value, grad

def _hex_init(rng):
    """Generate a hexagonal lattice initialization."""
    # Generate a grid of points in hexagonal pattern
    n_rows = 10
    n_cols = 10
    spacing = 0.15 # Initial guess for spacing, will be scaled
    points = []
    for i in range(n_rows):
        for j in range(n_cols):
            x = j * spacing + (i % 2) * spacing * 0.5
            y = i * spacing * np.sqrt(3) / 2
            points.append([x, y])
            
    points = np.array(points)
    
    # Center the grid
    cx = points[:, 0].mean()
    cy = points[:, 1].mean()
    points[:, 0] -= cx
    points[:, 1] -= cy
    
    # Scale to fit in square
    max_ext = np.max(np.abs(points))
    if max_ext > 0:
        points /= max_ext * 1.1 # Add margin
    
    # Randomly select N points
    idx = rng.choice(len(points), N, replace=False)
    selected = points[idx]
    
    # Add small jitter
    selected += rng.normal(0, 0.01, selected.shape)
    selected = np.clip(selected, -0.49, 0.49)
    
    return selected

def _grid_init(rng):
    """Create initial grid layout with slight jitter."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N].copy()
    
    grid = grid + rng.normal(0, 0.005, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)
    
    return grid

def _lbfgs_optimize(x0, tau, time_budget, lam_dirichlet=0.0):
    """Run L-BFGS-B optimization on the soft-min objective."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f, g = _soft_min_obj_and_grad(x, tau, lam_dirichlet)
        return f, g
    
    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8,
                     'maxfun': 10000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = np.random.default_rng(777)
    
    best_centers = None
    best_radius = 0.0
    
    # Define initializations
    inits = [
        _hex_init(rng),
        _grid_init(rng),
        rng.uniform(-0.45, 0.45, (N, 2))
    ]
    
    # Phase 1: Global Search with Soft Min + Dirichlet Regularization
    # Use larger tau to smooth the landscape, with some Dirichlet energy to enforce uniformity
    taus_global = [0.02, 0.01, 0.005]
    lams_dirichlet = [0.01, 0.005, 0.001]  # Decreasing regularization
    
    for init in inits:
        if time.time() > deadline - 40:
            break
            
        current = init.copy()
        
        # Annealing loop with decreasing tau and lam
        for tau, lam in zip(taus_global, lams_dirichlet):
            if time.time() > deadline - 30:
                break
                
            # Optimize with current tau and lam
            trial = _lbfgs_optimize(current.flatten(), tau, time_budget=10.0, lam_dirichlet=lam)
            trial_r = _compute_radius(trial)
            
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial.copy()
                
            # Use the optimized point as the base for the next tighter tau
            current = trial
            
    # Phase 2: Local Refinement with Small Tau (no Dirichlet regularization)
    if best_centers is None:
        best_centers = _grid_init(rng)
        best_radius = _compute_radius(best_centers)
        
    if time.time() < deadline - 15:
        taus_local = [0.001, 0.0005, 0.0001, 0.00005]
        current = best_centers.copy()
        
        for tau in taus_local:
            if time.time() > deadline - 10:
                break
                
            trial = _lbfgs_optimize(current.flatten(), tau, time_budget=8.0, lam_dirichlet=0.0)
            trial_r = _compute_radius(trial)
            
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial.copy()
                
            current = trial
            
    # Phase 3: Ultra-tight final polish
    if time.time() < deadline - 5:
        trial = _lbfgs_optimize(best_centers.flatten(), 0.00001, time_budget=5.0, lam_dirichlet=0.0)
        trial_r = _compute_radius(trial)
        
        if trial_r > best_radius:
            best_radius = trial_r
            best_centers = trial.copy()
            
    # Final Validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _grid_init(rng)
    
    return best_centers
