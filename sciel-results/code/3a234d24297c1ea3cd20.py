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

def _soft_min_obj_and_grad(x, tau):
    """
    Objective: minimize -soft_min(minimum distance to constraint boundary).
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
    
    # Weights for gradient
    weights = exp_vals / sum_exp
    
    grad = np.zeros_like(x)
    
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
        
    return value, grad

def _hex_init(rng):
    """Generate a hexagonal lattice initialization."""
    # Hexagonal packing density is pi/sqrt(12) ~ 0.9069
    # For N=50, we need a grid slightly larger than sqrt(50) ~ 7.07
    # Let's use a 7x8 or 8x7 grid and trim
    # Side length of hexagon 's' (distance to vertex)
    # Distance between centers is 2*r.
    # Let's just create a hex grid of points and pick the best 50.
    
    # Generate a grid of points in hexagonal pattern
    # Row 0: y=0, x = 0, 1, 2, ...
    # Row 1: y=sqrt(3)/2, x = 0.5, 1.5, 2.5, ...
    
    # We want to fit 50 points in [-0.5, 0.5]
    # Approx radius 0.071. Diameter 0.142.
    # 50 circles in area 1. Area per circle 0.02.
    # Hexagon area for 1 circle ~ (2r)^2 * sqrt(3)/2 = 4r^2 * 0.866 = 3.464 r^2.
    # 3.464 * 0.071^2 = 3.464 * 0.005 = 0.017.
    # So hex packing is denser than average, which is good.
    
    # Let's generate a large hex grid and center it.
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
    # Find max extent
    max_ext = np.max(np.abs(points))
    if max_ext > 0:
        points /= max_ext * 1.1 # Add margin
    
    # Select N points closest to center? Or just take first N?
    # Taking first N from a centered grid might be biased.
    # Let's just take a random subset of size N from the grid points?
    # Or better: Generate a grid that is roughly square in shape.
    
    # Alternative: Just use a square grid with hexagonal offset for every other row.
    # This is effectively what the code above does, but let's make it more compact.
    
    # Let's try a different approach: 
    # Create a grid of size 8x8 (64 points), remove 14 random ones.
    # This ensures good coverage.
    
    m = 8
    ticks_x = np.linspace(-0.45, 0.45, m)
    ticks_y = np.linspace(-0.45, 0.45, m)
    
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    
    # Apply hexagonal offset to every other row
    for i in range(0, m, 2):
        grid[i*m:(i+1)*m, 0] += 0.45 / m # Shift right
        
    # Clip to square
    grid = np.clip(grid, -0.49, 0.49)
    
    # Randomly select N points
    idx = rng.choice(len(grid), N, replace=False)
    selected = grid[idx]
    
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

def _lbfgs_optimize(x0, tau, time_budget):
    """Run L-BFGS-B optimization on the soft-min objective."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f, g = _soft_min_obj_and_grad(x, tau)
        return f, g
    
    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8,
                     'maxfun': 5000, 'disp': False}
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
    
    # Phase 1: Global Search with Soft Min
    # Use larger tau to smooth the landscape
    taus_global = [0.02, 0.01, 0.005]
    
    for init in inits:
        if time.time() > deadline - 60:
            break
            
        current = init.copy()
        
        # Anneling/Polishing loop
        for tau in taus_global:
            if time.time() > deadline - 50:
                break
                
            # Optimize with current tau
            trial = _lbfgs_optimize(current.flatten(), tau, time_budget=15.0)
            trial_r = _compute_radius(trial)
            
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial.copy()
                
            # Stochastic Perturbation to escape local minima
            # Add noise proportional to current radius or fixed small amount
            noise_scale = 0.05
            perturbed = trial + rng.normal(0, noise_scale, trial.shape)
            perturbed = np.clip(perturbed, -0.49, 0.49)
            
            # Accept perturbation if it doesn't make things too bad?
            # Or just use it as the next start point?
            # Let's use the optimized point as the base, but if we are stuck, use perturbed.
            # Simple strategy: always use the optimized point for the next tighter tau.
            current = trial
            
    # Phase 2: Local Refinement with Small Tau
    if best_centers is None:
        best_centers = _grid_init(rng)
        best_radius = _compute_radius(best_centers)
        
    if time.time() < deadline - 20:
        taus_local = [0.001, 0.0005, 0.0001]
        current = best_centers.copy()
        
        for tau in taus_local:
            if time.time() > deadline - 10:
                break
                
            trial = _lbfgs_optimize(current.flatten(), tau, time_budget=10.0)
            trial_r = _compute_radius(trial)
            
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial.copy()
                
            current = trial
            
    # Final Validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _grid_init(rng)
    
    return best_centers
