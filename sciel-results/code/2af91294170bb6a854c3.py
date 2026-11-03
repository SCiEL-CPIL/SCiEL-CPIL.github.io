"""Strategy: Smooth soft-min energy maximization via L-BFGS-B + exact pairwise local-search polish."""
N = 200
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 115.0  # safety margin below 150s kill


def _compute_radius(centers):
    """Return the largest common radius for given centers in [-0.5,0.5]^2."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint
    boundary = np.min(np.minimum(
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    if boundary < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(boundary, min_pair / 2.0)


def _soft_min_neg(centers, beta=100.0):
    """
    Smooth approximation of -min(distance/2, boundary) using soft-min.
    We maximize the soft-min of all constraints (pairwise distances/2 and boundary distances).
    Constraints:
      - For each pair (i,j): d_ij / 2
      - For each point i, each side: 0.5 - |x_i|, 0.5 - |y_i|
    Soft-min: -log(sum(exp(-x_k / beta))) * beta
    """
    # Pairwise distances / 2
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    # Boundary distances
    bx = 0.5 - np.abs(centers[:, 0])
    by = 0.5 - np.abs(centers[:, 1])
    bound_vals = np.concatenate([bx, by])
    
    # Combine all constraints
    all_vals = np.concatenate([pair_vals.flatten(), bound_vals])
    
    # Soft-min: -beta * log(sum(exp(-x/beta)))
    # For numerical stability, subtract max
    max_val = np.max(all_vals)
    exp_terms = np.exp(-(all_vals - max_val) / beta)
    soft_min = max_val - beta * np.log(np.sum(exp_terms))
    
    return -soft_min  # We minimize this, so maximize soft_min


def _grad_soft_min_neg(centers, beta=100.0):
    """
    Gradient of -soft_min.
    soft_min = max_val - beta * log(sum(exp(-(x_k - max_val)/beta)))
    d(soft_min)/dx_i = (1/sum(exp(...))) * sum_j (exp(-(x_j - max_val)/beta) / beta) * dx_j/dx_i
    Since we want gradient of -soft_min, we negate it.
    """
    # Pairwise distances / 2
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    # Boundary distances
    bx = 0.5 - np.abs(centers[:, 0])
    by = 0.5 - np.abs(centers[:, 1])
    bound_vals = np.concatenate([bx, by])
    
    all_vals = np.concatenate([pair_vals.flatten(), bound_vals])
    
    max_val = np.max(all_vals)
    exp_terms = np.exp(-(all_vals - max_val) / beta)
    sum_exp = np.sum(exp_terms)
    
    # Weights for each constraint
    weights = exp_terms / sum_exp / beta  # d(soft_min)/dx_k = sum_j w_j * dx_j/dx_k
    
    grad = np.zeros_like(centers)
    
    # Pairwise contribution
    # For pair (i,j), x_k = d_ij/2
    # d(d_ij/2)/d(c_i) = (c_i - c_j) / d_ij
    # d(d_ij/2)/d(c_j) = (c_j - c_i) / d_ij
    pair_grad = np.zeros_like(centers)
    for i in range(N):
        for j in range(i+1, N):
            d = dists[i, j]
            if d < 1e-12:
                continue
            w = weights[i * N + j]  # weight for this pair
            vec = (centers[i] - centers[j]) / d
            pair_grad[i] += w * vec
            pair_grad[j] -= w * vec
    
    # Boundary contribution
    # bx_i = 0.5 - |x_i|, d(bx_i)/dx_i = -sign(x_i)
    # by_i = 0.5 - |y_i|, d(by_i)/dy_i = -sign(y_i)
    bound_grad = np.zeros_like(centers)
    for i in range(N):
        w_x = weights[N*N + i]  # weight for bx_i
        w_y = weights[N*N + N + i]  # weight for by_i
        bound_grad[i, 0] += w_x * (-np.sign(centers[i, 0]))
        bound_grad[i, 1] += w_y * (-np.sign(centers[i, 1]))
    
    total_grad = pair_grad + bound_grad
    return -total_grad  # Gradient of -soft_min


def _initial_hex_grid(n_points):
    """Generate a hexagonal lattice initial guess."""
    # Estimate spacing
    # Area per point ~ 1/200, so spacing ~ sqrt(1/200) ~ 0.0707
    # Hex packing: spacing = 2*r, area per circle = r^2 * sqrt(3) ~ 0.5 * spacing^2 * sqrt(3)/4 * ... 
    # Simpler: just use a grid with hex offset
    m = int(np.ceil(np.sqrt(n_points)))
    # Adjust to get close to n_points
    points = []
    # Hex lattice: rows offset by half spacing
    # Let's try a square grid first, then perturb
    x_coords = np.linspace(-0.45, 0.45, m)
    y_coords = np.linspace(-0.45, 0.45, m)
    
    for row, y in enumerate(y_coords):
        for x in x_coords:
            # Offset every other row by half the x-spacing
            offset = (x_coords[1] - x_coords[0]) / 2.0 if len(x_coords) > 1 else 0
            if row % 2 == 1:
                x = x + offset
            points.append([x, y])
    
    points = np.array(points)
    
    # Trim or pad to n_points
    if len(points) > n_points:
        # Keep points closest to center
        center_dist = np.sum(points ** 2, axis=1)
        idx = np.argsort(center_dist)[:n_points]
        points = points[idx]
    elif len(points) < n_points:
        extra = np.random.uniform(-0.45, 0.45, size=(n_points - len(points), 2))
        points = np.vstack([points, extra])
    
    return points[:n_points]


def _local_search_polish(centers, max_iters=5000, step_size=0.001):
    """
    Local search on the exact radius.
    For each point, try small random moves and accept if radius improves.
    """
    centers = centers.copy()
    r_best = _compute_radius(centers)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
            
        # Pick a random point
        idx = np.random.randint(N)
        
        # Try a small random move
        move = np.random.normal(0, step_size, size=2)
        new_centers = centers.copy()
        new_centers[idx] += move
        
        # Clip to square
        new_centers[idx, 0] = np.clip(new_centers[idx, 0], -0.5, 0.5)
        new_centers[idx, 1] = np.clip(new_centers[idx, 1], -0.5, 0.5)
        
        r_new = _compute_radius(new_centers)
        
        if r_new > r_best:
            centers = new_centers
            r_best = r_new
            # Optionally decrease step size
            step_size *= 0.999
        else:
            # Occasionally reset step size if stuck
            if it % 1000 == 0:
                step_size = max(0.0001, step_size * 1.5)
    
    return centers, r_best


def construct():
    start_time = time.time()
    
    # 1. Generate initial hex grid
    centers = _initial_hex_grid(N)
    
    # 2. Run L-BFGS-B optimization on the soft-min objective
    # Use a moderate beta for smoothness
    beta = 50.0
    
    def objective(x):
        return _soft_min_neg(x.reshape(N, 2), beta)
    
    def gradient(x):
        return _grad_soft_min_neg(x.reshape(N, 2), beta)
    
    # Bounds: keep centers within [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 2)
    
    # Run optimization
    try:
        result = minimize(
            objective,
            centers.flatten(),
            jac=gradient,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        centers = result.x.reshape(N, 2)
    except Exception:
        pass  # Keep initial centers if optimization fails
    
    # 3. Local search polish on the exact radius
    centers, r_best = _local_search_polish(centers, max_iters=10000, step_size=0.002)
    
    # 4. Multi-start: Try a few different initializations if time permits
    for i in range(3):
        if time.time() > _DEADLINE - 20:
            break
            
        # Different initializations
        np.random.seed(42 + i * 100)
        if i == 0:
            init = _initial_hex_grid(N)
        elif i == 1:
            # Slightly perturbed hex
            init = _initial_hex_grid(N) + np.random.normal(0, 0.01, size=(N, 2))
            init[:, 0] = np.clip(init[:, 0], -0.49, 0.49)
            init[:, 1] = np.clip(init[:, 1], -0.49, 0.49)
        else:
            # Random initialization
            init = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        # Run optimization with different beta
        beta_i = 30.0 + i * 10
        
        def objective_i(x):
            return _soft_min_neg(x.reshape(N, 2), beta_i)
        
        def gradient_i(x):
            return _grad_soft_min_neg(x.reshape(N, 2), beta_i)
        
        try:
            result = minimize(
                objective_i,
                init.flatten(),
                jac=gradient_i,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            centers_i = result.x.reshape(N, 2)
        except Exception:
            centers_i = init
        
        # Local search polish
        centers_i, r_i = _local_search_polish(centers_i, max_iters=5000, step_size=0.001)
        
        if r_i > r_best:
            centers = centers_i
            r_best = r_i
    
    # Final validation
    r_final = _compute_radius(centers)
    if r_final <= 0:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        centers = grid[:N]
    
    return centers
