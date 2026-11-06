"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_radius(centers):
    """Compute the radius (min of pairwise dist/2 and wall dist)."""
    n = len(centers)
    # Wall distances
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_pair, np.min(wall_dists))

def _soft_min_radius(centers, beta=50.0):
    """Smooth approximation of the minimum radius using log-sum-exp."""
    n = len(centers)
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)  # shape (n, 3)
    wall_min_per_point = np.min(wall_dists, axis=1)  # shape (n,)
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0  # shape (n, n)
    
    # Collect all constraint values
    # Wall constraints: n values
    # Pair constraints: n*(n-1)/2 values (upper triangle)
    pair_upper = pair_dists[np.triu_indices(n, k=1)]
    
    all_constraints = np.concatenate([wall_min_per_point, pair_upper])
    
    # Soft min: -log(sum(exp(-x/beta)))/beta
    # For numerical stability, subtract max
    x = all_constraints
    x_max = np.max(x)
    soft_min = x_max - np.log(np.sum(np.exp(-(x - x_max) / beta))) * beta / len(x)
    # Actually: soft_min = -beta * log(mean(exp(-x/beta)))
    # = -beta * log(sum(exp(-x/beta))/N)
    # = x_max - beta * log(sum(exp(-(x-x_max)/beta))/N)
    log_sum = np.log(np.sum(np.exp(-(x - x_max) / beta)))
    soft_min = x_max - beta * (log_sum - np.log(len(x)))
    
    return -soft_min  # We want to minimize negative of soft_min

def _grad_soft_min_radius(centers, beta=50.0):
    """Gradient of negative soft-min radius."""
    n = len(centers)
    wall_dists = 0.5 - np.abs(centers)
    wall_min_idx = np.argmin(wall_dists, axis=1)
    wall_min_per_point = np.min(wall_dists, axis=1)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    pair_upper = pair_dists[np.triu_indices(n, k=1)]
    all_constraints = np.concatenate([wall_min_per_point, pair_upper])
    
    x = all_constraints
    x_max = np.max(x)
    exp_vals = np.exp(-(x - x_max) / beta)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # softmax-like weights for the min
    
    grad = np.zeros_like(centers)
    
    # Wall contributions
    wall_weights = weights[:n]
    for i in range(n):
        d = wall_dists[i]
        sign = -np.sign(centers[i])
        grad[i] += wall_weights[i] * sign / beta * (-1)  # d/dx of wall_dist
        # Actually: d(0.5 - |x|)/dx = -sign(x)
        # soft_min = x_max - beta*(log_sum - log(N))
        # d(soft_min)/dx_i = -beta * (1/sum_exp) * d(sum_exp)/dx_i
        # d(sum_exp)/dx_i = sum_j weights_j * (-1/beta) * d(x_j)/dx_i
        # This is getting complex. Let me use a different approach.
    
    # Simpler: use automatic differentiation via finite differences or just use a simpler objective
    # Let me use a penalty-based approach instead
    
    return grad

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    best_radius = 0.0
    best_centers = None
    
    # Initialize on a 5x5x4 grid pattern
    # Create 100 points, select 50
    def make_initial_grid(seed=0):
        rng = np.random.RandomState(seed)
        # 5x5x4 grid
        xs = np.linspace(-0.5 + 0.5/5, 0.5 - 0.5/5, 5)
        ys = np.linspace(-0.5 + 0.5/5, 0.5 - 0.5/5, 5)
        zs = np.linspace(-0.5 + 0.5/4, 0.5 - 0.5/4, 4)
        
        points = []
        for z in zs:
            for y in ys:
                for x in xs:
                    points.append([x, y, z])
        points = np.array(points)
        
        # Select 50 using a checkerboard-like pattern
        # Use alternating pattern in z-layers
        indices = []
        for iz in range(4):
            for iy in range(5):
                for ix in range(5):
                    idx = iz * 25 + iy * 5 + ix
                    # Checkerboard: include if (ix + iy + iz) % 2 == 0
                    if (ix + iy + iz) % 2 == 0:
                        indices.append(idx)
        
        # We have 50 from checkerboard
        selected = points[indices[:50]]
        
        # Add small random perturbation
        selected = selected + rng.randn(50, 3) * 0.001
        return selected
    
    def objective(params, beta=30.0):
        """Negative of soft-min radius (to minimize)."""
        centers = params.reshape(-1, 3)
        n = len(centers)
        
        # Wall distances per point
        wall_dists = 0.5 - np.abs(centers)
        wall_min_per_point = np.min(wall_dists, axis=1)
        
        # Pairwise distances / 2
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        pair_dists = np.sqrt(dists_sq) / 2.0
        
        pair_upper = pair_dists[np.triu_indices(n, k=1)]
        all_constraints = np.concatenate([wall_min_per_point, pair_upper])
        
        # Soft min
        x = all_constraints
        x_max = np.max(x)
        # Clip for numerical stability
        shifted = (x - x_max) / beta
        shifted = np.clip(shifted, -100, 0)
        log_sum = np.log(np.sum(np.exp(-shifted)))
        soft_min = x_max - beta * (log_sum - np.log(len(x)))
        
        return -soft_min
    
    def objective_grad(params, beta=30.0):
        """Gradient of negative soft-min radius."""
        centers = params.reshape(-1, 3)
        n = len(centers)
        
        wall_dists = 0.5 - np.abs(centers)
        wall_min_idx = np.argmin(wall_dists, axis=1)
        wall_min_per_point = np.min(wall_dists, axis=1)
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        pair_dists = np.sqrt(dists_sq) / 2.0
        
        pair_upper = pair_dists[np.triu_indices(n, k=1)]
        all_constraints = np.concatenate([wall_min_per_point, pair_upper])
        
        x = all_constraints
        x_max = np.max(x)
        shifted = (x - x_max) / beta
        shifted = np.clip(shifted, -100, 0)
        exp_vals = np.exp(-shifted)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
        
        grad = np.zeros_like(centers)
        
        # Wall contributions
        for i in range(n):
            w = weights[i]
            # d(wall_min_per_point[i])/d(centers[i]) = -sign(centers[i, wall_min_idx[i]])
            sign = -np.sign(centers[i, wall_min_idx[i]])
            if centers[i, wall_min_idx[i]] == 0:
                sign = 0
            grad[i, wall_min_idx[i]] += w * sign / beta
        
        # Pair contributions
        # For each pair (i,j) with i<j, constraint is dist(i,j)/2
        # d(dist(i,j)/2)/d(centers[i]) = (centers[i] - centers[j]) / (2 * dist(i,j))
        # d(dist(i,j)/2)/d(centers[j]) = (centers[j] - centers[i]) / (2 * dist(i,j))
        
        # Get upper triangle indices
        iu, ju = np.triu_indices(n, k=1)
        for k in range(len(iu)):
            i, j = iu[k], ju[k]
            w = weights[n + k]
            d = pair_dists[i, j]
            if d < 1e-10:
                continue
            direction = (centers[i] - centers[j]) / (2 * d)
            grad[i] += w * direction / beta
            grad[j] += w * (-direction) / beta
        
        return grad.flatten()
    
    # Multiple restarts
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() - start_time > deadline:
            break
        
        if restart == 0:
            centers = make_initial_grid(seed=0)
        elif restart == 1:
            centers = make_initial_grid(seed=1)
        elif restart == 2:
            centers = make_initial_grid(seed=2)
        else:
            # Perturb best solution found so far
            if best_centers is not None:
                centers = best_centers + np.random.randn(50, 3) * 0.01
            else:
                centers = make_initial_grid(seed=restart)
        
        # Clamp to cube
        centers = np.clip(centers, -0.49, 0.49)
        
        params = centers.flatten()
        
        # Try different beta values
        betas = [20.0, 40.0, 80.0]
        for beta in betas:
            if time.time() - start_time > deadline:
                break
            
            try:
                result = minimize(
                    objective,
                    params,
                    args=(beta,),
                    jac=objective_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
                )
                
                if result.success or result.fun < objective(params, beta):
                    opt_centers = result.x.reshape(-1, 3)
                    opt_centers = np.clip(opt_centers, -0.5, 0.5)
                    r = _compute_radius(opt_centers)
                    
                    if r > best_radius:
                        best_radius = r
                        best_centers = opt_centers.copy()
                        
            except Exception:
                pass
    
    if best_centers is None:
        # Fallback to grid
        centers = make_initial_grid(seed=0)
        best_centers = centers
        best_radius = _compute_radius(centers)
    
    return best_centers
# EVOLVE-BLOCK-END
