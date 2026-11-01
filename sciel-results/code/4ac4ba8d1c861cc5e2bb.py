import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 100
DEADLINE = 145.0  # Safety margin from 150s

def _compute_radius(centers):
    """Compute the valid radius for a given set of centers."""
    wall_dist = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_dist)
    d = pdist(centers, 'euclidean')
    r_pair = np.min(d) / 2.0
    return min(r_wall, r_pair)

def _soft_min_objective(centers, beta):
    """
    Soft-min objective: -log(sum(exp(-beta * constraints))) / beta
    This approximates min(constraints) as beta -> inf.
    Constraints are: pairwise distances / 2, and wall distances.
    We want to maximize the minimum constraint, so minimize -soft_min.
    """
    c = centers
    # Wall constraints: 0.5 - |c_ik| for each coordinate
    wall = 0.5 - np.abs(c)  # shape (N, 3)
    
    # Pairwise constraints: dist_ij / 2
    d = pdist(c, 'euclidean') / 2.0
    
    # Combine all constraints
    all_cons = np.concatenate([wall.flatten(), d])
    
    # Soft min: -log(sum(exp(-beta * x))) / beta
    # For numerical stability, shift by max
    m = np.max(all_cons)
    shifted = all_cons - m
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min_val = -(m + log_sum / beta)
    
    # Gradient: d/dx_i of -soft_min = (1/beta) * sum_j exp(-beta*x_j) / sum_k exp(-beta*x_k) * (-beta) * dx_j/dx_i
    # = - sum_j w_j * dx_j/dx_i where w_j = exp(-beta*x_j) / sum_k exp(-beta*x_k)
    weights = np.exp(-beta * shifted)
    weights /= np.sum(weights)
    
    grad = np.zeros_like(c)
    
    # Wall gradient: d(wall_ik)/d(c_ik) = -sign(c_ik)
    wall_flat = wall.flatten()
    wall_grad_flat = -np.sign(c).flatten()
    grad_flat = np.zeros(N * 3)
    grad_flat[:N*3] -= weights[:N*3] * wall_grad_flat
    grad = grad_flat.reshape(N, 3)
    
    # Pairwise gradient: d(dist_ij/2)/d(c_i) = (c_i - c_j) / (2 * dist_ij)
    # Need to compute pairwise distances and accumulate
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    np.fill_diagonal(dist, 1.0)  # avoid div by zero
    
    # weights for pairwise: d is upper triangle of pdist
    # pdist returns dist in order: (0,1), (0,2), ..., (0,N-1), (1,2), ...
    # We need to map weight index to (i,j) pair
    n_pairs = len(d)
    pair_weights = weights[N*3:]
    
    # Accumulate gradient for pairwise
    pair_grad = np.zeros_like(c)
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            if idx >= n_pairs:
                break
            w = pair_weights[idx]
            dist_ij = dist[i, j]
            if dist_ij > 1e-12:
                dir_vec = diff[i, j] / dist_ij  # unit vector from j to i
                pair_grad[i] += w * dir_vec * 0.5
                pair_grad[j] -= w * dir_vec * 0.5
            idx += 1
    
    grad += pair_grad
    
    return -soft_min_val, grad

def _build_halton_init(n, dim=3):
    """Generate Halton sequence points in [0, 1]^dim."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = np.zeros((n, dim))
    for i in range(n):
        for j in range(dim):
            points[i, j] = halton(i + 1, j + 2)
    return points - 0.5

def _build_bcc_init(n, dim=3):
    """Generate BCC-like initialization."""
    # Use a grid with some perturbation
    n_per_dim = int(np.ceil(n ** (1.0/dim)))
    grid = np.meshgrid(*[np.linspace(-0.45, 0.45, n_per_dim)] * dim)
    grid = np.array(grid).reshape(dim, -1).T
    points = grid[:n]
    # Add small random perturbation
    points += np.random.normal(0, 0.01, points.shape)
    return np.clip(points, -0.49, 0.49)

def _build_grid_init(n, dim=3):
    """Generate simple grid initialization."""
    n_per_dim = int(np.ceil(n ** (1.0/dim)))
    grid = np.meshgrid(*[np.linspace(-0.45, 0.45, n_per_dim)] * dim)
    grid = np.array(grid).reshape(dim, -1).T
    points = grid[:n]
    return points

def _optimize_from_initial(centers, beta, max_iter, ftol, gtol):
    """Run L-BFGS-B optimization from given initial centers."""
    bounds = [(-0.5, 0.5)] * (N * 3)
    x0 = centers.flatten()
    
    def obj_func(x):
        c = x.reshape(N, 3)
        val, _ = _soft_min_objective(c, beta)
        return val
    
    def grad_func(x):
        c = x.reshape(N, 3)
        _, g = _soft_min_objective(c, beta)
        return g.flatten()
    
    try:
        res = minimize(
            obj_func,
            x0,
            jac=grad_func,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
        )
        return res.x.reshape(N, 3)
    except Exception:
        return centers

def _simulated_annealing(centers, beta, n_iter, start_temp, end_temp):
    """Simple simulated annealing to escape local minima."""
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    for i in range(n_iter):
        temp = start_temp * (end_temp / start_temp) ** (i / n_iter)
        
        # Random perturbation
        perturbation = np.random.normal(0, temp, current.shape)
        proposed = np.clip(current + perturbation, -0.49, 0.49)
        
        proposed_r = _compute_radius(proposed)
        
        if proposed_r > current_r:
            current = proposed
            if proposed_r > best_r:
                best = proposed.copy()
                best_r = proposed_r
        else:
            # Accept with probability exp((proposed_r - current_r) / temp)
            delta = proposed_r - current_r
            if temp > 1e-10 and np.random.random() < np.exp(delta / temp):
                current = proposed
    
    return best

def _polish_high_beta(centers, beta, max_iter):
    """Final high-precision polish with high beta and tight tolerances."""
    return _optimize_from_initial(centers, beta, max_iter, 1e-14, 1e-10)

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Multiple initializations
    inits = []
    inits.append(_build_halton_init(N))
    inits.append(_build_bcc_init(N))
    inits.append(_build_grid_init(N))
    
    best_centers = None
    best_r = 0.0
    
    # Phase 1: Optimize from each initialization with moderate beta
    beta_start = 100.0
    for init in inits:
        if time_left() < 30:
            break
        
        # Optimize with moderate beta
        optimized = _optimize_from_initial(init, beta_start, 200, 1e-10, 1e-8)
        r = _compute_radius(optimized)
        
        if r > best_r:
            best_r = r
            best_centers = optimized.copy()
    
    # Phase 2: Simulated annealing to escape local minima
    if best_centers is not None and time_left() > 40:
        # Run SA with decreasing temperature
        sa_result = _simulated_annealing(
            best_centers,
            beta_start,
            n_iter=500,
            start_temp=0.05,
            end_temp=0.001
        )
        r_sa = _compute_radius(sa_result)
        
        if r_sa > best_r:
            best_r = r_sa
            best_centers = sa_result.copy()
    
    # Phase 3: High-precision polish with increasing beta
    if best_centers is not None and time_left() > 20:
        for beta in [500, 2000, 10000]:
            if time_left() < 15:
                break
            polished = _polish_high_beta(best_centers, beta, 300)
            r_polished = _compute_radius(polished)
            
            if r_polished > best_r:
                best_r = r_polished
                best_centers = polished.copy()
    
    # Final check
    if best_centers is None:
        best_centers = _build_halton_init(N)
    
    return best_centers
