"""Search-based packer for 100 equal spheres in a unit cube."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 100

def _score(points):
    """Return the maximal radius for the given set of centres."""
    if len(points) != N:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Inside the cube [-0.5, 0.5]^3
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    r_pair = min_dist / 2.0
    return min(r_pair, r_margin)

def _smooth_energy_and_grad(x, beta=50.0):
    """
    Smooth surrogate for the packing problem using log-sum-exp soft-min.
    Returns (energy, gradient) where energy approximates -radius.
    
    The radius is min over all constraints c_k of c_k, where:
    - For each pair (i,j): c = dist(i,j)/2
    - For each point i and axis k: c = 0.5 - |x_ik|
    
    We approximate min(c_k) via -log-sum-exp(-beta * c_k) / beta
    and minimize this (i.e., maximize the soft-min).
    """
    pts = x.reshape(N, 3)
    
    # Compute all pairwise distances (upper triangle only to save memory)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = (diff ** 2).sum(axis=2)
    # Set diagonal to a large value
    np.fill_diagonal(dists_sq, np.inf)
    # Only use upper triangle
    iu = np.triu_indices(N, k=1)
    dists = np.sqrt(dists_sq[iu])
    
    # Margins: distance from each point to the nearest face
    # For point i, margin = min over axes of (0.5 - |x_ik|)
    # Actually we need min over all (i, axis) pairs
    margins = 0.5 - np.abs(pts)  # shape (N, 3)
    margins_flat = margins.flatten()  # shape (3N,)
    
    # Combine all constraints: pairwise distances/2 and margins
    # Pairwise: dists/2, shape (N*(N-1)/2,)
    pair_constraints = dists / 2.0
    # Wall constraints: margins_flat, shape (3N,)
    all_constraints = np.concatenate([pair_constraints, margins_flat])
    
    # Soft-min: -log(sum(exp(-beta * c))) / beta
    # For numerical stability, shift by max
    c_max = all_constraints.max()
    shifted = -beta * (all_constraints - c_max)
    # Clip for numerical stability
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = c_max - log_sum / beta
    
    energy = -soft_min  # We minimize this
    
    # Gradient computation
    # d(soft_min)/dx = sum_k [exp(-beta*c_k) / sum_j exp(-beta*c_j)] * (-beta * dc_k/dx)
    # = -beta * sum_k [p_k * dc_k/dx] where p_k = exp(-beta*c_k) / sum exp(-beta*c_j)
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)  # shape (len(all_constraints),)
    
    # Gradient of soft_min w.r.t. x
    grad = np.zeros_like(pts)
    
    # Pairwise contribution
    # For pair (i,j), c = dist(i,j)/2
    # dc/dx_i = (x_i - x_j) / (2 * dist)
    # dc/dx_j = (x_j - x_i) / (2 * dist)
    pair_weights = weights[:len(pair_constraints)]
    
    # Rebuild the pair index mapping
    i_idx, j_idx = iu
    for k in range(len(pair_constraints)):
        i, j = i_idx[k], j_idx[k]
        d = dists[k]
        if d < 1e-12:
            continue
        w = pair_weights[k]
        # dc/dx_i = (x_i - x_j) / (2*d), dc/dx_j = (x_j - x_i) / (2*d)
        diff_vec = (pts[i] - pts[j]) / (2.0 * d)
        grad[i] += w * diff_vec
        grad[j] -= w * diff_vec
    
    # Wall contribution
    # For point i, axis a: c = 0.5 - |x_ia|
    # dc/dx_ia = -sign(x_ia) if x_ia != 0
    wall_weights = weights[len(pair_constraints):]
    wall_weights_reshaped = wall_weights.reshape(N, 3)
    signs = np.sign(pts)
    # Handle zero: gradient is subgradient, use 0
    signs[pts == 0] = 0.0
    grad -= wall_weights_reshaped * signs  # dc/dx = -sign(x), and we have -beta factor absorbed in weights
    
    # Total gradient of energy = -gradient of soft_min
    grad_energy = -grad
    
    return energy, grad_energy.flatten()

def _random_grid_initial(m=4):
    """Generate a simple grid-based initialisation."""
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.5 + 0.1, 0.5 - 0.1, nx)
    ys = np.linspace(-0.5 + 0.1, 0.5 - 0.1, ny)
    zs = np.linspace(-0.5 + 0.125, 0.5 - 0.125, nz)
    points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return points

def _local_optimize_smooth(points, maxiter=100, beta=50.0):
    """Locally optimise using smooth surrogate with analytical gradient."""
    
    def obj_and_grad(x):
        e, g = _smooth_energy_and_grad(x, beta=beta)
        return e, g
    
    x0 = points.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        result = minimize(obj_and_grad, x0, method='L-BFGS-B', jac=True,
                          bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-6})
        if result.fun < obj_and_grad(x0)[0]:
            return result.x.reshape(N, 3)
        else:
            return points
    except Exception:
        return points

def _local_optimize(points, method='L-BFGS-B'):
    """Locally optimise the radius using gradient-based methods with smooth surrogate."""
    return _local_optimize_smooth(points, maxiter=80, beta=50.0)

def _simulated_annealing(points, t_max=5.0, t_min=0.01, alpha=0.99, max_iter=500):
    """Simple Simulated Annealing to escape local minima."""
    current_points = points.copy()
    current_score = _score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    t = t_max
    for i in range(max_iter):
        if t < t_min:
            break
        idx = random.randint(0, N - 1)
        move_size = 0.05 * t
        delta = np.random.normal(0, move_size, 3)
        new_points = current_points.copy()
        new_points[idx] += delta
        new_points[idx] = np.clip(new_points[idx], -0.5, 0.5)
        
        new_score = _score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_points = current_points.copy()
                best_score = current_score
        else:
            delta_e = current_score - new_score
            if delta_e < 0:
                accept = True
            else:
                accept = random.random() < np.exp(-delta_e / t)
            
            if accept:
                current_points = new_points
                current_score = new_score
        
        t *= alpha
    
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial solution from grid
    points = _random_grid_initial()
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Phase 1: Local optimisation with smooth surrogate
    if time.time() < deadline - 30:
        opt_points = _local_optimize_smooth(points, maxiter=100, beta=50.0)
        opt_score = _score(opt_points)
        if opt_score > best_score:
            best_points = opt_points
            best_score = opt_score
    
    # Phase 2: Multi-start with random perturbations and local optimisation
    random.seed(42)
    np.random.seed(42)
    
    while time.time() < deadline - 15:
        if random.random() < 0.5:
            new_start = best_points + np.random.normal(0, 0.04, (N, 3))
            new_start = np.clip(new_start, -0.5, 0.5)
        else:
            new_start = _random_grid_initial() + np.random.normal(0, 0.02, (N, 3))
            new_start = np.clip(new_start, -0.5, 0.5)
        
        # Local optimise with smooth surrogate
        opt_points = _local_optimize_smooth(new_start, maxiter=60, beta=50.0)
        opt_score = _score(opt_points)
        
        if opt_score > best_score:
            best_points = opt_points
            best_score = opt_score
        
        # Occasionally try SA
        if random.random() < 0.25 and time.time() < deadline - 25:
            sa_points = _simulated_annealing(best_points, max_iter=80)
            sa_score = _score(sa_points)
            if sa_score > best_score:
                best_points = sa_points
                best_score = sa_score
    
    # Final polish with higher beta (sharper approximation to true min)
    if time.time() < deadline - 5:
        final_points = _local_optimize_smooth(best_points, maxiter=50, beta=200.0)
        final_score = _score(final_points)
        if final_score > best_score:
            best_points = final_points
            best_score = final_score
    
    return best_points

# EVOLVE-BLOCK-END
