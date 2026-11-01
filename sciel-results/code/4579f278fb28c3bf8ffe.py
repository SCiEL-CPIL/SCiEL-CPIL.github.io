"""Packing 50 equal spheres in a unit cube via structured optimization."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import KDTree

N = 50
DEADLINE = 145.0  # seconds

# EVOLVE-BLOCK-START
def _compute_radius(centers):
    """Compute the maximum common radius for a given set of centers."""
    if not np.all(np.isfinite(centers)):
        return -1.0
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    tree = KDTree(centers)
    pairs = tree.query_pairs(1.0)  # all pairs within 1.0
    if len(pairs) == 0:
        min_pair_dist = np.inf
    else:
        min_pair_dist = np.min([np.linalg.norm(centers[i] - centers[j]) for i, j in pairs])
    radii = np.minimum(wall_dists, min_pair_dist / 2.0)
    return np.min(radii)

def _compute_radius_full(centers):
    """Full O(N^2) computation for accuracy."""
    if not np.all(np.isfinite(centers)):
        return -1.0
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    radii = np.minimum(wall_dists, min_pair_dist / 2.0)
    return np.min(radii)

def _initial_grid():
    """Generate initial grid-based positions."""
    # 4x4x4 = 64 grid points, select 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    rng = np.random.default_rng(42)
    indices = rng.choice(len(grid), size=N, replace=False)
    return grid[indices].copy()

def _initial_fcc_like():
    """FCC-inspired initial configuration."""
    # FCC has 8 points per unit cell. Use a 3x3x3 supercell = 216 points, select 50.
    # Scale to fit in [-0.5, 0.5]
    a = 1.0  # lattice constant
    points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # FCC basis
                basis = np.array([
                    [0, 0, 0],
                    [0.5, 0.5, 0],
                    [0.5, 0, 0.5],
                    [0, 0.5, 0.5]
                ]) * a
                for b in basis:
                    pts = np.array([i, j, k]) * a + b
                    points.append(pts)
    points = np.array(points)
    # Center and scale
    points -= np.mean(points, axis=0)
    max_ext = np.max(np.abs(points))
    points *= (0.45 / max_ext)  # keep inside with margin
    rng = np.random.default_rng(42)
    indices = rng.choice(len(points), size=N, replace=False)
    return points[indices].copy()

def _initial_random():
    """Random initial configuration."""
    rng = np.random.default_rng(42)
    return rng.uniform(-0.45, 0.45, size=(N, 3))

def _softmin_radius(centers, beta):
    """Smooth approximation of the minimum radius."""
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_half_dists = dists / 2.0
    np.fill_diagonal(pair_half_dists, np.inf)
    
    M = np.zeros((N, N + 1))
    M[:, 0] = wall_dists
    M[:, 1:] = pair_half_dists
    
    x = -beta * M
    x_max = np.max(x, axis=1, keepdims=True)
    x_max = np.where(x_max == -np.inf, 0, x_max)
    lse_i = np.log(np.sum(np.exp(x - x_max), axis=1)) + np.squeeze(x_max, axis=1)
    r_i_smooth = -1.0 / beta * lse_i
    
    y = -beta * r_i_smooth
    y_max = np.max(y, keepdims=True)
    y_max = np.where(y_max == -np.inf, 0, y_max)
    lse_r = np.log(np.sum(np.exp(y - y_max))) + np.squeeze(y_max)
    r_smooth = -1.0 / beta * lse_r
    
    return r_smooth

def _softmin_grad(centers, beta):
    """Gradient of the softmin radius."""
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_half_dists = dists / 2.0
    np.fill_diagonal(pair_half_dists, np.inf)
    
    M = np.zeros((N, N + 1))
    M[:, 0] = wall_dists
    M[:, 1:] = pair_half_dists
    
    x = -beta * M
    x_max = np.max(x, axis=1, keepdims=True)
    x_max = np.where(x_max == -np.inf, 0, x_max)
    exp_x = np.exp(x - x_max)
    V_k = np.sum(exp_x, axis=1, keepdims=True)
    v = exp_x / V_k
    
    r_k = -1.0 / beta * (np.log(np.squeeze(V_k, axis=1)) + np.squeeze(x_max, axis=1))
    
    y = -beta * r_k
    y_max = np.max(y, keepdims=True)
    y_max = np.where(y_max == -np.inf, 0, y_max)
    exp_y = np.exp(y - y_max)
    U = np.sum(exp_y, keepdims=True)
    w = exp_y / U
    
    grad_d = -w * v
    grad_flat = np.zeros_like(centers.flatten())
    
    abs_centers = np.abs(centers)
    max_idx = np.argmax(abs_centers, axis=1)
    
    for k in range(N):
        g_wall = grad_d[k, 0]
        if g_wall != 0:
            dim = max_idx[k]
            c_val = centers[k, dim]
            grad_flat[k * 3 + dim] += g_wall * (-np.sign(c_val))
    
    for k in range(N):
        for j in range(1, N + 1):
            l = j - 1
            if k == l:
                continue
            g_pair = grad_d[k, j]
            if g_pair != 0:
                diff_vec = centers[k] - centers[l]
                dist_val = np.linalg.norm(diff_vec)
                if dist_val > 1e-9:
                    grad_vec = 0.5 * diff_vec / dist_val
                    grad_flat[k * 3 : (k + 1) * 3] += g_pair * grad_vec
                    grad_flat[l * 3 : (l + 1) * 3] += g_pair * (-grad_vec)
    
    return grad_flat

def _polish_lbfgs(centers, beta, max_iter, deadline):
    """Polish using L-BFGS-B with softmin objective."""
    if time.time() > deadline - 5:
        return centers
    
    flat_centers = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def obj(x):
        return -_softmin_radius(x.reshape(N, 3), beta)
    
    def grad(x):
        return _softmin_grad(x.reshape(N, 3), beta)
    
    try:
        res = minimize(
            obj,
            flat_centers,
            jac=grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        new_centers = res.x.reshape((N, 3))
        if np.all(np.isfinite(new_centers)):
            return new_centers
    except Exception:
        pass
    return centers

def _simulated_annealing(centers, t_max, t_min, alpha, deadline):
    """Simple simulated annealing to escape local minima."""
    if time.time() > deadline - 10:
        return centers
    
    current = centers.copy()
    current_radius = _compute_radius(current)
    best = current.copy()
    best_radius = current_radius
    T = t_max
    rng = np.random.default_rng(42)
    
    while T > t_min and time.time() < deadline - 10:
        # Propose a move
        i = rng.integers(0, N)
        delta = rng.normal(0, 0.01, size=3)
        proposed = current.copy()
        proposed[i] += delta
        proposed[i] = np.clip(proposed[i], -0.5, 0.5)
        
        proposed_radius = _compute_radius(proposed)
        delta_r = proposed_radius - current_radius
        
        if delta_r > 0 or rng.random() < np.exp(delta_r / T):
            current = proposed
            current_radius = proposed_radius
            if current_radius > best_radius:
                best = current.copy()
                best_radius = current_radius
        
        T *= alpha
    
    return best

def _local_refine(centers, deadline):
    """Local refinement using coordinate descent."""
    if time.time() > deadline - 5:
        return centers
    
    current = centers.copy()
    current_radius = _compute_radius(current)
    best = current.copy()
    best_radius = current_radius
    
    for iteration in range(100):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(N):
            # Try moving point i in each direction
            for dim in range(3):
                for sign in [-1, 1]:
                    proposed = current.copy()
                    step = 0.005 * sign
                    proposed[i, dim] += step
                    if proposed[i, dim] < -0.5 or proposed[i, dim] > 0.5:
                        continue
                    proposed_radius = _compute_radius(proposed)
                    if proposed_radius > current_radius:
                        current = proposed
                        current_radius = proposed_radius
                        improved = True
                        if current_radius > best_radius:
                            best = current.copy()
                            best_radius = current_radius
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Generate multiple initial configurations
    inits = [
        _initial_grid(),
        _initial_fcc_like(),
        _initial_random()
    ]
    
    best_centers = inits[0].copy()
    best_radius = _compute_radius(best_centers)
    
    # Evaluate all initial configurations
    for init in inits[1:]:
        r = _compute_radius(init)
        if r > best_radius:
            best_radius = r
            best_centers = init.copy()
    
    # Phase 1: Simulated annealing from the best initial
    if time.time() < deadline - 30:
        sa_centers = _simulated_annealing(best_centers, t_max=0.05, t_min=0.001, alpha=0.95, deadline=deadline)
        r = _compute_radius(sa_centers)
        if r > best_radius:
            best_radius = r
            best_centers = sa_centers.copy()
    
    # Phase 2: Local refinement
    if time.time() < deadline - 20:
        refined = _local_refine(best_centers, deadline)
        r = _compute_radius(refined)
        if r > best_radius:
            best_radius = r
            best_centers = refined.copy()
    
    # Phase 3: L-BFGS polish with increasing beta
    betas = [50, 100, 200, 500, 1000]
    for beta in betas:
        if time.time() > deadline - 10:
            break
        polished = _polish_lbfgs(best_centers, beta, max_iter=50, deadline=deadline)
        r = _compute_radius(polished)
        if r > best_radius:
            best_radius = r
            best_centers = polished.copy()
    
    # Final validation
    final_radius = _compute_radius_full(best_centers)
    if final_radius < 0 or not np.all(np.isfinite(best_centers)):
        # Fallback
        best_centers = _initial_grid()
    
    return best_centers
# EVOLVE-BLOCK-END
