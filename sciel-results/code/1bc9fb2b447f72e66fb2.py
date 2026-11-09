"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

DEADLINE = 145.0  # safety margin before 150s limit

def compute_radius(points):
    """Compute the maximum common radius for a set of points."""
    points = np.asarray(points)
    if len(points) == 0:
        return 0.0
    # Distance to walls
    wall_dists = np.min(np.stack([
        points[:, 0] + 0.5,
        0.5 - points[:, 0],
        points[:, 1] + 0.5,
        0.5 - points[:, 1],
        points[:, 2] + 0.5,
        0.5 - points[:, 2],
    ], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    # Pairwise distances
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_wall, min_pair)

def soft_min_objective_and_grad(points, alpha):
    """
    Compute soft-min objective and its gradient.
    We want to maximize min of all distances, so minimize -soft_min.
    soft_min = -alpha * log(sum(exp(-d_i / alpha)))
    Objective = -soft_min = alpha * log(sum(exp(-d_i / alpha)))
    Gradient of objective w.r.t. points.
    """
    points = np.asarray(points).reshape(-1, 3)
    n = len(points)
    
    # Pairwise distances (lower triangle only)
    tri_i, tri_j = np.tril_indices(n, -1)
    diff = points[tri_i] - points[tri_j]  # shape (m, 3)
    dists = np.linalg.norm(diff, axis=1)  # shape (m,)
    
    # Wall distances
    wall_dists = np.concatenate([
        points[:, 0] + 0.5,
        0.5 - points[:, 0],
        points[:, 1] + 0.5,
        0.5 - points[:, 1],
        points[:, 2] + 0.5,
        0.5 - points[:, 2],
    ])  # shape (6n,)
    
    # Combine all distances
    all_dists = np.concatenate([dists, wall_dists])
    
    # Soft min computation with log-sum-exp trick
    neg_dists = -all_dists / alpha
    neg_max = np.max(neg_dists)
    log_sum = neg_max + np.log(np.sum(np.exp(neg_dists - neg_max)))
    soft_min = -alpha * log_sum
    objective = -soft_min  # = alpha * log_sum
    
    # Gradient
    # d(objective)/d(all_dists) = alpha * exp(-d_i/alpha) / sum(exp(-d_j/alpha))
    # = alpha * exp(neg_dists_i - neg_max) / sum(exp(neg_dists_j - neg_max))
    weights = alpha * np.exp(neg_dists - neg_max) / np.sum(np.exp(neg_dists - neg_max))
    # weights shape: (m + 6n,)
    
    pair_weights = weights[:len(dists)]
    wall_weights = weights[len(dists):]
    
    # Gradient w.r.t. points
    grad = np.zeros_like(points)
    
    # Pairwise contributions
    # For each pair (i,j) with weight w_ij:
    # d(objective)/d(p_i) += w_ij * (p_i - p_j) / d_ij  [since d/dp_i ||p_i-p_j|| = (p_i-p_j)/d_ij]
    # d(objective)/d(p_j) += w_ij * (p_j - p_i) / d_ij
    for idx in range(len(dists)):
        i = tri_i[idx]
        j = tri_j[idx]
        w = pair_weights[idx]
        d = dists[idx]
        if d > 1e-12:
            diff_vec = diff[idx] / d  # unit vector from j to i
            grad[i] += w * diff_vec
            grad[j] -= w * diff_vec
    
    # Wall contributions
    # Wall distances: [p0+0.5, 0.5-p0, p1+0.5, 0.5-p1, p2+0.5, 0.5-p2] for each point
    # d/dp_x of (p_x + 0.5) = 1, d/dp_x of (0.5 - p_x) = -1
    for k in range(n):
        # x-component: wall_dists[6k] = p[k,0]+0.5 (weight wall_weights[6k])
        #               wall_dists[6k+1] = 0.5-p[k,0] (weight wall_weights[6k+1])
        grad[k, 0] += wall_weights[6*k] * 1.0 + wall_weights[6*k+1] * (-1.0)
        grad[k, 1] += wall_weights[6*k+2] * 1.0 + wall_weights[6*k+3] * (-1.0)
        grad[k, 2] += wall_weights[6*k+4] * 1.0 + wall_weights[6*k+5] * (-1.0)
    
    return objective, grad

def construct():
    start_time = time.time()
    
    best_points = None
    best_radius = -np.inf
    
    def elapsed():
        return time.time() - start_time
    
    def time_remaining():
        return DEADLINE - elapsed()
    
    # Multiple initialization strategies
    def grid_init(m=None):
        if m is None:
            m = int(np.ceil(N ** (1.0/3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()
    
    def random_init(scale=0.4):
        return (np.random.rand(N, 3) - 0.5) * scale
    
    def fcc_init():
        """Face-centered cubic lattice initialization."""
        # FCC lattice points in a cube
        a = 1.0 / 4.0  # lattice parameter
        points_list = []
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    x = -0.5 + i * a + 0.5 * a
                    y = -0.5 + j * a + 0.5 * a
                    z = -0.5 + k * a + 0.5 * a
                    # Corner and face-center points
                    points_list.append([x, y, z])
                    # Face centers
                    if i < 4:
                        points_list.append([x + a/2, y, z])
                        points_list.append([x, y + a/2, z])
                        points_list.append([x, y, z + a/2])
        points = np.array(points_list[:N])
        # Center and scale
        points = (points - points.mean(axis=0)) * 0.8
        return points
    
    # List of initializations to try
    inits = [
        grid_init(),
        random_init(0.4),
        random_init(0.3),
        fcc_init(),
    ]
    
    # For each initialization, run L-BFGS-B optimization with decreasing alpha
    for init_idx, init_pts in enumerate(inits):
        if time_remaining() < 15:
            break
        
        # Clip to valid region
        init_pts = np.clip(init_pts, -0.49, 0.49)
        
        # Multi-stage alpha: start smooth, end sharp
        alphas = [10.0, 5.0, 2.0, 1.0, 0.5, 0.3]
        
        current = init_pts.copy()
        
        for alpha in alphas:
            if time_remaining() < 10:
                break
            
            # Define objective and gradient for this alpha
            def obj_and_grad(flat_pts, a=alpha):
                pts = flat_pts.reshape(-1, 3)
                val, g = soft_min_objective_and_grad(pts, a)
                return val, g.flatten()
            
            # L-BFGS-B optimization
            try:
                res = minimize(
                    obj_and_grad,
                    current.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * 3 * N,
                    options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-10, 'maxls': 50},
                )
                new_pts = res.x.reshape(-1, 3)
                
                # Check if improved
                r = compute_radius(new_pts)
                if r > best_radius:
                    best_radius = r
                    best_points = new_pts.copy()
                
                current = new_pts.copy()
                
            except Exception:
                # If optimization fails, continue with current points
                pass
        
        # After all alphas, check final radius
        if best_points is None:
            r = compute_radius(current)
            if r > best_radius:
                best_radius = r
                best_points = current.copy()
    
    # Final polish: if we have time, do a very fine optimization on best points
    if best_points is not None and time_remaining() > 20:
        for alpha in [0.2, 0.1, 0.05]:
            if time_remaining() < 10:
                break
            def obj_and_grad(flat_pts, a=alpha):
                pts = flat_pts.reshape(-1, 3)
                val, g = soft_min_objective_and_grad(pts, a)
                return val, g.flatten()
            
            try:
                res = minimize(
                    obj_and_grad,
                    best_points.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * 3 * N,
                    options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-12, 'maxls': 50},
                )
                new_pts = res.x.reshape(-1, 3)
                r = compute_radius(new_pts)
                if r > best_radius:
                    best_radius = r
                    best_points = new_pts.copy()
            except Exception:
                pass
    
    # Ensure validity
    if best_points is None:
        best_points = grid_init()
    
    best_points = np.clip(best_points, -0.5, 0.5)
    
    return best_points
# EVOLVE-BLOCK-END
