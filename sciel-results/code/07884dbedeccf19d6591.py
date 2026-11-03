"""
Improved sphere packing for 50 equal spheres in a unit cube using SDP relaxation.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_radius(pts):
    """Compute the maximum common radius for given sphere centers."""
    n = len(pts)
    # Pairwise distances (vectorized)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    # Wall distances
    wall_dists = 0.5 - np.abs(pts)
    min_wall = np.min(wall_dists)
    return min(min_dist / 2.0, min_wall)

def _sdp_init():
    """
    Formulate and solve an SDP relaxation to get a high-quality initial configuration.
    We maximize the sum of squared pairwise distances (trace of Gram matrix) 
    subject to the Gram matrix being PSD and points staying within bounds.
    
    Since scipy doesn't have a native SDP solver, we approximate the SDP structure 
    using a smooth surrogate that penalizes negative eigenvalues of the Gram matrix 
    of the lifted features, or more practically, we use the SDP insight to define 
    a better objective: maximizing the minimum distance is hard, but maximizing 
    the sum of distances (repulsion) in a convexified space is easier.
    
    However, a true SDP relaxation for sphere packing usually involves lifting 
    x_i to z_i = [1, x_i, x_i^T x_i]. 
    Here we use a heuristic inspired by SDP: 
    1. Lift points to a higher dimension implicitly by optimizing a potential 
       that encourages uniform distribution (like Thomson problem but in 3D cube).
    2. Use L-BFGS-B to minimize a smooth approximation of the negative radius.
    """
    # We will use a multi-start strategy with a very smooth objective 
    # that mimics the SDP relaxation's tendency to spread points evenly.
    
    # Initial guess: Jittered grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    np.random.seed(42)
    grid += np.random.randn(*grid.shape) * 0.02
    grid = np.clip(grid, -0.49, 0.49)
    pts = grid[:N].copy()
    
    # Objective: Minimize negative soft-min radius
    # Soft min of distances: -1/beta * log(sum(exp(-beta * d)))
    beta = 100.0
    
    def objective(x):
        p = x.reshape(N, 3)
        # Pairwise distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Soft min pair dist
        min_pair = -np.log(np.sum(np.exp(-beta * dists))) / beta
        
        # Wall distances
        wall_dists = 0.5 - np.abs(p)
        min_wall = -np.log(np.sum(np.exp(-beta * wall_dists))) / beta
        
        # We want to maximize min(min_pair, min_wall)
        # Equivalent to minimizing -min(min_pair, min_wall)
        # Use soft min of the two minima to keep it smooth
        val = min(min_pair, min_wall)
        return -val
    
    def gradient(x):
        p = x.reshape(N, 3)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Gradient of soft min pair dist
        # f = -1/beta log(sum(exp(-beta d_ij)))
        # df/dp_i = sum_j (exp(-beta d_ij) * beta * (p_i - p_j) / d_ij) / sum_k exp(-beta d_ik)
        # Note: derivative of d_ij w.r.t p_i is (p_i - p_j)/d_ij
        # exp terms
        exp_dists = np.exp(-beta * dists)
        np.fill_diagonal(exp_dists, 0) # Exclude self
        sum_exp = np.sum(exp_dists, axis=1, keepdims=True)
        
        # Weight matrix W_ij = exp(-beta d_ij) / sum_k exp(-beta d_ik)
        W = exp_dists / sum_exp
        
        # grad_pair_i = sum_j W_ij * (p_i - p_j) / d_ij
        # Vectorized: 
        # (p_i - p_j) / d_ij is unit vector u_ij
        u = diff / dists[:, :, np.newaxis]
        np.fill_diagonal(u, 0) # Avoid division by zero artifacts on diagonal
        
        # grad_pair_i = sum_j W_ij * u_ij
        grad_pair = np.sum(W[:, :, np.newaxis] * u, axis=1)
        
        # Wall gradient
        wall_dists = 0.5 - np.abs(p)
        exp_walls = np.exp(-beta * wall_dists)
        sum_exp_w = np.sum(exp_walls, axis=1, keepdims=True)
        W_w = exp_walls / sum_exp_w
        
        # d wall_dist / d p_ik = -sign(p_ik)
        signs = np.sign(p)
        grad_wall = -W_w * signs
        
        # Combine
        # We are minimizing -min(pair, wall). 
        # If pair < wall, grad is -grad_pair. Else -grad_wall.
        # To keep it smooth, we can just sum them with weights or use the min.
        # Simple approach: if min_pair < min_wall, use grad_pair, else grad_wall.
        # But this is non-smooth. 
        # Let's use a weighted sum based on which is closer to the min.
        # Or simply return the gradient of the active constraint.
        
        # For L-BFGS, a consistent gradient is needed. 
        # Let's use the gradient of the soft-min of the two values directly?
        # Actually, objective is -min(A, B). 
        # Let's approximate by -0.5*(A+B) + 0.5*|A-B|? No, that's complex.
        
        # Let's stick to the simpler: 
        # If min_pair < min_wall: return -grad_pair
        # Else: return -grad_wall
        # This is subgradient, L-BFGS might struggle.
        
        # Better: Minimize A + B - min(A,B) ? No.
        # Let's minimize A * B / (A+B) ? Harmonic mean? 
        # Let's just minimize the sum of the two soft mins? 
        # No, we want the bottleneck.
        
        # Let's use a smooth minimum: 
        # S = -1/gamma log(exp(-gamma A) + exp(-gamma B))
        # dS/dA = exp(-gamma A) / (exp(-gamma A) + exp(-gamma B))
        # dS/dB = exp(-gamma B) / (exp(-gamma A) + exp(-gamma B))
        
        gamma = 20.0
        A = min_pair
        B = min_wall
        
        wA = np.exp(-gamma * A) / (np.exp(-gamma * A) + np.exp(-gamma * B))
        wB = np.exp(-gamma * B) / (np.exp(-gamma * A) + np.exp(-gamma * B))
        
        # d objective / dx = - (wA * dA/dx + wB * dB/dx)
        # dA/dx = grad_pair (from soft min of pair dists)
        # dB/dx = grad_wall (from soft min of wall dists)
        
        grad = -(wA * grad_pair + wB * grad_wall)
        return grad.flatten()
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Optimize
    res = minimize(objective, pts.flatten(), method='L-BFGS-B', jac=gradient, bounds=bounds, 
                   options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    
    best_pts = res.x.reshape(N, 3)
    best_pts = np.clip(best_pts, -0.4999999, 0.4999999)
    
    return best_pts

def _local_optimize(pts, time_limit):
    """Local optimization using gradient descent with adaptive step."""
    start_time = time.time()
    n = len(pts)
    pts = pts.copy()
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    
    # Phase 1: Gradient-based polish
    beta = 200.0
    step_size = 0.01
    
    def obj(x):
        p = x.reshape(N, 3)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = -np.log(np.sum(np.exp(-beta * dists))) / beta
        wall_dists = 0.5 - np.abs(p)
        min_wall = -np.log(np.sum(np.exp(-beta * wall_dists))) / beta
        return -min(min_pair, min_wall)
        
    def grad(x):
        p = x.reshape(N, 3)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        exp_dists = np.exp(-beta * dists)
        np.fill_diagonal(exp_dists, 0)
        sum_exp = np.sum(exp_dists, axis=1, keepdims=True)
        W = exp_dists / sum_exp
        u = diff / dists[:, :, np.newaxis]
        np.fill_diagonal(u, 0)
        grad_pair = np.sum(W[:, :, np.newaxis] * u, axis=1)
        
        wall_dists = 0.5 - np.abs(p)
        exp_walls = np.exp(-beta * wall_dists)
        sum_exp_w = np.sum(exp_walls, axis=1, keepdims=True)
        W_w = exp_walls / sum_exp_w
        signs = np.sign(p)
        grad_wall = -W_w * signs
        
        gamma = 20.0
        A = min_pair
        B = min_wall
        wA = np.exp(-gamma * A) / (np.exp(-gamma * A) + np.exp(-gamma * B))
        wB = np.exp(-gamma * B) / (np.exp(-gamma * A) + np.exp(-gamma * B))
        
        return -(wA * grad_pair + wB * grad_wall).flatten()

    res = minimize(obj, pts.flatten(), method='L-BFGS-B', jac=grad, 
                   bounds=[(-0.5, 0.5)] * (N * 3),
                   options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
    
    if res.success or res.x is not None:
        trial_pts = res.x.reshape(N, 3)
        trial_pts = np.clip(trial_pts, -0.4999999, 0.4999999)
        trial_r = _compute_radius(trial_pts)
        if trial_r > best_r:
            best_r = trial_r
            best_pts = trial_pts

    # Phase 2: Coordinate descent for final precision
    elapsed = time.time() - start_time
    if time_limit - elapsed > 5.0:
        step_cd = 0.005
        for _ in range(10): # Fixed number of passes if time allows
            if time.time() - start_time > time_limit - 5.0:
                break
            improved = False
            for i in range(n):
                for dim in range(3):
                    for direction in [1, -1]:
                        trial = best_pts.copy()
                        trial[i][dim] += direction * step_cd
                        if abs(trial[i][dim]) > 0.5:
                            continue
                        r = _compute_radius(trial)
                        if r > best_r + 1e-12:
                            best_pts = trial
                            best_r = r
                            improved = True
            if not improved:
                step_cd *= 0.5
                if step_cd < 1e-5:
                    break

    return best_pts, best_r

def construct():
    start_time = time.time()
    time_limit = 120.0  # Safety margin
    
    # 1. Get initial guess from SDP-inspired optimization
    pts_init = _sdp_init()
    
    # 2. Local optimization
    best_pts, best_r = _local_optimize(pts_init, time_limit)
    
    # 3. Ensure validity
    best_pts = np.clip(best_pts, -0.4999999, 0.4999999)
    
    return best_pts
