import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140  # seconds, safety margin for 150s limit

def _compute_radius(c):
    """Compute the packing radius for centres c of shape (N,3)."""
    if c.shape != (N, 3) or not np.all(np.isfinite(c)):
        return 0.0
    # Wall distance
    wall = 0.5 - np.max(np.abs(c), axis=1)
    min_wall = np.min(wall)
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(min_pair / 2.0, min_wall)

def _softmin_radius(c, beta):
    """
    Smooth approximation of the min radius.
    Uses log-sum-exp trick for numerical stability.
    beta controls smoothness (higher = closer to true min).
    """
    # Wall constraints: 0.5 - |c_ik| for each i,k
    # We need min over all i,k of (0.5 - |c_ik|)
    # And min over all i<j of |c_i - c_j| / 2
    
    # For the softmin, we combine all constraints.
    # Let's compute the "violations" or rather the distances.
    # r = min( min_wall, min_pair/2 )
    
    # To make it differentiable, we use -softmin(-x) = -(-1/beta * log(sum(exp(-beta*x))))
    # Actually, softmin(x, beta) = -1/beta * log(sum(exp(-beta*x_i)))
    # We want to maximize r, so we minimize -r.
    # -r = max( -min_wall, -min_pair/2 ) = max( max_i,k (|c_ik| - 0.5), max_i<j (0.5 - |c_i-c_j|/2) )
    # This is getting complicated. Let's just use the softmin on the radius components directly.
    
    # Simpler approach: compute all candidate radii, then softmin them.
    # Candidates:
    # 1. For each i,k: 0.5 - |c_ik|  (50*3 = 150 values)
    # 2. For each i<j: |c_i - c_j| / 2  (50*49/2 = 1225 values)
    
    wall_dists = 0.5 - np.abs(c)  # shape (50, 3)
    wall_flat = wall_dists.flatten()  # shape (150,)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Get upper triangle
    iu = np.triu_indices(N, 1)
    pair_dists = dists[iu] / 2.0  # shape (1225,)
    
    # Combine all
    all_dists = np.concatenate([wall_flat, pair_dists])
    
    # Softmin
    # To avoid overflow, subtract max
    m = np.max(all_dists)
    if not np.isfinite(m):
        return 0.0
    exps = np.exp(-beta * (all_dists - m))
    s = np.sum(exps)
    if s == 0:
        return 0.0
    return m - np.log(s) / beta

def _softmin_grad(c, beta):
    """
    Gradient of the softmin radius.
    Returns (grad, radius) where grad is shape (N, 3).
    """
    wall_dists = 0.5 - np.abs(c)
    wall_flat = wall_dists.flatten()
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, 1)
    pair_dists = dists[iu] / 2.0
    
    all_dists = np.concatenate([wall_flat, pair_dists])
    
    m = np.max(all_dists)
    if not np.isfinite(m):
        return np.zeros((N, 3)), 0.0
    exps = np.exp(-beta * (all_dists - m))
    s = np.sum(exps)
    if s == 0:
        return np.zeros((N, 3)), 0.0
    weights = exps / s  # shape (1375,)
    
    radius = m - np.log(s) / beta
    
    grad = np.zeros((N, 3))
    
    # Wall part: first 150 weights
    w_wall = weights[:150].reshape(N, 3)
    # d(0.5 - |c_ik|)/dc_ik = -sign(c_ik)
    # So grad += -w_wall * sign(c)
    # But we want to maximize radius, so we want to move in direction that increases radius.
    # The softmin is a smooth max of the "badness" (or rather, it's a smooth min of the "goodness").
    # Actually, softmin(x) approximates min(x).
    # We want to maximize min(x), so we want to move in direction of +grad(min(x)).
    # grad(softmin(x)) = -sum(w_i * grad(x_i)) / ... wait.
    # Let f(x) = softmin(x) = -1/beta * log(sum(exp(-beta x_i)))
    # df/dx_i = -1/beta * (1/sum) * (-beta * exp(-beta x_i)) = exp(-beta x_i) / sum = w_i
    # So grad(f) = sum_i w_i * grad(x_i)
    # x_i for walls: 0.5 - |c_ik|
    # grad(x_i) wrt c_ik = -sign(c_ik)
    # So grad(f) wrt c_ik = w_ik * (-sign(c_ik))
    
    grad_wall = -w_wall * np.sign(c)
    grad += grad_wall
    
    # Pair part: next 1225 weights
    w_pair = weights[150:]
    # x_i = |c_i - c_j| / 2
    # grad(x_i) wrt c_i = (c_i - c_j) / (2 * |c_i - c_j|)
    # grad(x_i) wrt c_j = (c_j - c_i) / (2 * |c_i - c_j|) = -grad(x_i) wrt c_i
    
    for idx in range(len(w_pair)):
        i, j = iu[0][idx], iu[1][idx]
        d = c[i] - c[j]
        dist = np.linalg.norm(d)
        if dist < 1e-12:
            continue
        grad_i = d / (2 * dist)  # grad of (|d|/2) wrt c_i
        grad[i] += w_pair[idx] * grad_i
        grad[j] -= w_pair[idx] * grad_i
    
    return grad, radius

def _local_search(c, max_iter=500, step_size=0.01):
    """
    Simple local search: try moving each sphere in random directions,
    accept if radius improves.
    """
    c = c.copy()
    r = _compute_radius(c)
    best_c = c.copy()
    best_r = r
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 10:
            break
        # Pick a random sphere
        i = np.random.randint(N)
        # Try a few random directions
        improved = False
        for _ in range(10):
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            for s in [step_size, step_size * 0.5, step_size * 2]:
                c_new = c.copy()
                c_new[i] += direction * s
                # Clip to cube
                c_new = np.clip(c_new, -0.499, 0.499)
                r_new = _compute_radius(c_new)
                if r_new > r + 1e-12:
                    c = c_new
                    r = r_new
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Reduce step size if stuck
            step_size *= 0.8
            if step_size < 1e-6:
                break
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    return best_c, best_r

def _repulsion_anneal(c, beta_start=10, beta_end=100, n_iter=50):
    """
    Simulated annealing with repulsion forces.
    """
    c = c.copy()
    r = _compute_radius(c)
    best_c = c.copy()
    best_r = r
    
    T = 0.01  # temperature
    for it in range(n_iter):
        if time.time() > DEADLINE - 20:
            break
        beta = beta_start * (beta_end / beta_start) ** (it / n_iter)
        
        # Compute softmin gradient
        grad, r_soft = _softmin_grad(c, beta)
        
        # Move in gradient direction
        c_new = c + grad * 0.01
        c_new = np.clip(c_new, -0.499, 0.499)
        
        r_new = _compute_radius(c_new)
        
        # Accept if better or with probability exp(-delta/T)
        if r_new > r:
            c = c_new
            r = r_new
            if r > best_r:
                best_r = r
                best_c = c.copy()
        else:
            if np.random.rand() < np.exp(-(r - r_new) / T):
                c = c_new
                r = r_new
        
        T *= 0.95
    
    return best_c, best_r

def construct():
    start_time = time.time()
    
    # Initialize with a grid
    m = 5
    ticks_xy = (np.arange(m) + 0.5) / m - 0.5
    ticks_z = np.array([-0.25, 0.25])
    
    grid = []
    for z in ticks_z:
        for y in ticks_xy:
            for x in ticks_xy:
                grid.append([x, y, z])
    
    c = np.array(grid)
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Phase 1: Softmin optimization using scipy
    def objective(x):
        c_x = x.reshape(-1, 3)
        return -_softmin_radius(c_x, 50)
    
    def objective_grad(x):
        c_x = x.reshape(-1, 3)
        grad, _ = _softmin_grad(c_x, 50)
        return -grad.flatten()
    
    x0 = c.flatten()
    bounds = [(-0.499, 0.499)] * (N * 3)
    
    try:
        res = minimize(objective, x0, jac=objective_grad, method='L-BFGS-B', 
                      bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.x is not None and np.all(np.isfinite(res.x)):
            c_opt = res.x.reshape(-1, 3)
            r_opt = _compute_radius(c_opt)
            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt
    except Exception:
        pass
    
    # Phase 2: Repulsion annealing
    if time.time() < DEADLINE - 30:
        c2, r2 = _repulsion_anneal(best_c, beta_start=20, beta_end=200, n_iter=100)
        if r2 > best_r:
            best_r = r2
            best_c = c2
    
    # Phase 3: Local search
    if time.time() < DEADLINE - 15:
        c3, r3 = _local_search(best_c, max_iter=1000, step_size=0.005)
        if r3 > best_r:
            best_r = r3
            best_c = c3
    
    # Phase 4: Final polish with softmin
    if time.time() < DEADLINE - 10:
        try:
            def objective2(x):
                c_x = x.reshape(-1, 3)
                return -_softmin_radius(c_x, 100)
            
            def objective_grad2(x):
                c_x = x.reshape(-1, 3)
                grad, _ = _softmin_grad(c_x, 100)
                return -grad.flatten()
            
            x1 = best_c.flatten()
            res2 = minimize(objective2, x1, jac=objective_grad2, method='L-BFGS-B', 
                          bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res2.x is not None and np.all(np.isfinite(res2.x)):
                c_opt2 = res2.x.reshape(-1, 3)
                r_opt2 = _compute_radius(c_opt2)
                if r_opt2 > best_r:
                    best_r = r_opt2
                    best_c = c_opt2
        except Exception:
            pass
    
    # Final local search
    if time.time() < DEADLINE - 5:
        c4, r4 = _local_search(best_c, max_iter=500, step_size=0.002)
        if r4 > best_r:
            best_r = r4
            best_c = c4
    
    return best_c
