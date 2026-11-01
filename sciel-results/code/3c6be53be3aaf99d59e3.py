import numpy as np
from scipy.optimize import minimize
import time

N = 32

def soft_min(a, b, t=50.0):
    """Smooth approximation of min(a, b) using log-sum-exp trick"""
    m = np.minimum(a, b)
    return m - np.logaddexp(-t*a, -t*b) / t

def sum_radii_smooth(c, t=50.0):
    """
    c: flat array of length 2*N, representing N centers (x, y)
    Returns: sum of inradii using soft-min for smoothness
    """
    p = c.reshape(N, 2)
    # Distances to walls
    dist_walls = np.minimum(p, 1 - p)  # shape (N, 2)
    min_wall_dist = np.min(dist_walls, axis=1)  # shape (N,)
    
    # Distances to other centers / 2
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    
    radii = soft_min(min_wall_dist, min_neighbor_dist, t)
    return np.sum(radii)

def grad_sum_radii_smooth(c, t=50.0):
    """
    Gradient of sum_radii_smooth
    """
    p = c.reshape(N, 2)
    grad = np.zeros_like(p)
    
    # Wall distances
    dist_walls = np.minimum(p, 1 - p)  # (N, 2)
    min_wall_dist = np.min(dist_walls, axis=1)  # (N,)
    
    # Neighbor distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    min_neighbor_idx = np.argmin(dists, axis=1)
    
    # For each point, compute gradient of soft_min(wall_r, neigh_r)
    for k in range(N):
        wall_r = min_wall_dist[k]
        neigh_r = min_neighbor_dist[k]
        
        # soft_min(a, b, t) = -1/t * log(exp(-t*a) + exp(-t*b))
        # d/da = exp(-t*a) / (exp(-t*a) + exp(-t*b))
        # d/db = exp(-t*b) / (exp(-t*a) + exp(-t*b))
        
        ea = np.exp(-t * wall_r)
        eb = np.exp(-t * neigh_r)
        denom = ea + eb
        
        if denom < 1e-300:
            # Both very large, use min
            dwall = 1.0
            dneigh = 0.0
        else:
            dwall = ea / denom
            dneigh = eb / denom
        
        # Gradient of wall_r with respect to p[k]
        # wall_r = min(p_kx, 1-p_kx, p_ky, 1-p_ky)
        px, py = p[k]
        walls = [px, 1 - px, py, 1 - py]
        wall_grad = np.zeros(2)
        min_wall_idx = np.argmin(walls)
        if min_wall_idx == 0:  # px
            wall_grad[0] = 1.0
        elif min_wall_idx == 1:  # 1-px
            wall_grad[0] = -1.0
        elif min_wall_idx == 2:  # py
            wall_grad[1] = 1.0
        else:  # 1-py
            wall_grad[1] = -1.0
        
        # Gradient of neigh_r with respect to p[k]
        # neigh_r = 0.5 * dist(k, j) where j is closest neighbor
        j = min_neighbor_idx[k]
        d = dists[k, j]
        if d > 1e-10:
            vec = p[k] - p[j]
            neigh_grad_k = 0.5 * vec / d
        else:
            neigh_grad_k = np.zeros(2)
        
        # Total gradient for p[k]
        grad[k] = dwall * wall_grad + dneigh * neigh_grad_k
    
    return grad.flatten()

def sum_radii_exact(c):
    """
    c: flat array of length 2*N, representing N centers (x, y)
    Returns: sum of inradii (exact, non-smooth)
    """
    p = c.reshape(N, 2)
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    return np.sum(radii)

def create_grid_init(rows, cols, n):
    """Create a grid initialization"""
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) < n:
                points.append([xs[j], ys[i]])
    return np.array(points).flatten()

def construct():
    start_time = time.time()
    timeout = 110
    
    best_c = None
    best_val = -np.inf
    
    # Multiple initialization strategies
    inits = []
    
    # Regular grids
    inits.append(create_grid_init(6, 6, N))
    inits.append(create_grid_init(5, 7, N))
    inits.append(create_grid_init(7, 5, N))
    inits.append(create_grid_init(4, 8, N))
    inits.append(create_grid_init(8, 4, N))
    
    # Jittered grids
    np.random.seed(42)
    for base_rows, base_cols in [(6, 6), (5, 7), (7, 5)]:
        base = create_grid_init(base_rows, base_cols, N).reshape(N, 2)
        for _ in range(5):
            jitter = np.random.uniform(-0.08, 0.08, size=(N, 2))
            p = np.clip(base + jitter, 0.02, 0.98)
            inits.append(p.flatten())
    
    # Random inits
    for _ in range(10):
        p = np.random.uniform(0.05, 0.95, size=(N, 2))
        inits.append(p.flatten())
    
    bounds = [(0.01, 0.99), (0.01, 0.99)] * N
    
    t_smooth = 30.0  # Smoothing parameter
    
    for idx, c_init in enumerate(inits):
        if time.time() - start_time > timeout - 5:
            break
        
        # Phase 1: Smooth optimization with L-BFGS-B
        try:
            res = minimize(lambda c: -sum_radii_smooth(c, t_smooth), c_init,
                          jac=lambda c: -grad_sum_radii_smooth(c, t_smooth),
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
            val_smooth = -res.fun
            val_exact = sum_radii_exact(res.x)
            if val_exact > best_val:
                best_val = val_exact
                best_c = res.x.copy()
        except:
            pass
        
        # Phase 2: Local search with perturbations
        if best_c is not None and time.time() - start_time < timeout - 10:
            for _ in range(5):
                if time.time() - start_time > timeout - 10:
                    break
                # Add small random perturbation
                noise = np.random.normal(0, 0.01, size=best_c.shape)
                c_perturbed = np.clip(best_c + noise, 0.01, 0.99)
                
                try:
                    res2 = minimize(lambda c: -sum_radii_smooth(c, t_smooth), c_perturbed,
                                  jac=lambda c: -grad_sum_radii_smooth(c, t_smooth),
                                  method='L-BFGS-B', bounds=bounds,
                                  options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
                    val_exact2 = sum_radii_exact(res2.x)
                    if val_exact2 > best_val:
                        best_val = val_exact2
                        best_c = res2.x.copy()
                except:
                    pass
    
    # Final refinement with exact objective using Powell
    if best_c is not None and time.time() - start_time < timeout - 5:
        try:
            res_final = minimize(lambda c: -sum_radii_exact(c), best_c,
                                method='Powell',
                                options={'maxiter': 50, 'disp': False})
            val_final = -res_final.fun
            if val_final > best_val:
                best_val = val_final
                best_c = res_final.x.copy()
        except:
            pass
    
    if best_c is None:
        best_c = create_grid_init(6, 6, N)
        best_val = sum_radii_exact(best_c)
    
    # Compute final radii
    p = best_c.reshape(N, 2)
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    
    result = np.column_stack((p, radii))
    return result
