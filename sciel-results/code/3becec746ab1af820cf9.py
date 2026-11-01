import numpy as np
from scipy.optimize import minimize
import time

N = 32

def sum_radii(c):
    """
    c: flat array of length 2*N, representing N centers (x, y)
    Returns: sum of inradii of the Voronoi cells clipped to [0,1]^2
    """
    p = c.reshape(N, 2)
    # Distances to walls
    dist_walls = np.minimum(p, 1 - p) # shape (N, 2)
    min_wall_dist = np.min(dist_walls, axis=1) # shape (N,)
    
    # Distances to other centers / 2
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    return np.sum(radii)

def grad_sum_radii(c):
    """
    Gradient of sum_radii
    """
    p = c.reshape(N, 2)
    grad = np.zeros_like(p)
    
    # Wall gradients
    # r_i = min(p_i, 1-p_i)
    # If p_i < 0.5, r_i = p_i, grad = 1
    # If p_i > 0.5, r_i = 1-p_i, grad = -1
    # If limited by neighbor, wall grad is 0
    
    # Neighbor gradients
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Find closest neighbor for each point
    min_neighbor_dist = np.min(dists, axis=1)
    min_neighbor_idx = np.argmin(dists, axis=1)
    
    radii = np.zeros(N)
    # Determine limiting factor
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    
    for i in range(N):
        wall_r = min_wall_dist[i]
        neigh_r = min_neighbor_dist[i] / 2.0
        
        if wall_r < neigh_r:
            # Limited by wall
            radii[i] = wall_r
            # Gradient with respect to p[i]
            # r = min(p_ix, p_iy, 1-p_ix, 1-p_iy)
            # Find which wall is closest
            px, py = p[i]
            if px <= 1 - px and px <= py and px <= 1 - py:
                grad[i, 0] += 1.0
            elif 1 - px <= px and 1 - px <= py and 1 - px <= 1 - py:
                grad[i, 0] -= 1.0
            elif py <= px and py <= 1 - px and py <= 1 - py:
                grad[i, 1] += 1.0
            else:
                grad[i, 1] -= 1.0
        else:
            # Limited by neighbor j
            radii[i] = neigh_r
            j = min_neighbor_idx[i]
            # r = 0.5 * dist(i, j)
            # dr/dp_i = 0.5 * (p_i - p_j) / dist
            # dr/dp_j = 0.5 * (p_j - p_i) / dist
            d = min_neighbor_dist[i]
            if d > 1e-10:
                vec = p[i] - p[j]
                grad[i] += 0.5 * vec / d
                grad[j] += 0.5 * (-vec) / d
            # Note: This is only valid if i's closest neighbor is j.
            # But j's radius might be limited by i or someone else.
            # The gradient of the SUM is sum of gradients of each r_k.
            # So we must compute the gradient for ALL points k, not just i.
            
    # The above loop is inefficient and incorrect for the sum.
    # We need to compute the gradient for each point k independently.
    
    grad = np.zeros_like(p)
    for k in range(N):
        pk = p[k]
        wall_r = np.min(np.minimum(pk, 1 - pk))
        
        # Find closest neighbor
        dists_k = np.sqrt(np.sum((p - pk)**2, axis=1))
        dists_k[k] = np.inf
        min_d = np.min(dists_k)
        j = np.argmin(dists_k)
        neigh_r = min_d / 2.0
        
        if wall_r < neigh_r:
            # Limited by wall
            px, py = pk
            if px <= 1 - px and px <= py and px <= 1 - py:
                grad[k, 0] += 1.0
            elif 1 - px <= px and 1 - px <= py and 1 - px <= 1 - py:
                grad[k, 0] -= 1.0
            elif py <= px and py <= 1 - px and py <= 1 - py:
                grad[k, 1] += 1.0
            else:
                grad[k, 1] -= 1.0
        else:
            # Limited by neighbor j
            if min_d > 1e-10:
                vec = pk - p[j]
                grad[k] += 0.5 * vec / min_d
                # This gradient contribution is for r_k.
                # The term for r_j will be computed when k=j.
    
    return grad.flatten()

def create_grid_init(rows, cols, n):
    """Create a grid initialization with specified rows and cols, taking first n points"""
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
    
    # Initializations
    inits = []
    # Grids
    inits.append(create_grid_init(6, 6, N))
    inits.append(create_grid_init(5, 7, N))
    inits.append(create_grid_init(7, 5, N))
    
    # Jittered Grids
    np.random.seed(42)
    base = create_grid_init(6, 6, N).reshape(N, 2)
    for _ in range(10):
        jitter = np.random.uniform(-0.05, 0.05, size=(N, 2))
        p = np.clip(base + jitter, 0.01, 0.99)
        inits.append(p.flatten())
        
    # Random
    for _ in range(5):
        p = np.random.uniform(0.05, 0.95, size=(N, 2))
        inits.append(p.flatten())
        
    bounds = [(0, 1), (0, 1)] * N
    
    for idx, c_init in enumerate(inits):
        if time.time() - start_time > timeout:
            break
            
        # 1. L-BFGS-B
        try:
            res = minimize(lambda c: -sum_radii(c), c_init, jac=lambda c: -grad_sum_radii(c),
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
        except:
            pass
        
        if time.time() - start_time > timeout:
            break
            
        # 2. Powell Refinement
        if best_c is not None and time.time() - start_time < timeout - 10:
            try:
                res = minimize(lambda c: -sum_radii(c), best_c, method='Powell',
                              options={'maxiter': 100, 'disp': False})
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_c = res.x.copy()
            except:
                pass
    
    if best_c is None:
        best_c = create_grid_init(6, 6, N)
        best_val = sum_radii(best_c)
        
    p = best_c.reshape(N, 2)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    
    result = np.column_stack((p, radii))
    return result
