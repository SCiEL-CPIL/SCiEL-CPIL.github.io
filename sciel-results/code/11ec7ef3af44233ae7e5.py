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
    
    # Distances to walls
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    
    # Distances to other centers
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Find closest neighbor for each point
    min_neighbor_dist = np.min(dists, axis=1)
    min_neighbor_idx = np.argmin(dists, axis=1)
    
    for i in range(N):
        wall_r = min_wall_dist[i]
        neigh_r = min_neighbor_dist[i] / 2.0
        
        if wall_r < neigh_r:
            # Limited by wall
            # r_i = min(p_ix, p_iy, 1-p_ix, 1-p_iy)
            px, py = p[i]
            # Determine which wall is active
            # Gradients:
            # d(p_x)/dp_x = 1, d(p_y)/dp_y = 1
            # d(1-p_x)/dp_x = -1, d(1-p_y)/dp_y = -1
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
            j = min_neighbor_idx[i]
            d = min_neighbor_dist[i]
            if d > 1e-10:
                vec = p[i] - p[j]
                # r_i = 0.5 * ||p_i - p_j||
                # grad r_i wrt p_i = 0.5 * (p_i - p_j) / d
                grad[i] += 0.5 * vec / d
                # Note: The gradient of the sum includes the term for r_j as well.
                # r_j is limited by i (since i is j's closest neighbor? Not necessarily, but i is a neighbor).
                # Actually, the gradient of the sum is sum_k grad r_k.
                # When we process k=i, we add grad r_i.
                # When we process k=j, we will add grad r_j.
                # Does grad r_i affect p_j? Yes.
                # d r_i / d p_j = d (0.5 * ||p_i - p_j||) / d p_j = -0.5 * (p_i - p_j) / d = 0.5 * (p_j - p_i) / d
                grad[j] += -0.5 * vec / d
    
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
    # If we need more points than grid size, we might need to add jittered ones or handle differently
    # For N=32, 6x6=36 is enough.
    return np.array(points[:n]).flatten()

def halton_sequence(n, dims=2):
    """Generate n points in the unit square using Halton sequences"""
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    
    points = []
    for i in range(1, n + 1):
        points.append([halton(i, 2), halton(i, 3)])
    return np.array(points)

def construct():
    start_time = time.time()
    timeout = 110
    
    best_c = None
    best_val = -np.inf
    
    # Initializations
    inits = []
    
    # 1. Standard Grids
    inits.append(create_grid_init(6, 6, N))
    inits.append(create_grid_init(5, 7, N))
    inits.append(create_grid_init(7, 5, N))
    
    # 2. Halton Sequence (Low Discrepancy)
    halton_points = halton_sequence(N)
    inits.append(halton_points.flatten())
    
    # 3. Jittered Grids
    np.random.seed(42)
    base = create_grid_init(6, 6, N).reshape(N, 2)
    for _ in range(5):
        jitter = np.random.uniform(-0.02, 0.02, size=(N, 2))
        p = np.clip(base + jitter, 0.01, 0.99)
        inits.append(p.flatten())
        
    # 4. Random
    for _ in range(5):
        p = np.random.uniform(0.05, 0.95, size=(N, 2))
        inits.append(p.flatten())
        
    bounds = [(0, 1), (0, 1)] * N
    
    for idx, c_init in enumerate(inits):
        if time.time() - start_time > timeout - 10: # Leave time for final polish
            break
            
        # 1. L-BFGS-B
        try:
            res = minimize(lambda c: -sum_radii(c), c_init, jac=lambda c: -grad_sum_radii(c),
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
        except:
            pass
        
        # Check time
        if time.time() - start_time > timeout - 5:
            break

    # Final Polish with Powell on the best solution
    if best_c is not None and time.time() - start_time < timeout - 2:
        try:
            res = minimize(lambda c: -sum_radii(c), best_c, method='Powell',
                          options={'maxiter': 200, 'disp': False, 'ftol': 1e-12})
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
