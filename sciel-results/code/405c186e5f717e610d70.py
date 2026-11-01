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
    Gradient of sum_radii using subgradient method for non-smooth points
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
        
        if wall_r < neigh_r - 1e-10:
            # Limited by wall - use subgradient (average of active constraints)
            px, py = p[i]
            walls = [px, 1-px, py, 1-py]
            wall_grads = [[1,0], [-1,0], [0,1], [0,-1]]
            min_val = min(walls)
            active = [g for w, g in zip(walls, wall_grads) if abs(w - min_val) < 1e-10]
            if active:
                avg_grad = np.mean(active, axis=0)
                grad[i] += avg_grad
        elif neigh_r < wall_r - 1e-10:
            # Limited by neighbor j
            j = min_neighbor_idx[i]
            d = min_neighbor_dist[i]
            if d > 1e-10:
                vec = p[i] - p[j]
                # r_i = 0.5 * ||p_i - p_j||
                # grad r_i wrt p_i = 0.5 * (p_i - p_j) / d
                grad[i] += 0.5 * vec / d
                # Gradient contribution to p_j from r_i
                grad[j] += -0.5 * vec / d
        else:
            # At a kink point - use average of both gradients
            px, py = p[i]
            walls = [px, 1-px, py, 1-py]
            wall_grads = [[1,0], [-1,0], [0,1], [0,-1]]
            min_val = min(walls)
            active_walls = [g for w, g in zip(walls, wall_grads) if abs(w - min_val) < 1e-10]
            wall_grad = np.mean(active_walls, axis=0) if active_walls else np.zeros(2)
            
            j = min_neighbor_idx[i]
            d = min_neighbor_dist[i]
            if d > 1e-10:
                vec = p[i] - p[j]
                neigh_grad_i = 0.5 * vec / d
                neigh_grad_j = -0.5 * vec / d
            else:
                neigh_grad_i = np.zeros(2)
                neigh_grad_j = np.zeros(2)
            
            # Average of wall and neighbor gradients
            grad[i] += 0.5 * (wall_grad + neigh_grad_i)
            grad[j] += 0.5 * neigh_grad_j
    
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

def voronoi_refine(c, max_iter=100, tol=1e-10):
    """
    Refine using Voronoi-based projection:
    For each circle, move it toward the center of its Voronoi cell's inradius
    """
    p = c.reshape(N, 2).copy()
    
    for _ in range(max_iter):
        p_old = p.copy()
        
        for i in range(N):
            # Compute inradius and direction to improve
            px, py = p[i]
            
            # Distance to walls
            wall_dists = [px, 1-px, py, 1-py]
            min_wall = min(wall_dists)
            
            # Distance to neighbors
            diff = p[i] - p
            dists = np.sqrt(np.sum(diff**2, axis=1))
            dists[i] = np.inf
            min_neigh = np.min(dists) / 2.0
            min_neigh_idx = np.argmin(dists)
            
            r = min(min_wall, min_neigh)
            
            if min_wall < min_neigh - 1e-10:
                # Move away from closest wall
                if px == min_wall:
                    p[i, 0] += 0.01
                elif 1-px == min_wall:
                    p[i, 0] -= 0.01
                elif py == min_wall:
                    p[i, 1] += 0.01
                else:
                    p[i, 1] -= 0.01
            elif min_neigh < min_wall - 1e-10:
                # Move away from closest neighbor
                vec = p[i] - p[min_neigh_idx]
                d = np.linalg.norm(vec)
                if d > 1e-10:
                    p[i] += 0.01 * vec / d
            else:
                # At kink - try small random perturbation
                p[i] += np.random.uniform(-0.005, 0.005, 2)
            
            p[i] = np.clip(p[i], 0.001, 0.999)
        
        if np.max(np.abs(p - p_old)) < tol:
            break
    
    return p.flatten()

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
    inits.append(create_grid_init(4, 8, N))
    inits.append(create_grid_init(8, 4, N))
    
    # 2. Halton Sequence (Low Discrepancy)
    halton_points = halton_sequence(N)
    inits.append(halton_points.flatten())
    
    # 3. Jittered Grids with varying amounts
    np.random.seed(42)
    base = create_grid_init(6, 6, N).reshape(N, 2)
    for jitter_scale in [0.01, 0.02, 0.03, 0.05]:
        for _ in range(3):
            jitter = np.random.uniform(-jitter_scale, jitter_scale, size=(N, 2))
            p = np.clip(base + jitter, 0.01, 0.99)
            inits.append(p.flatten())
        
    # 4. Random
    for _ in range(5):
        p = np.random.uniform(0.05, 0.95, size=(N, 2))
        inits.append(p.flatten())
        
    # 5. Hexagonal-like pattern
    hex_points = []
    for row in range(6):
        for col in range(6):
            if len(hex_points) < N:
                x = (col + 0.5 + (row % 2) * 0.5) / 6.0
                y = (row + 0.5) / 6.0
                hex_points.append([x, y])
    inits.append(np.array(hex_points[:N]).flatten())
    
    bounds = [(0.001, 0.999), (0.001, 0.999)] * N
    
    for idx, c_init in enumerate(inits):
        if time.time() - start_time > timeout - 15: # Leave time for final polish
            break
            
        # 1. L-BFGS-B with improved settings
        try:
            res = minimize(lambda c: -sum_radii(c), c_init, jac=lambda c: -grad_sum_radii(c),
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
        except:
            pass
        
        # 2. Voronoi-based refinement
        if time.time() - start_time < timeout - 10:
            try:
                refined = voronoi_refine(res.x if 'res' in locals() else c_init, max_iter=50)
                val = sum_radii(refined)
                if val > best_val:
                    best_val = val
                    best_c = refined.copy()
            except:
                pass
        
        # Check time
        if time.time() - start_time > timeout - 8:
            break

    # Final Polish with multiple methods
    if best_c is not None:
        # Powell refinement
        if time.time() - start_time < timeout - 3:
            try:
                res = minimize(lambda c: -sum_radii(c), best_c, method='Powell',
                              options={'maxiter': 300, 'disp': False, 'ftol': 1e-14, 'xtol': 1e-14})
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_c = res.x.copy()
            except:
                pass
        
        # Additional L-BFGS-B polish
        if time.time() - start_time < timeout - 1:
            try:
                res = minimize(lambda c: -sum_radii(c), best_c, jac=lambda c: -grad_sum_radii(c),
                               method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
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
