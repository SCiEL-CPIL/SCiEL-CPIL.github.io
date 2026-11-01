import numpy as np
from scipy.optimize import minimize
import time

N = 32

def sum_radii(c):
    p = c.reshape(N, 2)
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    return np.sum(radii)

def grad_sum_radii(c):
    p = c.reshape(N, 2)
    grad = np.zeros_like(p)
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1)
    min_neighbor_idx = np.argmin(dists, axis=1)
    
    for i in range(N):
        wall_r = min_wall_dist[i]
        neigh_r = min_neighbor_dist[i] / 2.0
        
        # Determine active constraints for circle i
        # We want to move i to increase min(wall_r, neigh_r)
        # If wall_r < neigh_r, move away from wall
        # If neigh_r < wall_r, move away from neighbor
        # If equal, move in direction that increases both (or average subgradient)
        
        if wall_r < neigh_r - 1e-10:
            px, py = p[i]
            walls = [px, 1-px, py, 1-py]
            wall_grads = [[1,0], [-1,0], [0,1], [0,-1]]
            min_val = min(walls)
            active = [g for w, g in zip(walls, wall_grads) if abs(w - min_val) < 1e-10]
            if active:
                avg_grad = np.mean(active, axis=0)
                grad[i] += avg_grad
        elif neigh_r < wall_r - 1e-10:
            j = min_neighbor_idx[i]
            d = min_neighbor_dist[i]
            if d > 1e-10:
                vec = p[i] - p[j]
                grad[i] += 0.5 * vec / d
                grad[j] += -0.5 * vec / d
        else:
            # Mixed or degenerate case: use average of available subgradients
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
                
            # If both are active, the gradient of the min is in the convex hull of the gradients
            # We use the average
            grad[i] += 0.5 * (wall_grad + neigh_grad_i)
            grad[j] += 0.5 * neigh_grad_j
    return grad.flatten()

def phyllotaxis_points(n, radius=0.5, center=(0.5, 0.5)):
    """Generate points using phyllotaxis arrangement (sunflower pattern)"""
    points = []
    for i in range(n):
        r = radius * np.sqrt(i / n)
        theta = i * 137.508 * np.pi / 180.0 # Golden angle in radians
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        # Clip to unit square
        x = np.clip(x, 0.01, 0.99)
        y = np.clip(y, 0.01, 0.99)
        points.append([x, y])
    return np.array(points)

def local_search(c, n_iter=200, time_limit=5.0):
    """
    Perform a local random search to escape local minima.
    """
    p = c.reshape(N, 2).copy()
    current_val = sum_radii(p.flatten())
    start_time = time.time()
    step_size = 0.02
    
    for _ in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        i = np.random.randint(N)
        # Try several random perturbations for this point
        for _ in range(5):
            new_p = p.copy()
            noise = np.random.normal(0, step_size, size=2)
            new_p[i] = np.clip(new_p[i] + noise, 0.01, 0.99)
            new_val = sum_radii(new_p.flatten())
            if new_val > current_val + 1e-8:
                p = new_p
                current_val = new_val
                break
    return p.flatten()

def create_grid_init(rows, cols, n):
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) < n:
                points.append([xs[j], ys[i]])
    return np.array(points[:n]).flatten()

def halton_sequence(n, dims=2):
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
    timeout = 115
    
    best_c = None
    best_val = -np.inf
    
    inits = []
    
    # 1. Grids
    for r, c in [(6,6), (5,7), (7,5), (4,8), (8,4), (6,5), (5,6)]:
        inits.append(create_grid_init(r, c, N))
        
    # 2. Halton
    inits.append(halton_sequence(N).flatten())
    
    # 3. Phyllotaxis
    inits.append(phyllotaxis_points(N).flatten())
    
    # 4. Jittered Grids
    np.random.seed(42)
    base = create_grid_init(6, 6, N).reshape(N, 2)
    for jitter_scale in [0.02, 0.04, 0.06]:
        for _ in range(5):
            jitter = np.random.uniform(-jitter_scale, jitter_scale, size=(N, 2))
            p = np.clip(base + jitter, 0.01, 0.99)
            inits.append(p.flatten())
            
    # 5. Random
    for _ in range(10):
        p = np.random.uniform(0.05, 0.95, size=(N, 2))
        inits.append(p.flatten())
        
    bounds = [(0.001, 0.999), (0.001, 0.999)] * N
    
    # Optimization loop
    for idx, c_init in enumerate(inits):
        if time.time() - start_time > timeout - 20:
            break
            
        # 1. L-BFGS-B
        try:
            res = minimize(lambda c: -sum_radii(c), c_init, jac=lambda c: -grad_sum_radii(c),
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
        except:
            pass
        
        # 2. Local Search Refinement (replaces coordinate descent)
        if time.time() - start_time < timeout - 10:
            try:
                # Refine the result of LBFGS
                start_point = res.x if 'res' in locals() and res.x is not None else c_init
                refined = local_search(start_point, n_iter=100, time_limit=3.0)
                val = sum_radii(refined)
                if val > best_val:
                    best_val = val
                    best_c = refined.copy()
            except:
                pass
        
        if time.time() - start_time > timeout - 15:
            break

    if best_c is not None:
        # Final Polish
        if time.time() - start_time < timeout - 5:
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
        
        if time.time() - start_time < timeout - 2:
            try:
                refined = local_search(best_c, n_iter=100, time_limit=2.0)
                val = sum_radii(refined)
                if val > best_val:
                    best_val = val
                    best_c = refined.copy()
            except:
                pass

    if best_c is None:
        best_c = create_grid_init(6, 6, N)
        
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
