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

def hexagonal_init(n):
    """
    Generate a hexagonal close-packed initial configuration fitted to the unit square.
    """
    best_points = None
    best_score = -np.inf
    
    # Try a few different scales
    for scale in [0.1, 0.12, 0.15, 0.18, 0.2]:
        points = []
        row = 0
        y = 0.0
        while y < 1.0 and len(points) < n:
            x_offset = 0.0 if row % 2 == 0 else scale / 2.0
            x = x_offset
            while x < 1.0 and len(points) < n:
                if x > 0.05 and x < 0.95 and y > 0.05 and y < 0.95:
                    points.append([x, y])
                x += scale
            y += scale * np.sqrt(3) / 2.0
            row += 1
        
        if len(points) < n:
            continue
            
        points = np.array(points[:n])
        
        min_x, min_y = np.min(points[:,0]), np.min(points[:,1])
        max_x, max_y = np.max(points[:,0]), np.max(points[:,1])
        
        width = max_x - min_x
        height = max_y - min_y
        
        if width < 1e-6 or height < 1e-6:
            continue
            
        target_w = 0.8
        target_h = 0.8
        scale_x = target_w / width
        scale_y = target_h / height
        s = min(scale_x, scale_y)
        
        points[:, 0] = (points[:, 0] - min_x) * s + 0.1
        points[:, 1] = (points[:, 1] - min_y) * s + 0.1
        
        points = np.clip(points, 0.01, 0.99)
        
        val = sum_radii(points.flatten())
        if val > best_score:
            best_score = val
            best_points = points
            
    if best_points is None:
        return create_grid_init(6, 6, N)
        
    return best_points.flatten()

def coordinate_descent_refine(c, max_sweeps=10, time_limit=10.0):
    """
    Refine using coordinate descent: optimize each point's position 
    to maximize its individual radius, holding others fixed.
    """
    p = c.reshape(N, 2).copy()
    start_time = time.time()
    
    for sweep in range(max_sweeps):
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(N):
            p_i = p[i].copy()
            # Objective: maximize radius of i
            # r_i = min(min_wall(p_i), 0.5 * min_j dist(p_i, p_j))
            # We minimize -r_i
            
            def neg_r_i(pos):
                pos = np.array(pos)
                # Walls
                dw = np.min(np.minimum(pos, 1-pos))
                # Neighbors
                diffs = p - pos
                diffs[i] = np.inf # Exclude self
                d_n = np.min(np.sqrt(np.sum(diffs**2, axis=1))) / 2.0
                return -min(dw, d_n)
            
            # Use Nelder-Mead or BFGS on 2D subproblem
            # Nelder-Mead is derivative-free and robust for this small dim
            try:
                res = minimize(neg_r_i, p_i, method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < -sum_radii(c.reshape(N,2)[i:i+1, :].flatten()) - 1e-10: # Rough check, better to just compare global sum later or local radius
                    # Actually, just accept if local radius improved
                    old_r_i = -neg_r_i(p_i)
                    new_r_i = -res.fun
                    if new_r_i > old_r_i + 1e-10:
                        p[i] = res.x
                        improved = True
            except:
                pass
            
        if not improved:
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
    
    # 4. Hexagonal
    inits.append(hexagonal_init(N))
    
    # 5. Jittered Grids
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
        
        # 2. Coordinate Descent Refinement
        if time.time() - start_time < timeout - 10:
            try:
                # Refine the result of LBFGS
                refined = coordinate_descent_refine(res.x if 'res' in locals() and res.x is not None else c_init, max_sweeps=5, time_limit=5.0)
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
                refined = coordinate_descent_refine(best_c, max_sweeps=5, time_limit=3.0)
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
