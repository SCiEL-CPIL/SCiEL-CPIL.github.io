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

def neg_sum_radii(c):
    return -sum_radii(c)

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
    np.random.seed(42)
    
    # 1. Grids
    for r, c in [(6,6), (5,7), (7,5)]:
        xs = (np.arange(c) + 0.5) / c
        ys = (np.arange(r) + 0.5) / r
        pts = []
        for i in range(r):
            for j in range(c):
                if len(pts) < N:
                    pts.append([xs[j], ys[i]])
        while len(pts) < N:
            pts.append(np.random.uniform(0.1, 0.9, 2))
        inits.append(np.array(pts).flatten())
        
    # 2. Random
    for _ in range(5):
        inits.append(np.random.uniform(0.05, 0.95, size=(N, 2)).flatten())

    bounds = [(0.001, 0.999), (0.001, 0.999)] * N
    
    for c_init in inits:
        if time.time() - start_time > timeout - 10:
            break
            
        # 1. Powell (Derivative-free, robust for non-smooth)
        try:
            res = minimize(neg_sum_radii, c_init, method='Powell', bounds=bounds,
                           options={'maxiter': 50, 'maxfev': 1000})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
        except:
            pass
            
        # 2. L-BFGS-B (Fast, gradient-based)
        if time.time() - start_time < timeout - 5:
            try:
                res2 = minimize(neg_sum_radii, c_init, jac=lambda c: -grad_sum_radii(c),
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 100, 'ftol': 1e-12})
                val2 = -res2.fun
                if val2 > best_val:
                    best_val = val2
                    best_c = res2.x.copy()
            except:
                pass

    if best_c is None:
        best_c = inits[0]
        
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
