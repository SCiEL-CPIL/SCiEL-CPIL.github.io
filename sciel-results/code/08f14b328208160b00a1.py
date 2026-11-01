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

def clip_polygon(polygon, a, b, c):
    """
    Clip polygon by half-plane ax + by + c <= 0.
    polygon: list of (x, y)
    """
    if not polygon:
        return []
    
    output = []
    n = len(polygon)
    for i in range(n):
        current = polygon[i]
        previous = polygon[i-1]
        
        # Calculate values
        current_val = a*current[0] + b*current[1] + c
        previous_val = a*previous[0] + b*previous[1] + c
        
        current_inside = current_val <= 0
        previous_inside = previous_val <= 0
        
        if current_inside:
            if not previous_inside:
                # Intersection
                # Line from previous to current
                # Parametric: P = Prev + t * (Curr - Prev)
                # a(Px) + b(Py) + c = 0
                dx = current[0] - previous[0]
                dy = current[1] - previous[1]
                denom = a*dx + b*dy
                if abs(denom) > 1e-12:
                    t = -(a*previous[0] + b*previous[1] + c) / denom
                    if 0 <= t <= 1:
                        ix = previous[0] + t*dx
                        iy = previous[1] + t*dy
                        output.append((ix, iy))
            output.append(current)
        else:
            if previous_inside:
                # Intersection
                dx = current[0] - previous[0]
                dy = current[1] - previous[1]
                denom = a*dx + b*dy
                if abs(denom) > 1e-12:
                    t = -(a*previous[0] + b*previous[1] + c) / denom
                    if 0 <= t <= 1:
                        ix = previous[0] + t*dx
                        iy = previous[1] + t*dy
                        output.append((ix, iy))
                        
    return output

def voronoi_centroid(p, i):
    """
    Compute the centroid of the Voronoi cell for point i.
    """
    # Start with the unit square
    polygon = [(0,0), (1,0), (1,1), (0,1)]
    
    pi = p[i]
    
    # Clip against walls
    # x >= 0 -> -x <= 0
    polygon = clip_polygon(polygon, -1, 0, 0)
    # x <= 1 -> x - 1 <= 0
    polygon = clip_polygon(polygon, 1, 0, -1)
    # y >= 0 -> -y <= 0
    polygon = clip_polygon(polygon, 0, -1, 0)
    # y <= 1 -> y - 1 <= 0
    polygon = clip_polygon(polygon, 0, 1, -1)
    
    if not polygon:
        return pi
        
    # Clip against other points
    for j in range(N):
        if j == i:
            continue
        pj = p[j]
        # Midpoint
        mid = (pi + pj) / 2.0
        # Normal vector pointing from pj to pi
        normal = pi - pj
        # Plane equation: normal . (x - mid) <= 0
        # normal . x - normal . mid <= 0
        a = normal[0]
        b = normal[1]
        c = -np.dot(normal, mid)
        
        polygon = clip_polygon(polygon, a, b, c)
        if not polygon:
            return pi
            
    if not polygon:
        return pi
        
    # Calculate centroid
    # Area and Centroid of polygon
    area = 0
    cx = 0
    cy = 0
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i+1)%n]
        cross = x0*y1 - x1*y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
        
    if abs(area) < 1e-10:
        return pi
        
    area /= 2.0
    cx /= (6 * area)
    cy /= (6 * area)
    
    return np.array([cx, cy])

def lloyd_relaxation(p, steps=20):
    """
    Perform Lloyd's relaxation (move to Voronoi centroids).
    """
    p = p.copy()
    for _ in range(steps):
        new_p = np.zeros_like(p)
        for i in range(N):
            new_p[i] = voronoi_centroid(p, i)
        # Clip to unit square to be safe
        new_p = np.clip(new_p, 0.01, 0.99)
        p = new_p
    return p

def construct():
    start_time = time.time()
    timeout = 115
    
    best_c = None
    best_val = -np.inf
    
    bounds = [(0.001, 0.999), (0.001, 0.999)] * N
    
    # Generate initial candidates
    candidates = []
    
    # 1. Random
    np.random.seed(42)
    for _ in range(5):
        candidates.append(np.random.uniform(0.1, 0.9, size=(N, 2)))
        
    # 2. Jittered Grid
    base = np.array([( (j+0.5)/6.0, (i+0.5)/6.0 ) for i in range(6) for j in range(6)])
    for _ in range(5):
        jitter = np.random.uniform(-0.05, 0.05, size=(N, 2))
        candidates.append(np.clip(base + jitter, 0.05, 0.95))
        
    # 3. Halton-like (low discrepancy)
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    halton_pts = np.array([[halton(i+1, 2), halton(i+1, 3)] for i in range(N)])
    candidates.append(halton_pts)
    
    for idx, p_init in enumerate(candidates):
        if time.time() - start_time > timeout - 10:
            break
            
        # Step 1: Lloyd's Relaxation (Voronoi-based)
        p_lloyd = lloyd_relaxation(p_init, steps=15)
        
        # Step 2: L-BFGS-B Refinement
        c_flat = p_lloyd.flatten()
        try:
            res = minimize(lambda c: -sum_radii(c), c_flat, jac=lambda c: -grad_sum_radii(c),
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
        except:
            pass
            
        # Also try direct L-BFGS-B from original init (in case Lloyd's degrades sum)
        if time.time() - start_time < timeout - 10:
            c_flat_orig = p_init.flatten()
            try:
                res = minimize(lambda c: -sum_radii(c), c_flat_orig, jac=lambda c: -grad_sum_radii(c),
                               method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_c = res.x.copy()
            except:
                pass

    if best_c is None:
        best_c = candidates[0].flatten()
        
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
