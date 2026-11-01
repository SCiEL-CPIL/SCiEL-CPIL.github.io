import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi, voronoi_plot_2d
import time

N = 32

def compute_voronoi_neighbors(p):
    """
    Returns a list of neighbors for each point, and a flag for whether the point is constrained by a wall.
    Actually, we just need to compute the gradient accurately.
    """
    try:
        vor = Voronoi(p)
    except Exception:
        # Fallback if Voronoi fails (e.g., duplicate points)
        return None

    # Map Voronoi point indices to original indices
    # vor.points is the input p
    # vor.regions are lists of vertex indices
    # vor.ridge_points are pairs of point indices for each ridge
    
    neighbors = [[] for _ in range(N)]
    for i, j in vor.ridge_points:
        # Check if the ridge is inside the unit square
        # vor.ridge_vertices[i] gives the vertex indices of the ridge
        verts = vor.ridge_vertices[i]
        if -1 in verts:
            # Ridge extends to infinity, but we are in a box.
            # We need to check if the segment is within the box.
            # For now, assume if it's a ridge, it's a neighbor.
            # Better: check if the midpoint is inside the box?
            pass
        neighbors[i].append(j)
        neighbors[j].append(i)
        
    return neighbors

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
    
    # Compute distances
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    
    # For each point, find all active constraints
    for i in range(N):
        r_i = min(min_wall_dist[i], min_neighbor_dist[i])
        
        # Find active walls
        active_wall_grads = []
        for w_idx in range(4):
            # Walls: 0:x, 1:1-x, 2:y, 3:1-y
            if w_idx == 0:
                d = p[i, 0]
                g = np.array([1.0, 0.0])
            elif w_idx == 1:
                d = 1.0 - p[i, 0]
                g = np.array([-1.0, 0.0])
            elif w_idx == 2:
                d = p[i, 1]
                g = np.array([0.0, 1.0])
            elif w_idx == 3:
                d = 1.0 - p[i, 1]
                g = np.array([0.0, -1.0])
            
            if abs(d - r_i) < 1e-6:
                active_wall_grads.append(g)
                
        # Find active neighbors
        active_neighbor_grads_i = []
        active_neighbor_grads_j = {} # j -> grad
        
        for j in range(N):
            if i == j: continue
            d = dists[i, j]
            if d < 1e-10: continue
            if abs(d / 2.0 - r_i) < 1e-6:
                vec = p[i] - p[j]
                g_i = 0.5 * vec / d
                g_j = -0.5 * vec / d
                active_neighbor_grads_i.append(g_i)
                if j not in active_neighbor_grads_j:
                    active_neighbor_grads_j[j] = np.zeros(2)
                active_neighbor_grads_j[j] += g_j
                
        # Sum gradients
        total_grad_i = np.zeros(2)
        count = 0
        for g in active_wall_grads:
            total_grad_i += g
            count += 1
        for g in active_neighbor_grads_i:
            total_grad_i += g
            count += 1
            
        if count > 0:
            # The gradient of the min function is the average of the gradients of the active functions
            # if we assume the "active set" is unique or we are at a kink.
            # Actually, for max-min, the subgradient is the convex hull.
            # Using the average is a standard heuristic for these packing problems.
            grad[i] += total_grad_i / count
            
        # Update neighbors
        for j, g_j in active_neighbor_grads_j.items():
            # We need to be careful: the gradient for j due to constraint (i,j)
            # is only valid if (i,j) is also an active constraint for r_j?
            # Or does the objective F = sum r_k include r_i which depends on p_j?
            # Yes, F = ... + r_i + ...
            # r_i depends on p_j.
            # So we MUST add the gradient contribution to p_j.
            # However, we also need to consider if r_j is constrained by i.
            # The loop above only adds to grad[j] if i is active for j.
            # But r_i's dependence on p_j must be accounted for regardless of r_j's state.
            # So we add g_j to grad[j].
            # But wait, if we do this for every i, we might double count or miss things?
            # No, each pair (i,j) is considered when i is the "center" of the radius.
            # But the constraint is symmetric.
            # If i is active for j, we will handle it when we process j?
            # No, the gradient of r_i w.r.t p_j is distinct from the gradient of r_j w.r.t p_i.
            # We must add the contribution of r_i to grad[j].
            grad[j] += g_j

    return grad.flatten()

def construct():
    start_time = time.time()
    timeout = 115
    
    best_c = None
    best_val = -np.inf
    
    # Inits
    inits = []
    # Grid
    for r, c in [(6,6), (5,7), (7,5)]:
        xs = (np.arange(c) + 0.5) / c
        ys = (np.arange(r) + 0.5) / r
        points = []
        for i in range(r):
            for j in range(c):
                if len(points) < N:
                    points.append([xs[j], ys[i]])
        inits.append(np.array(points).flatten())
        
    # Random
    np.random.seed(42)
    for _ in range(5):
        inits.append(np.random.uniform(0.1, 0.9, size=(N, 2)).flatten())
        
    bounds = [(0.001, 0.999), (0.001, 0.999)] * N
    
    for c_init in inits:
        if time.time() - start_time > timeout - 20:
            break
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

    if best_c is None:
        best_c = np.array([(0.5, 0.5)] * N).flatten()
        
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
