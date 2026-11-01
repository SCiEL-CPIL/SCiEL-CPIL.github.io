import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi
import time

N = 32

def voronoi_inradii(points):
    """Compute inradii for each point in its Voronoi cell."""
    points = np.array(points)
    n = len(points)
    inradii = np.zeros(n)
    
    # Add boundary points to handle edges
    # We need to handle the case where Voronoi cells extend to infinity
    # by adding a large bounding box
    bbox = np.array([
        [-1, -1], [2, -1], [2, 2], [-1, 2],
        [-1, 0.5], [2, 0.5], [0.5, -1], [0.5, 2]
    ])
    all_points = np.vstack([points, bbox])
    
    try:
        vor = Voronoi(all_points)
    except:
        # Fallback: use minimum distance to other points and walls
        inradii = np.zeros(n)
        for i in range(n):
            p = points[i]
            wall_dist = np.min(np.minimum(p, 1 - p))
            diffs = points - p
            diffs[i] = np.inf
            neighbor_dist = np.min(np.sqrt(np.sum(diffs**2, axis=1))) / 2.0
            inradii[i] = min(wall_dist, neighbor_dist)
        return inradii
    
    for i in range(n):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if len(region) == 0 or -1 in region:
            # Unbounded region - use distance to walls
            p = points[i]
            wall_dist = np.min(np.minimum(p, 1 - p))
            inradii[i] = wall_dist
            continue
        
        # Get the vertices of this cell
        vertices = vor.vertices[region]
        
        if len(vertices) == 0:
            inradii[i] = 0.0
            continue
        
        # Compute inradius: minimum distance from point to cell edges
        p = points[i]
        min_dist = np.inf
        
        # Check distance to each edge of the cell
        for j in range(len(vertices)):
            v1 = vertices[j]
            v2 = vertices[(j + 1) % len(vertices)]
            
            # Distance from point to line segment
            # Project point onto line
            edge_vec = v2 - v1
            edge_len_sq = np.dot(edge_vec, edge_vec)
            
            if edge_len_sq < 1e-12:
                dist = np.linalg.norm(p - v1)
            else:
                t = np.dot(p - v1, edge_vec) / edge_len_sq
                t = np.clip(t, 0, 1)
                proj = v1 + t * edge_vec
                dist = np.linalg.norm(p - proj)
            
            min_dist = min(min_dist, dist)
        
        # Also check distance to walls (in case cell extends beyond unit square)
        wall_dist = np.min(np.minimum(p, 1 - p))
        inradii[i] = min(min_dist, wall_dist)
    
    return inradii

def sum_voronoi_inradii(points):
    """Sum of Voronoi inradii."""
    return np.sum(voronoi_inradii(points))

def grad_voronoi_inradii(points):
    """Numerical gradient of sum of Voronoi inradii."""
    points = np.array(points)
    n = len(points)
    grad = np.zeros_like(points)
    eps = 1e-5
    
    base_val = sum_voronoi_inradii(points)
    
    for i in range(n):
        for d in range(2):
            perturbed = points.copy()
            perturbed[i, d] += eps
            val_plus = sum_voronoi_inradii(perturbed)
            
            perturbed = points.copy()
            perturbed[i, d] -= eps
            val_minus = sum_voronoi_inradii(perturbed)
            
            grad[i, d] = (val_plus - val_minus) / (2 * eps)
    
    return grad

def phyllotaxis_points(n, radius=0.5, center=(0.5, 0.5)):
    """Generate points using phyllotaxis arrangement (sunflower pattern)"""
    points = []
    for i in range(n):
        r = radius * np.sqrt(i / n)
        theta = i * 137.508 * np.pi / 180.0
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        x = np.clip(x, 0.02, 0.98)
        y = np.clip(y, 0.02, 0.98)
        points.append([x, y])
    return np.array(points)

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
            p = np.clip(base + jitter, 0.02, 0.98)
            inits.append(p.flatten())
            
    # 5. Random
    for _ in range(10):
        p = np.random.uniform(0.05, 0.95, size=(N, 2))
        inits.append(p.flatten())
        
    bounds = [(0.005, 0.995), (0.005, 0.995)] * N
    
    for idx, c_init in enumerate(inits):
        if time.time() - start_time > timeout - 25:
            break
        
        # L-BFGS-B with numerical gradient
        try:
            def neg_obj(c):
                p = c.reshape(N, 2)
                return -sum_voronoi_inradii(p)
            
            def neg_grad(c):
                p = c.reshape(N, 2)
                return -grad_voronoi_inradii(p)
            
            res = minimize(neg_obj, c_init, jac=neg_grad,
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
        except:
            pass
        
        if time.time() - start_time > timeout - 15:
            break

    if best_c is not None:
        # Final Polish
        if time.time() - start_time < timeout - 10:
            try:
                def neg_obj(c):
                    p = c.reshape(N, 2)
                    return -sum_voronoi_inradii(p)
                
                def neg_grad(c):
                    p = c.reshape(N, 2)
                    return -grad_voronoi_inradii(p)
                
                res = minimize(neg_obj, best_c, jac=neg_grad,
                               method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12})
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_c = res.x.copy()
            except:
                pass

    if best_c is None:
        best_c = create_grid_init(6, 6, N)
        
    p = best_c.reshape(N, 2)
    
    # Compute final radii as minimum of Voronoi inradius and wall distance
    radii = voronoi_inradii(p)
    
    # Ensure circles don't overlap by taking minimum of Voronoi inradius and half-distance to nearest neighbor
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    
    # Use the minimum of all constraints
    final_radii = np.minimum(radii, np.minimum(min_neighbor_dist, min_wall_dist))
    
    result = np.column_stack((p, final_radii))
    return result
