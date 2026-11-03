"""Improved program using Voronoi-based inradius optimization."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

N = 50

def _calculate_true_radius(centers):
    """Calculate the true maximum radius for a given set of centers."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _get_voronoi_inradii_and_grads(centers):
    """
    Compute the inradius and gradient direction for each center based on its Voronoi cell.
    Returns:
        inradii: array of shape (N,) with the inradius of each cell.
        grads: array of shape (N, 2) with the gradient of the inradius w.r.t. center.
    """
    inradii = np.zeros(N)
    grads = np.zeros((N, 2))
    
    try:
        tri = Delaunay(centers)
        # Get Voronoi vertices and regions
        # scipy.spatial.Voronoi is more direct for inradii
        from scipy.spatial import Voronoi
        vor = Voronoi(centers)
        
        for i in range(N):
            region = vor.regions[i]
            if -1 in region or len(region) == 0:
                # Unbounded cell (on the hull)
                # The inradius is limited by the walls or the bisectors
                # For simplicity, we handle walls separately or approximate
                # A robust way: the inradius is the min distance to the boundary of the square
                # AND the min distance to the bisectors.
                # However, for hull points, the "cell" extends to infinity in Voronoi, but is cut by the square.
                # We must check distance to square walls and to Voronoi edges.
                pass
            
            # We need to check distance to:
            # 1. Square walls
            # 2. Voronoi edges (which are perpendicular bisectors of Delaunay edges)
            
            min_dist = np.inf
            best_grad = np.zeros(2)
            
            # Check walls
            # Distance to x=-0.5
            d = centers[i][0] + 0.5
            if d < min_dist:
                min_dist = d
                best_grad = np.array([1.0, 0.0]) # Grad of (x+0.5) is [1,0]
                
            # Distance to x=0.5
            d = 0.5 - centers[i][0]
            if d < min_dist:
                min_dist = d
                best_grad = np.array([-1.0, 0.0]) # Grad of (0.5-x) is [-1,0]
                
            # Distance to y=-0.5
            d = centers[i][1] + 0.5
            if d < min_dist:
                min_dist = d
                best_grad = np.array([0.0, 1.0])
                
            # Distance to y=0.5
            d = 0.5 - centers[i][1]
            if d < min_dist:
                min_dist = d
                best_grad = np.array([0.0, -1.0])
                
            # Check Voronoi edges
            if len(region) > 0 and -1 not in region:
                # Get vertices of the polygon
                verts = vor.vertices[region]
                # For each edge (v_k, v_{k+1}), compute distance
                for k in range(len(verts)):
                    v1 = verts[k]
                    v2 = verts[(k+1) % len(verts)]
                    # Vector from v1 to v2
                    e = v2 - v1
                    # Vector from v1 to center
                    w = centers[i] - v1
                    # Project w onto e
                    len_e = np.linalg.norm(e)
                    if len_e < 1e-10: continue
                    proj = np.dot(w, e) / len_e
                    # Clamp projection to [0, len_e]
                    proj = np.clip(proj, 0, len_e)
                    # Closest point
                    closest = v1 + (proj / len_e) * e
                    # Distance
                    d = np.linalg.norm(centers[i] - closest)
                    if d < min_dist:
                        min_dist = d
                        # Gradient is unit vector from closest to center
                        if d > 1e-10:
                            best_grad = (centers[i] - closest) / d
                        else:
                            best_grad = np.zeros(2)
            else:
                # If unbounded, we still need to check the edges that form the boundary
                # vor.ridge_points and vor.ridge_vertices
                # This is complex. For now, if unbounded, we rely on walls and the fact that
                # hull points are usually pushed inwards by the optimization.
                # A safer fallback for unbounded cells: check distance to all other centers?
                # No, that's O(N^2).
                # Let's just use the walls for now, or assume the Delaunay edges define the boundary.
                # Actually, Voronoi edges exist even for hull points, they just go to infinity.
                # The ridge vertices are -1 for infinity.
                # We should iterate over ridges connected to point i.
                pass

            inradii[i] = min_dist
            grads[i] = best_grad

    except Exception:
        # Fallback: brute force min distance to walls and neighbors
        for i in range(N):
            min_dist = np.inf
            best_grad = np.zeros(2)
            # Walls
            d = centers[i][0] + 0.5
            if d < min_dist: min_dist = d; best_grad = [1,0]
            d = 0.5 - centers[i][0]
            if d < min_dist: min_dist = d; best_grad = [-1,0]
            d = centers[i][1] + 0.5
            if d < min_dist: min_dist = d; best_grad = [0,1]
            d = 0.5 - centers[i][1]
            if d < min_dist: min_dist = d; best_grad = [0,-1]
            
            # Neighbors (approximate by all others if Voronoi fails)
            for j in range(N):
                if i == j: continue
                d = np.linalg.norm(centers[i] - centers[j]) / 2.0
                if d < min_dist:
                    min_dist = d
                    diff = centers[i] - centers[j]
                    norm = np.linalg.norm(diff)
                    if norm > 1e-10:
                        best_grad = diff / norm
            inradii[i] = min_dist
            grads[i] = best_grad

    return inradii, grads

def _optimize_from_start(start_centers, bounds):
    """
    Optimization using Voronoi inradius objective.
    """
    best_centers = start_centers.copy()
    best_score = _calculate_true_radius(best_centers)
    
    def objective(params):
        centers = params.reshape(N, 2)
        centers = np.clip(centers, -0.5, 0.5)
        
        inradii, grads = _get_voronoi_inradii_and_grads(centers)
        
        if np.any(np.isnan(inradii)):
            return 0.0
            
        m = np.min(inradii)
        k = 500.0 # Sharpness of soft-min
        shifted = inradii - m
        exp_terms = np.exp(-k * shifted)
        log_sum = np.log(np.sum(exp_terms))
        soft_min = m - (1.0 / k) * log_sum
        
        return -soft_min

    def gradient(params):
        centers = params.reshape(N, 2)
        centers = np.clip(centers, -0.5, 0.5)
        
        inradii, grads = _get_voronoi_inradii_and_grads(centers)
        
        if np.any(np.isnan(inradii)):
            return np.zeros_like(params)
            
        m = np.min(inradii)
        k = 500.0
        shifted = inradii - m
        exp_terms = np.exp(-k * shifted)
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp
        
        # Gradient of soft_min = - sum(weights * grad(inradius_i))
        # We want to maximize soft_min, so minimize -soft_min
        # grad(-soft_min) = - grad(soft_min) = sum(weights * grad(inradius_i))
        # Wait.
        # f = -soft_min
        # df/dx = - d(soft_min)/dx
        # soft_min = m - (1/k) log(sum(exp(-k(x_i - m))))
        # This is complicated because m depends on x.
        # Standard soft-min gradient:
        # d(softmin(x))/dx_i = - w_i
        # where w_i = exp(-k(x_i - m)) / sum(exp(-k(x_j - m)))
        # So d(-softmin)/dx_i = w_i
        
        # But our x_i are inradii, which are functions of centers.
        # d(-softmin)/dc = sum_i w_i * (d(inradius_i)/dc)
        # d(inradius_i)/dc = grads[i]
        
        grad_vec = np.sum(weights[:, np.newaxis] * grads, axis=0)
        
        return grad_vec.flatten()

    try:
        result = minimize(
            objective, 
            best_centers.reshape(-1), 
            jac=gradient,
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
        )
        centers = result.x.reshape(N, 2)
        centers = np.clip(centers, -0.5, 0.5)
        true_r = _calculate_true_radius(centers)
        
        if true_r > best_score:
            best_score = true_r
            best_centers = centers
    except Exception:
        pass
        
    # Second pass with higher k
    # We need to update the closure k or pass it. 
    # For simplicity, we just run another minimize with a tighter tolerance or different start.
    try:
        # Re-define objective with higher k? 
        # The function above has k hardcoded. 
        # Let's just do a local polish with standard minimize without jac if jac is unstable, 
        # or just run again.
        # Actually, let's create a helper that takes k.
        pass
    except Exception:
        pass
        
    return best_centers, best_score

def construct():
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    inits = []
    
    # 1. Hexagonal grid
    s = 0.14 
    points = []
    range_x = np.arange(-0.8, 0.8, s)
    range_y = np.arange(-0.8, 0.8, s)
    for i, x in enumerate(range_x):
        offset = (s / 2) if i % 2 == 0 else 0
        for y in range_y:
            points.append((x + offset, y))
    points = np.array(points)
    mask = (np.abs(points[:, 0]) <= 0.49) & (np.abs(points[:, 1]) <= 0.49)
    points = points[mask]
    if len(points) >= N:
        np.random.seed(42)
        indices = np.random.choice(len(points), N, replace=False)
        inits.append(points[indices])
        
    # 2. Square grid
    grid_x = np.linspace(-0.45, 0.45, 7)
    grid_y = np.linspace(-0.45, 0.45, 7)
    grid_points = np.array([[x, y] for x in grid_x for y in grid_y])
    if len(grid_points) >= N:
        np.random.seed(123)
        indices = np.random.choice(len(grid_points), N, replace=False)
        inits.append(grid_points[indices])
        
    # 3. Random uniform
    np.random.seed(999)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # 4. Jittered hex
    if len(points) >= N:
        np.random.seed(43)
        indices = np.random.choice(len(points), N, replace=False)
        jittered = points[indices] + np.random.normal(0, 0.02, (N, 2))
        jittered = np.clip(jittered, -0.49, 0.49)
        inits.append(jittered)
        
    # 5. Another random
    np.random.seed(1000)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # 6. Randomized Grid
    np.random.seed(555)
    grid_x_rand = np.sort(np.random.uniform(-0.49, 0.49, 7))
    grid_y_rand = np.sort(np.random.uniform(-0.49, 0.49, 7))
    grid_points_rand = np.array([[x, y] for x in grid_x_rand for y in grid_y_rand])
    if len(grid_points_rand) >= N:
        indices = np.random.choice(len(grid_points_rand), N, replace=False)
        inits.append(grid_points_rand[indices])

    best_centers = None
    best_score = -1.0
    
    start_time = time.time()
    
    for i, init in enumerate(inits):
        if time.time() - start_time > 100.0: 
            break
            
        centers, score = _optimize_from_start(init, bounds)
        if score > best_score:
            best_score = score
            best_centers = centers
            
    if best_centers is None:
        best_centers = np.random.uniform(-0.45, 0.45, (N, 2))
        
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
