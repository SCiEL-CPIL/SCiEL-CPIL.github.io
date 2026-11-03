"""Voronoi-based circle packing optimization."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay, Voronoi
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

def _get_delaunay_neighbors(centers):
    """Get Delaunay neighbor list for each point."""
    try:
        tri = Delaunay(centers)
        neighbors = [set() for _ in range(N)]
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    neighbors[a].add(b)
                    neighbors[b].add(a)
        return [list(neigh) for neigh in neighbors]
    except Exception:
        return [[j for j in range(N) if j != i] for i in range(N)]

def _voronoi_inradius(centers, neighbors):
    """
    Compute the inradius for each site based on its Delaunay neighbors and walls.
    The inradius is the minimum distance to any constraint (neighbor bisector or wall).
    """
    inradii = np.zeros(N)
    for i in range(N):
        ci = centers[i]
        min_dist = 0.5 - np.abs(ci).max()  # wall constraint
        
        for j in neighbors[i]:
            d = np.linalg.norm(ci - centers[j]) / 2.0
            if d < min_dist:
                min_dist = d
        
        inradii[i] = min_dist
    return inradii

def _optimize_voronoi(start_centers, bounds, max_time=15.0):
    """
    Optimize using Voronoi-aware objective.
    """
    best_centers = start_centers.copy()
    best_score = _calculate_true_radius(best_centers)
    start_time = time.time()
    
    def objective(params):
        centers = params.reshape(N, 2)
        centers = np.clip(centers, -0.4999, 0.4999)
        
        neighbors = _get_delaunay_neighbors(centers)
        inradii = _voronoi_inradius(centers, neighbors)
        
        # Soft-min to maximize the minimum inradius
        k = 100.0
        m = np.min(inradii)
        shifted = inradii - m
        exp_terms = np.exp(-k * shifted)
        log_sum = np.log(np.sum(exp_terms))
        soft_min = m - (1.0 / k) * log_sum
        return -soft_min
    
    def objective_grad(params):
        """Numerical gradient for L-BFGS-B."""
        centers = params.reshape(N, 2)
        centers = np.clip(centers, -0.4999, 0.4999)
        
        neighbors = _get_delaunay_neighbors(centers)
        inradii = _voronoi_inradius(centers, neighbors)
        
        k = 100.0
        m = np.min(inradii)
        shifted = inradii - m
        exp_terms = np.exp(-k * shifted)
        log_sum = np.log(np.sum(exp_terms))
        soft_min = m - (1.0 / k) * log_sum
        
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(params)
        for idx in range(len(params)):
            params_plus = params.copy()
            params_plus[idx] += eps
            val_plus = objective(params_plus)
            params_minus = params.copy()
            params_minus[idx] -= eps
            val_minus = objective(params_minus)
            grad[idx] = (val_plus - val_minus) / (2 * eps)
        
        return grad
    
    # Try optimization
    try:
        result = minimize(
            objective,
            best_centers.reshape(-1),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-15, 'disp': False,
                     'maxfun': 20000}
        )
        centers = result.x.reshape(N, 2)
        centers = np.clip(centers, -0.4999, 0.4999)
        true_r = _calculate_true_radius(centers)
        
        if true_r > best_score:
            best_score = true_r
            best_centers = centers
    except Exception:
        pass
    
    # Polishing pass with tighter tolerance
    if time.time() - start_time < max_time - 2.0:
        try:
            result = minimize(
                objective,
                best_centers.reshape(-1),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16, 'disp': False,
                         'maxfun': 15000}
            )
            centers = result.x.reshape(N, 2)
            centers = np.clip(centers, -0.4999, 0.4999)
            true_r = _calculate_true_radius(centers)
            
            if true_r > best_score:
                best_score = true_r
                best_centers = centers
        except Exception:
            pass
    
    return best_centers, best_score

def _generate_hex_init(spacing=0.14):
    """Generate a hexagonal grid initialization."""
    points = []
    range_x = np.arange(-0.8, 0.8, spacing)
    range_y = np.arange(-0.8, 0.8, spacing)
    for i, x in enumerate(range_x):
        offset = (spacing / 2) if i % 2 == 0 else 0
        for y in range_y:
            points.append((x + offset, y))
    points = np.array(points)
    mask = (np.abs(points[:, 0]) <= 0.49) & (np.abs(points[:, 1]) <= 0.49)
    points = points[mask]
    return points

def construct():
    bounds = [(-0.4999, 0.4999)] * (2 * N)
    
    inits = []
    
    # 1. Hexagonal grid (tight)
    hex_points = _generate_hex_init(0.135)
    if len(hex_points) >= N:
        np.random.seed(42)
        indices = np.random.choice(len(hex_points), N, replace=False)
        inits.append(hex_points[indices])
    
    # 2. Hexagonal grid (looser)
    hex_points2 = _generate_hex_init(0.145)
    if len(hex_points2) >= N:
        np.random.seed(43)
        indices = np.random.choice(len(hex_points2), N, replace=False)
        inits.append(hex_points2[indices])
    
    # 3. Square grid
    grid_x = np.linspace(-0.45, 0.45, 7)
    grid_y = np.linspace(-0.45, 0.45, 7)
    grid_points = np.array([[x, y] for x in grid_x for y in grid_y])
    if len(grid_points) >= N:
        np.random.seed(123)
        indices = np.random.choice(len(grid_points), N, replace=False)
        inits.append(grid_points[indices])
    
    # 4. Jittered hex
    if len(hex_points) >= N:
        np.random.seed(44)
        indices = np.random.choice(len(hex_points), N, replace=False)
        jittered = hex_points[indices] + np.random.normal(0, 0.015, (N, 2))
        jittered = np.clip(jittered, -0.49, 0.49)
        inits.append(jittered)
    
    # 5. Random uniform
    np.random.seed(999)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # 6. Another random with different seed
    np.random.seed(1001)
    inits.append(np.random.uniform(-0.44, 0.44, (N, 2)))
    
    # 7. Concentric rings
    np.random.seed(777)
    ring_points = []
    for ring in range(5):
        r = 0.05 + ring * 0.1
        n_on_ring = 8 + ring * 4
        angles = np.linspace(0, 2*np.pi, n_on_ring, endpoint=False)
        for a in angles:
            ring_points.append([r * np.cos(a), r * np.sin(a)])
    ring_points = np.array(ring_points)
    if len(ring_points) >= N:
        np.random.seed(778)
        indices = np.random.choice(len(ring_points), N, replace=False)
        inits.append(ring_points[indices])
    
    # 8. Fibonacci sphere projected
    np.random.seed(888)
    fib_points = []
    golden = (1 + np.sqrt(5)) / 2
    for i in range(N):
        z = 1 - 2 * i / (N - 1)
        r = np.sqrt(1 - z**2)
        theta = 2 * np.pi * i / golden
        fib_points.append([0.45 * r * np.cos(theta), 0.45 * z])
    inits.append(np.array(fib_points))
    
    best_centers = None
    best_score = -1.0
    
    start_time = time.time()
    
    for i, init in enumerate(inits):
        if time.time() - start_time > 105.0:
            break
            
        centers, score = _optimize_voronoi(init, bounds, max_time=12.0)
        if score > best_score:
            best_score = score
            best_centers = centers
    
    if best_centers is None:
        best_centers = np.random.uniform(-0.45, 0.45, (N, 2))
    
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
