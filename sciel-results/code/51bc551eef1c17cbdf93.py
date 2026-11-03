"""Voronoi-informed circle packing optimization."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi, Delaunay
import time

N = 50

def _compute_radius(centers):
    """Compute the maximum inscribed radius."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _voronoi_active_constraints(centers):
    """
    Identify active constraints using Voronoi/Delaunay diagram.
    Returns list of (type, indices) where type is 'pair' or 'wall'.
    """
    try:
        # Get Delaunay edges for pair constraints
        active_pairs = []
        try:
            tri = Delaunay(centers)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        a, b = simplex[i], simplex[j]
                        if a > b:
                            a, b = b, a
                        edges.add((a, b))
            active_pairs = list(edges)
        except Exception:
            # Fallback: use distance threshold
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(centers[i] - centers[j])
                    if d < 0.15:
                        active_pairs.append((i, j))
        
        # Wall constraints: points near boundaries
        active_walls = []
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist = np.min(wall_dists)
        for i in range(N):
            for k in range(2):
                if wall_dists[i, k] < min_wall_dist * 1.15:
                    active_walls.append((i, k))
        
        return active_pairs, active_walls
    
    except Exception:
        # Fallback: use distance-based heuristics
        active_pairs = []
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < 0.15:
                    active_pairs.append((i, j))
        
        active_walls = []
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist = np.min(wall_dists)
        for i in range(N):
            for k in range(2):
                if wall_dists[i, k] < min_wall_dist * 1.15:
                    active_walls.append((i, k))
        
        return active_pairs, active_walls

def _sparse_objective(params, active_pairs, active_walls, k):
    """
    Sparse objective using only active constraints.
    """
    centers = params.reshape(N, 2)
    centers = np.clip(centers, -0.5, 0.5)
    
    values = []
    
    # Pair distances
    for i, j in active_pairs:
        d = np.linalg.norm(centers[i] - centers[j]) / 2.0
        values.append(d)
    
    # Wall distances
    for i, k in active_walls:
        values.append(0.5 - np.abs(centers[i, k]))
    
    if len(values) == 0:
        return 0.0
    
    values = np.array(values)
    m = np.min(values)
    shifted = values - m
    exp_terms = np.exp(-k * shifted)
    log_sum = np.log(np.sum(exp_terms) + 1e-300)
    soft_min = m - (1.0 / k) * log_sum
    return -soft_min

def _refine(centers, bounds, max_iter=3000):
    """
    Multi-phase refinement using Voronoi-aware sparse objectives.
    """
    best_centers = centers.copy()
    best_score = _compute_radius(best_centers)
    
    phases = [
        (200, 1500),
        (400, 1500),
        (2000, 2000),
    ]
    
    for k, maxit in phases:
        # Recompute active constraints
        active_pairs, active_walls = _voronoi_active_constraints(best_centers)
        
        def obj(params):
            return _sparse_objective(params, active_pairs, active_walls, k)
        
        try:
            result = minimize(
                obj,
                best_centers.reshape(-1),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxit, 'ftol': 1e-15, 'gtol': 1e-14, 'disp': False}
            )
            new_centers = result.x.reshape(N, 2)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            new_score = _compute_radius(new_centers)
            
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers
        except Exception:
            pass
    
    return best_centers, best_score

def _generate_initial_configs():
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. Hexagonal lattice
    s = 0.14
    pts = []
    for i, x in enumerate(np.arange(-0.7, 0.7, s)):
        offset = s/2 if i % 2 == 0 else 0
        for y in np.arange(-0.7, 0.7, s):
            pts.append([x + offset, y])
    pts = np.array(pts)
    mask = (np.abs(pts[:,0]) <= 0.49) & (np.abs(pts[:,1]) <= 0.49)
    pts = pts[mask]
    if len(pts) >= N:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(pts), N, replace=False)
        inits.append(pts[idx])
    
    # 2. Square grid with jitter
    gx = np.linspace(-0.45, 0.45, 7)
    gy = np.linspace(-0.45, 0.45, 7)
    grid = np.array([[x, y] for x in gx for y in gy])
    if len(grid) >= N:
        rng = np.random.RandomState(123)
        idx = rng.choice(len(grid), N, replace=False)
        jittered = grid[idx] + rng.normal(0, 0.01, (N, 2))
        inits.append(np.clip(jittered, -0.49, 0.49))
    
    # 3. Pure random
    rng = np.random.RandomState(999)
    inits.append(rng.uniform(-0.45, 0.45, (N, 2)))
    
    # 4. Randomized grid
    rng = np.random.RandomState(555)
    gx_r = np.sort(rng.uniform(-0.49, 0.49, 7))
    gy_r = np.sort(rng.uniform(-0.49, 0.49, 7))
    grid_r = np.array([[x, y] for x in gx_r for y in gy_r])
    if len(grid_r) >= N:
        idx = rng.choice(len(grid_r), N, replace=False)
        inits.append(grid_r[idx])
    
    # 5. Another random
    rng = np.random.RandomState(1000)
    inits.append(rng.uniform(-0.45, 0.45, (N, 2)))
    
    # 6. Concentric rings
    rng = np.random.RandomState(777)
    pts = []
    for ring in range(6):
        r = 0.05 + ring * 0.08
        n_pts = max(1, int(2 * np.pi * r / 0.12))
        angles = rng.uniform(0, 2*np.pi, n_pts)
        for a in angles:
            pts.append([r * np.cos(a), r * np.sin(a)])
    pts = np.array(pts)
    mask = (np.abs(pts[:,0]) <= 0.49) & (np.abs(pts[:,1]) <= 0.49)
    pts = pts[mask]
    if len(pts) >= N:
        idx = rng.choice(len(pts), N, replace=False)
        inits.append(pts[idx])
    
    return inits

def construct():
    bounds = [(-0.5, 0.5)] * (2 * N)
    inits = _generate_initial_configs()
    
    best_centers = None
    best_score = -1.0
    start_time = time.time()
    
    for init in inits:
        if time.time() - start_time > 105.0:
            break
        
        centers, score = _refine(init.copy(), bounds)
        if score > best_score:
            best_score = score
            best_centers = centers
    
    if best_centers is None:
        best_centers = np.random.uniform(-0.45, 0.45, (N, 2))
    
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    return best_centers
