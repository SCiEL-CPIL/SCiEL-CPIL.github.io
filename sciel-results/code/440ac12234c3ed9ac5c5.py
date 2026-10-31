"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize

def _precompute_indices(n):
    """Precompute all (i, j, k) index triples for i < j < k."""
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
    return np.array(triples, dtype=np.int32)

_TRIPLES = _precompute_indices(N)

def _min_triangle_area_vec(points):
    """
    Compute minimum triangle area over all C(n,3) triangles.
    points: array of shape (n, 2)
    """
    n = len(points)
    # For each triple (i,j,k), area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    i_idx, j_idx, k_idx = _TRIPLES[:, 0], _TRIPLES[:, 1], _TRIPLES[:, 2]
    pi = points[i_idx]  # (m, 2)
    pj = points[j_idx]  # (m, 2)
    pk = points[k_idx]  # (m, 2)
    # cross product
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    x, y = p
    h = np.sqrt(3) / 2
    eps = 1e-10
    
    y = max(eps, y)
    y = min(y, h - eps)
    
    x_min_left = y / np.sqrt(3)
    x_max_right = 1 - y / np.sqrt(3)
    
    x = max(x_min_left + eps, x)
    x = min(x_max_right - eps, x)
    
    return np.array([x, y])

def _clamp_all(points):
    """Clamp all points to be inside the triangle."""
    return np.array([_clamp_to_triangle(p) for p in points])

def _critical_triangles(points, k=10):
    """
    Find the k smallest area triangles using Delaunay triangulation.
    Returns list of (i, j, k) indices for the critical triangles.
    """
    try:
        # Add a bounding box to ensure Delaunay works
        h = np.sqrt(3) / 2
        # Use Delaunay on the points
        tri = Delaunay(points)
        # Get all triangle indices from Delaunay
        # tri.simplices has shape (num_triangles, 3)
        simplices = tri.simplices
        
        # Compute areas of Delaunay triangles
        areas = np.zeros(len(simplices))
        for idx, tri_idx in enumerate(simplices):
            i, j, k = tri_idx
            pi, pj, pk = points[i], points[j], points[k]
            cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
            areas[idx] = 0.5 * np.abs(cross)
        
        # Get the k smallest
        k = min(k, len(areas))
        smallest_idx = np.argsort(areas)[:k]
        return [tuple(simplices[idx]) for idx in smallest_idx]
    except Exception:
        # Fallback: just use a subset of precomputed triples
        return list(_TRIPLES[:k])

def _objective(points, critical_tris, deadline=None):
    """
    Objective function: maximize the minimum area among critical triangles.
    Returns negative of the min area (for minimization).
    """
    if deadline is not None and time.time() > deadline:
        return 1e10
    
    # Compute areas of critical triangles
    areas = []
    for (i, j, k) in critical_tris:
        pi, pj, pk = points[i], points[j], points[k]
        cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
        areas.append(0.5 * np.abs(cross))
    
    if not areas:
        return 0
    
    return -np.min(areas)

def _local_optimize_critical(points, max_iter=500, deadline=None):
    """
    Local optimization focusing on critical triangles.
    Uses Powell's method for robustness on non-smooth objectives.
    """
    x0 = np.array(points).flatten()
    
    # Get critical triangles
    critical_tris = _critical_triangles(points, k=15)
    
    def objective(x_flat):
        pts = np.array(x_flat).reshape(-1, 2)
        pts = _clamp_all(pts)
        return _objective(pts, critical_tris, deadline)
    
    # Use Powell's method for derivative-free optimization
    result = minimize(
        objective,
        x0,
        method='Powell',
        options={'maxiter': max_iter, 'xtol': 1e-6, 'ftol': 1e-8}
    )
    
    opt_points = np.array(result.x).reshape(-1, 2)
    opt_points = _clamp_all(opt_points)
    return opt_points

def _full_evaluate(points):
    """Full evaluation using all triangles."""
    return _min_triangle_area_vec(points)

def _generate_initial_configs(seed=42):
    """Generate diverse initial configurations."""
    h = np.sqrt(3) / 2
    s3 = np.sqrt(3)
    rng = np.random.RandomState(seed)
    
    configs = []
    
    # Config 1: Symmetric arrangement with vertices
    config1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/3],
        [0.25, h/6],
        [0.75, h/6],
        [0.5, 2*h/3],
        [0.2, h/4],
        [0.8, h/4],
        [0.35, 3*h/8],
        [0.65, 3*h/8],
    ])
    configs.append(config1)
    
    # Config 2: Grid-based
    config2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, 0.0],
        [0.25, h/2],
        [0.75, h/2],
        [0.5, h/3],
        [0.25, h/6],
        [0.75, h/6],
        [0.5, 2*h/3],
        [0.5, h/2],
    ])
    configs.append(config2)
    
    # Config 3: Random with some structure
    config3 = []
    # 3 vertices
    config3.append([0.0, 0.0])
    config3.append([1.0, 0.0])
    config3.append([0.5, h])
    # 8 interior points
    for _ in range(8):
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        x = u + 0.5 * v
        y = h * v
        config3.append([x, y])
    configs.append(np.array(config3))
    
    # Config 4: Hexagonal-like
    config4 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/4],
        [0.3, h/8],
        [0.7, h/8],
        [0.5, 3*h/4],
        [0.25, h/3],
        [0.75, h/3],
        [0.4, 2*h/5],
        [0.6, 2*h/5],
    ])
    configs.append(config4)
    
    # Config 5: Perturbed symmetric
    base = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/3],
        [0.25, h/6],
        [0.75, h/6],
        [0.5, 2*h/3],
        [0.2, h/4],
        [0.8, h/4],
        [0.35, 3*h/8],
        [0.65, 3*h/8],
    ])
    config5 = base + 0.03 * rng.randn(11, 2)
    configs.append(config5)
    
    # Config 6: Another random
    config6 = []
    for _ in range(N):
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        x = u + 0.5 * v
        y = h * v
        config6.append([x, y])
    configs.append(np.array(config6))
    
    return configs

def _simulated_annealing(points, start_temp=0.05, end_temp=0.001, n_iter=2000, deadline=None):
    """
    Simulated annealing to escape local optima.
    """
    current_points = points.copy()
    current_score = _full_evaluate(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    n = len(current_points)
    h = np.sqrt(3) / 2
    
    for i in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Temperature schedule
        t = start_temp * (end_temp / start_temp) ** (i / n_iter)
        
        # Random perturbation
        perturbation = t * np.random.randn(n, 2)
        new_points = current_points + perturbation
        new_points = _clamp_all(new_points)
        
        new_score = _full_evaluate(new_points)
        
        # Accept if better or with probability exp((new - current) / T)
        if new_score >= current_score:
            current_points = new_points
            current_score = new_score
            if new_score > best_score:
                best_points = new_points.copy()
                best_score = new_score
        else:
            accept_prob = np.exp((new_score - current_score) / t)
            if np.random.random() < accept_prob:
                current_points = new_points
                current_score = new_score
    
    return best_points, best_score

def construct():
    """
    Place 11 points on or inside a unit-area equilateral triangle to maximize
    the smallest area of a triangle formed by three points.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    h = np.sqrt(3) / 2
    
    # Generate initial configurations
    initial_configs = _generate_initial_configs(seed=42)
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Optimize each initial config with critical triangle approach
    for i, config in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        points = _clamp_all(config)
        current_score = _full_evaluate(points)
        
        if current_score > best_score:
            best_points = points.copy()
            best_score = current_score
        
        # Local optimization with critical triangles
        if time.time() < deadline - 20:
            opt_points = _local_optimize_critical(points, max_iter=300, deadline=deadline - 15)
            opt_score = _full_evaluate(opt_points)
            if opt_score > best_score:
                best_points = opt_points
                best_score = opt_score
    
    # Phase 2: Simulated annealing from best point
    if time.time() < deadline - 25:
        sa_points, sa_score = _simulated_annealing(
            best_points, 
            start_temp=0.03, 
            end_temp=0.0005, 
            n_iter=3000,
            deadline=deadline - 15
        )
        if sa_score > best_score:
            best_points = sa_points
            best_score = sa_score
    
    # Phase 3: Final local optimization
    if time.time() < deadline - 10:
        final_points = _local_optimize_critical(best_points, max_iter=500, deadline=deadline - 5)
        final_score = _full_evaluate(final_points)
        if final_score > best_score:
            best_points = final_points
            best_score = final_score
    
    # Ensure all points are inside the triangle
    if best_points is None:
        best_points = _clamp_all(initial_configs[0])
    else:
        best_points = _clamp_all(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
