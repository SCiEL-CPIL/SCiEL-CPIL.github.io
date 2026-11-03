"""Circle packing in unit square using Delaunay-based radius computation and force optimization."""
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START

def _compute_radii(centers):
    """
    Given N centers in [0,1]^2, compute the maximum radius for each circle
    such that all circles have disjoint interiors and stay inside [0,1]^2.
    
    r_i = 0.5 * min(
        min_j ||c_i - c_j|| for j != i,
        c_i[0], 1 - c_i[0], c_i[1], 1 - c_i[1]
    )
    """
    n = len(centers)
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Distance to boundaries
    boundary_dists = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    
    # Min neighbor distance
    min_neighbor = np.min(dists, axis=1)
    
    # Radius is half the minimum of neighbor dist and boundary dist
    radii = 0.5 * np.minimum(min_neighbor, boundary_dists)
    
    return radii


def _get_delaunay_edges(centers):
    """Get edges from Delaunay triangulation."""
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges.add(edge)
        return list(edges)
    except Exception:
        # Fallback: all pairs
        return [(i, j) for i in range(N) for j in range(i+1, N)]


def _force_based_step(centers, step_size=0.001, iterations=100):
    """
    Force-based optimization: repulsion between circles and attraction to boundaries.
    Each circle pushes away from others based on overlap, and is pulled toward center.
    """
    centers = centers.copy()
    n = len(centers)
    
    for it in range(iterations):
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    continue
                # Repulsion force proportional to overlap
                r_i = 0.5 * min(np.min([np.linalg.norm(centers[i] - centers[k]) for k in range(n) if k != i]), 
                               centers[i][0], 1 - centers[i][0], centers[i][1], 1 - centers[i][1])
                r_j = 0.5 * min(np.min([np.linalg.norm(centers[j] - centers[k]) for k in range(n) if k != i]), 
                               centers[j][0], 1 - centers[j][0], centers[j][1], 1 - centers[j][1])
                
                overlap = r_i + r_j - dist
                if overlap > 0:
                    force_mag = overlap * 10.0
                    direction = diff / dist
                    forces[i] += force_mag * direction
                    forces[j] -= force_mag * direction
        
        # Boundary repulsion
        for i in range(n):
            # Push away from boundaries
            if centers[i][0] < 0.1:
                forces[i][0] += (0.1 - centers[i][0]) * 5.0
            elif centers[i][0] > 0.9:
                forces[i][0] -= (centers[i][0] - 0.9) * 5.0
            
            if centers[i][1] < 0.1:
                forces[i][1] += (0.1 - centers[i][1]) * 5.0
            elif centers[i][1] > 0.9:
                forces[i][1] -= (centers[i][1] - 0.9) * 5.0
        
        # Apply forces with damping
        centers += step_size * forces
        centers = np.clip(centers, 0.01, 0.99)
        
        # Decay step size
        step_size *= 0.995
    
    return centers


def _objective(centers_flat):
    """Objective: negative sum of radii (to maximize sum of radii)."""
    centers = centers_flat.reshape(-1, 2)
    radii = _compute_radii(centers)
    return -np.sum(radii)


def _gradient(centers_flat):
    """Compute gradient using finite differences."""
    centers = centers_flat.reshape(-1, 2)
    eps = 1e-7
    grad = np.zeros_like(centers_flat)
    
    for i in range(len(centers_flat)):
        orig = centers_flat[i]
        centers_flat[i] = orig + eps
        f_plus = _objective(centers_flat)
        centers_flat[i] = orig - eps
        f_minus = _objective(centers_flat)
        centers_flat[i] = orig
        grad[i] = (f_plus - f_minus) / (2 * eps)
    
    return grad


def _initial_grid(n_points):
    """Create initial grid layout."""
    side = int(np.ceil(np.sqrt(n_points)))
    centers = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= n_points:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            centers.append([x, y])
            count += 1
    return np.array(centers)


def _initial_hexagonal(n_points):
    """Create initial hexagonal packing layout."""
    centers = []
    row = 0
    while len(centers) < n_points:
        offset = 0.0 if row % 2 == 0 else 0.5
        for col in range(6):
            if len(centers) >= n_points:
                break
            x = (col + offset + 0.5) / 6.0
            y = (row + 0.5) / 6.0
            centers.append([x, y])
        row += 1
    return np.array(centers[:n_points])


def _initial_random(n_points, seed=42):
    """Create random initial layout."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0.1, 0.9, size=(n_points, 2))


def construct():
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer
    
    # Multiple initial configurations
    initial_configs = [
        _initial_grid(N),
        _initial_hexagonal(N),
        _initial_random(N, seed=42),
        _initial_random(N, seed=123),
        _initial_random(N, seed=456),
    ]
    
    best_obj = np.inf
    best_centers = None
    
    for config_idx, initial in enumerate(initial_configs):
        if time.time() - start_time > time_limit:
            break
        
        centers = initial.copy()
        centers_flat = centers.flatten().copy()
        
        # Step 1: Force-based pre-optimization
        centers = _force_based_step(centers, step_size=0.002, iterations=50)
        centers_flat = centers.flatten().copy()
        
        # Step 2: L-BFGS-B optimization
        try:
            result = minimize(
                _objective,
                centers_flat,
                method='L-BFGS-B',
                bounds=[(0.001, 0.999)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_centers = result.x.copy()
        except Exception:
            continue
        
        # Step 3: Another force-based refinement
        if time.time() - start_time < time_limit - 10:
            centers = best_centers.reshape(-1, 2) if best_centers is not None else initial
            centers = _force_based_step(centers, step_size=0.001, iterations=30)
            centers_flat = centers.flatten().copy()
            
            try:
                result = minimize(
                    _objective,
                    centers_flat,
                    method='L-BFGS-B',
                    bounds=[(0.0001, 0.9999)] * (2 * N),
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_centers = result.x.copy()
            except Exception:
                pass
    
    # Final refinement if time permits
    if best_centers is not None and time.time() - start_time < time_limit - 5:
        try:
            result = minimize(
                _objective,
                best_centers,
                method='L-BFGS-B',
                bounds=[(0.00001, 0.99999)] * (2 * N),
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_centers = result.x.copy()
        except Exception:
            pass
    
    centers = best_centers.reshape(-1, 2) if best_centers is not None else initial_configs[0]
    radii = _compute_radii(centers)
    
    # Verify validity and clamp
    result = np.column_stack([centers, radii])
    
    # Ensure circles are strictly inside and non-overlapping
    for i in range(N):
        x, y, r = result[i]
        # Clamp radius to ensure inside boundary
        r = min(r, x, 1 - x, y, 1 - y)
        r = max(r, 1e-6)
        result[i, 2] = r
    
    # Verify no overlaps (adjust if needed)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(result[i, :2] - result[j, :2])
            if result[i, 2] + result[j, 2] > d:
                # Scale down radii to ensure no overlap
                scale = d / (result[i, 2] + result[j, 2] + 1e-8)
                result[i, 2] *= scale
                result[j, 2] *= scale
    
    return result

# EVOLVE-BLOCK-END
