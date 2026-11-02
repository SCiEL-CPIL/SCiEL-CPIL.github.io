"""Heilbronn 11-point problem: maximize minimum triangle area in unit equilateral triangle.

Strategy: Direct Active-Set Local Search.
1. Initialize with a heuristic grid/hexagonal pattern.
2. Iteratively:
   a. Identify active constraints (triangles with area close to the current minimum).
   b. Solve a local quadratic subproblem to maximize the minimum area of active triangles.
   c. Project back to the feasible region (triangle).
   d. Repeat until convergence or time limit.
"""
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
TRI_AREA = 0.5 * 1.0 * S3H

# Precompute all (i,j,k) triples
_TRIPLES = list(combinations(range(N), 3))
_N_TRIPLES = len(_TRIPLES)  # 165

# Convert triple indices to flat indices for vectorized area computation
_I_IDX = np.array([t[0] for t in _TRIPLES])
_J_IDX = np.array([t[1] for t in _TRIPLES])
_K_IDX = np.array([t[2] for t in _TRIPLES])


def _all_areas(points):
    """Compute all triangle areas. points: (N,2) array."""
    p_i = points[_I_IDX]
    p_j = points[_J_IDX]
    p_k = points[_K_IDX]
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    return 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)


def _min_area(points):
    """Compute minimum triangle area among all triples."""
    return np.min(_all_areas(points))


def _project_to_triangle(points):
    """
    Project points to be inside the equilateral triangle with vertices (0,0), (1,0), (1/2, sqrt(3)/2).
    Uses barycentric coordinates.
    """
    pts = points.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        # Convert to barycentric: p = u * (1,0) + v * (0.5, sqrt(3)/2) + w * (0,0)
        # x = u + 0.5*v
        # y = sqrt(3)/2 * v  => v = y / (sqrt(3)/2)
        v = y / S3H
        u = x - 0.5 * v
        w = 1.0 - u - v
        
        # Project (u,v,w) to simplex u>=0, v>=0, w>=0
        if u < 0:
            u = 0
        if v < 0:
            v = 0
        if u + v > 1:
            s = u + v
            if s > 0:
                u /= s
                v /= s
            else:
                u = 0
                v = 0
        
        pts[i, 0] = u + 0.5 * v
        pts[i, 1] = S3H * v
    return pts


def _active_set_subproblem(points, active_indices, max_iter=50):
    """
    Solve a local subproblem to maximize the minimum area of triangles in active_indices.
    Uses L-BFGS-B to minimize the negative of the minimum area of the active set,
    but with a smooth approximation (log-sum-exp) to handle non-differentiability.
    """
    n_active = len(active_indices)
    if n_active == 0:
        return points

    # Extract active triples
    active_i = _I_IDX[active_indices]
    active_j = _J_IDX[active_indices]
    active_k = _K_IDX[active_indices]

    def objective(x):
        pts = np.empty((N, 2))
        pts[:, 0] = x[0::2]
        pts[:, 1] = x[1::2]
        
        # Compute areas for active triangles
        p_i = pts[active_i]
        p_j = pts[active_j]
        p_k = pts[active_k]
        dx1 = p_j[:, 0] - p_i[:, 0]
        dy1 = p_j[:, 1] - p_i[:, 1]
        dx2 = p_k[:, 0] - p_i[:, 0]
        dy2 = p_k[:, 1] - p_i[:, 1]
        areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
        
        # Smooth minimum: -log(sum(exp(-areas/tau))) * tau
        tau = 1e-3
        log_sum = np.log(np.sum(np.exp(-areas / tau)))
        return tau * log_sum

    x0 = np.column_stack([points[:, 0], points[:, 1]]).flatten()
    
    try:
        res = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res.x is not None:
            new_pts = np.empty((N, 2))
            new_pts[:, 0] = res.x[0::2]
            new_pts[:, 1] = res.x[1::2]
            return _project_to_triangle(new_pts)
    except Exception:
        pass
        
    return points


def _hexagonal_init():
    """
    Generate an initial configuration based on a hexagonal lattice pattern.
    This is known to be a good starting point for Heilbronn problems.
    """
    pts = np.zeros((N, 2))
    
    # Approximate hexagonal packing in a triangle
    # Side length of small hexagons/triangles
    # We can use a recursive subdivision or a fixed grid.
    # For N=11, a common good config is roughly:
    # 1 point at center, 6 around it, 4 at corners? 
    # Let's use a simple grid sampling and pick the best subset or just a heuristic layout.
    
    # Heuristic: Place points at vertices and centroids of sub-triangles.
    # Divide triangle into 9 smaller triangles (3x3 grid).
    # Vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Midpoints: (0.5, 0), (0.25, sqrt(3)/4), (0.75, sqrt(3)/4)
    # Centroid: (0.5, sqrt(3)/6)
    
    # Let's try a specific known good configuration for N=11.
    # Reference: "The Heilbronn triangle problem" by various authors.
    # A known local optimum for N=11 has min area ~ 0.0218 (normalized to 1).
    
    # Let's use a random restart with a few good seeds.
    rng = np.random.RandomState(42)
    
    # Seed 1: Symmetric
    # Corners
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 0.0])
    p2 = np.array([0.5, S3H])
    
    # Edge midpoints
    p3 = np.array([0.5, 0.0])
    p4 = np.array([0.25, S3H/2])
    p5 = np.array([0.75, S3H/2])
    
    # Inner hexagon
    # Center
    p6 = np.array([0.5, S3H/3])
    
    # Other inner points
    # Let's just use a random initialization for robustness, but guided by symmetry.
    # Actually, let's use a simple grid search for the initial 11 points.
    
    # A better init: Use the result of a quick differential evolution or random search.
    # But we want to be fast.
    
    # Let's use a "spiral" or "grid" init.
    # Grid: 4x4 grid in barycentric coords?
    # Let's just return a random valid configuration. The active set search should fix it.
    
    # However, a good init helps. Let's try a few random starts and keep the best.
    best_pts = None
    best_score = -np.inf
    
    for _ in range(20):
        # Random points in triangle
        u = rng.random(N)
        v = rng.random(N)
        # Ensure u+v <= 1
        # Sample from simplex
        w = rng.random(N)
        s = u + v + w
        u = u / s
        v = v / s
        
        pts = np.empty((N, 2))
        pts[:, 0] = u + 0.5 * v
        pts[:, 1] = S3H * v
        
        score = _min_area(pts)
        if score > best_score:
            best_score = score
            best_pts = pts
            
    return best_pts


def construct():
    start_time = time.time()
    TIME_LIMIT = 140  # Safety margin for 150s limit
    
    def _time_up():
        return (time.time() - start_time) > TIME_LIMIT
    
    # Stage 1: Initialization
    points = _hexagonal_init()
    best_points = points.copy()
    best_score = _min_area(best_points)
    
    # Stage 2: Active Set Local Search
    # We iterate: find active set, solve subproblem, update.
    max_outer_iter = 50
    active_threshold_factor = 1.05  # Consider triangles with area < 1.05 * min_area as active
    
    for outer_iter in range(max_outer_iter):
        if _time_up():
            break
            
        areas = _all_areas(points)
        min_area = np.min(areas)
        
        # Find active constraints: triangles with area close to the minimum
        # This creates a smooth surrogate locally
        active_mask = areas < (min_area * active_threshold_factor)
        active_indices = np.where(active_mask)[0]
        
        if len(active_indices) == 0:
            break
            
        # Solve subproblem for active set
        new_points = _active_set_subproblem(points, active_indices, max_iter=30)
        
        # Check if we improved
        new_score = _min_area(new_points)
        
        if new_score > best_score + 1e-12:
            best_score = new_score
            best_points = new_points.copy()
            points = new_points
        else:
            # If no improvement, try a slightly different active set or perturb
            # Relax the threshold or add noise
            points = _project_to_triangle(points + np.random.normal(0, 0.01, (N, 2)))
            
            # If stuck, try a smaller active set (just the single min)
            min_idx = np.argmin(areas)
            new_points = _active_set_subproblem(points, [min_idx], max_iter=50)
            new_score = _min_area(new_points)
            if new_score > best_score + 1e-12:
                best_score = new_score
                best_points = new_points.copy()
                points = new_points
            else:
                # Break if no progress for a while?
                # For now, just continue with perturbation
                pass
                
    # Stage 3: Final Polish with Global Active Set (all 165)
    # This is expensive but can find the global min in the local neighborhood
    if not _time_up():
        # Use all triangles as active? No, that's just the original problem.
        # Instead, do a final L-BFGS-B on the log-sum-exp of ALL areas with small tau.
        def final_objective(x):
            pts = np.empty((N, 2))
            pts[:, 0] = x[0::2]
            pts[:, 1] = x[1::2]
            
            areas = _all_areas(pts)
            tau = 1e-4
            log_sum = np.log(np.sum(np.exp(-areas / tau)))
            return tau * log_sum
            
        x0 = np.column_stack([best_points[:, 0], best_points[:, 1]]).flatten()
        try:
            res = minimize(
                final_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.x is not None:
                new_pts = np.empty((N, 2))
                new_pts[:, 0] = res.x[0::2]
                new_pts[:, 1] = res.x[1::2]
                new_pts = _project_to_triangle(new_pts)
                new_score = _min_area(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_pts
        except Exception:
            pass

    return best_points
