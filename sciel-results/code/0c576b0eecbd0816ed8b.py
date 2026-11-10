import numpy as np
import time
from scipy.spatial import ConvexHull
import math

N = 130

def _compute_covering_radius_and_centers(points):
    """
    Computes the covering radius (max circumradius of hull facets) 
    and returns the radius, the circumcenter of the worst facet, 
    the index of the worst facet, and the hull object.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0, None, -1, None

    max_radius = 0.0
    worst_center = None
    worst_facet_idx = -1

    # Precompute dot products for all simplices to speed up? 
    # For N=130, ~256 facets. Direct calculation is fine.
    
    for i, simplex in enumerate(hull.simplices):
        u = points[simplex[0]]
        v = points[simplex[1]]
        w = points[simplex[2]]
        
        uv = np.dot(u, v)
        uw = np.dot(u, w)
        vw = np.dot(v, w)
        
        M = np.array([
            [1.0, uv, uw],
            [uv, 1.0, vw],
            [uw, vw, 1.0]
        ])
        
        try:
            x = np.linalg.solve(M, np.ones(3))
        except np.linalg.LinAlgError:
            continue
            
        xT_M_x = np.dot(x, M @ x)
        
        if xT_M_x <= 1e-10:
            continue
            
        cos_R = 1.0 / np.sqrt(xT_M_x)
        cos_R = np.clip(cos_R, -1.0, 1.0)
        
        C = x[0] * u + x[1] * v + x[2] * w
        norm_C = np.linalg.norm(C)
        if norm_C < 1e-10:
            continue
        C = C / norm_C
        
        R = math.acos(cos_R)
        
        if R > max_radius:
            max_radius = R
            worst_center = C
            worst_facet_idx = i

    return math.degrees(max_radius), worst_center, worst_facet_idx, hull

def _move_point_towards(p, c, step_size):
    """Move point p on the sphere towards c by step_size radians."""
    dot_pc = np.dot(p, c)
    # Tangent vector from p to c
    t = c - dot_pc * p
    norm_t = np.linalg.norm(t)
    if norm_t < 1e-10:
        return p
    t = t / norm_t
    new_p = p + step_size * t
    new_p = new_p / np.linalg.norm(new_p)
    return new_p

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # 1. Initialization: Fibonacci spiral with deterministic jitter
    np.random.seed(42)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Add small random jitter to break symmetries
    noise = np.random.randn(N, 3) * 0.01
    points = points + noise
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    best_points = points.copy()
    best_radius, _, _, _ = _compute_covering_radius_and_centers(best_points)
    
    # 2. Phase 1: Global Repulsion
    # Spread points out to avoid clusters
    alpha = 0.05
    for step in range(300):
        if time.time() > deadline:
            break
            
        forces = np.zeros_like(points)
        for i in range(N):
            for j in range(i + 1, N):
                diff = points[i] - points[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-12:
                    continue
                # Repulsive force ~ 1/d^3
                force = diff / (dist_sq ** 1.5)
                forces[i] += force
                forces[j] -= force
        
        # Normalize forces to prevent large jumps
        forces_norm = np.linalg.norm(forces, axis=1, keepdims=True)
        forces = forces / np.where(forces_norm == 0, 1, forces_norm)
        
        new_points = points + alpha * forces
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        radius, _, _, _ = _compute_covering_radius_and_centers(new_points)
        if radius < best_radius:
            best_radius = radius
            best_points = new_points.copy()
            points = new_points
        else:
            # If repulsion makes it worse, it's likely converged or oscillating
            # Reduce alpha
            alpha *= 0.95
            if alpha < 0.005:
                break

    # 3. Phase 2: Local Search targeting the largest facet
    # We use an adaptive step size and a stagnation counter
    alpha = 0.05
    stagnation_counter = 0
    max_stagnation = 50
    
    while time.time() < deadline:
        radius, center, facet_idx, hull = _compute_covering_radius_and_centers(points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        if center is None or facet_idx == -1:
            break
            
        simplex = hull.simplices[facet_idx]
        v0, v1, v2 = simplex
        
        # Strategy: Try moving the 3 vertices of the worst facet towards its circumcenter
        # We try different step sizes to find a descent direction
        
        improved = False
        
        # Try moving all 3 vertices
        new_points = points.copy()
        new_points[v0] = _move_point_towards(points[v0], center, alpha)
        new_points[v1] = _move_point_towards(points[v1], center, alpha)
        new_points[v2] = _move_point_towards(points[v2], center, alpha)
        
        new_radius, _, _, _ = _compute_covering_radius_and_centers(new_points)
        
        if new_radius < radius:
            points = new_points
            improved = True
        else:
            # Try moving vertices individually
            for v_idx in [v0, v1, v2]:
                test_points = points.copy()
                test_points[v_idx] = _move_point_towards(points[v_idx], center, alpha)
                test_radius, _, _, _ = _compute_covering_radius_and_centers(test_points)
                if test_radius < radius:
                    points = test_points
                    improved = True
                    break
            
        if not improved:
            # Reduce step size
            alpha *= 0.8
            if alpha < 0.001:
                # Stuck. Perturb the system to escape local minimum
                # Add noise to the worst facet's vertices
                noise = np.random.randn(3, 3) * 0.05
                new_points = points.copy()
                new_points[v0] = points[v0] + noise[0]
                new_points[v1] = points[v1] + noise[1]
                new_points[v2] = points[v2] + noise[2]
                
                # Normalize
                for idx in [v0, v1, v2]:
                    new_points[idx] = new_points[idx] / np.linalg.norm(new_points[idx])
                
                pert_radius, _, _, _ = _compute_covering_radius_and_centers(new_points)
                
                # Accept if it's not too bad, or always accept to explore
                # Simulated annealing style: accept if better, or with prob exp(-delta/T)
                # For simplicity, if pert_radius < best_radius * 1.1, accept
                if pert_radius < best_radius * 1.05:
                    points = new_points
                    alpha = 0.05 # Reset step size
                    stagnation_counter = 0
                else:
                    # If we are stuck and perturbation didn't help, 
                    # restart from best_points with new jitter?
                    # To save time, just break if we've been stuck for too long
                    if stagnation_counter > max_stagnation:
                        break
                    else:
                        # Try a smaller perturbation
                        alpha = 0.01

    return best_points
