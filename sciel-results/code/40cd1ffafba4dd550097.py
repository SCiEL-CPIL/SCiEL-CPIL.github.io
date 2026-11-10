"""
Combined program for covering the sphere with 130 equal caps.
Integrates fast circumradius evaluation via linear systems (Inspiration 1)
with a targeted local search strategy that moves the vertices of the worst
facet towards its circumcenter. Includes a global repulsion phase and
simulated annealing-style perturbations to escape local minima.
"""
import numpy as np
import time
from scipy.spatial import ConvexHull
import math

N = 130

def _compute_covering_radius_and_centers(points):
    """
    Computes the covering radius (max circumradius of hull facets) 
    and returns the radius in degrees, the circumcenter of the worst facet, 
    the indices of the worst facet's vertices, and the hull object.
    
    Uses the linear system method for robust circumcenter calculation:
    For a spherical triangle with vertices u, v, w, the circumcenter C
    satisfies C = x*u + y*v + z*w where x,y,z are found by solving:
    M @ [x,y,z]^T = [1,1,1]^T
    where M is the Gram matrix of the vertices.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0, None, None, None

    max_radius_deg = 0.0
    worst_center = None
    worst_vertices = None

    for simplex in hull.simplices:
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
            coeffs = np.linalg.solve(M, np.ones(3))
        except np.linalg.LinAlgError:
            continue
            
        C = coeffs[0] * u + coeffs[1] * v + coeffs[2] * w
        norm_C = np.linalg.norm(C)
        if norm_C < 1e-10:
            continue
        C = C / norm_C
        
        # The dot product of the circumcenter with any vertex is cos(R)
        cos_R = np.dot(C, u)
        cos_R = np.clip(cos_R, -1.0, 1.0)
        
        R_rad = math.acos(cos_R)
        R_deg = math.degrees(R_rad)
        
        if R_deg > max_radius_deg:
            max_radius_deg = R_deg
            worst_center = C
            worst_vertices = simplex

    return max_radius_deg, worst_center, worst_vertices, hull

def _move_point_towards(p, c, step_size):
    """
    Move point p on the unit sphere towards point c by step_size radians.
    Uses great circle interpolation (slerp) approximation via tangent projection.
    """
    dot_pc = np.dot(p, c)
    # Tangent vector from p towards c
    t = c - dot_pc * p
    norm_t = np.linalg.norm(t)
    if norm_t < 1e-10:
        return p
    t = t / norm_t
    
    # Move along the tangent
    new_p = p + step_size * t
    new_p = new_p / np.linalg.norm(new_p)
    return new_p

def _global_repulsion(points, deadline, alpha=0.05):
    """
    Phase 1: Spread points out using repulsive forces to avoid clusters.
    """
    points = np.array(points, dtype=float)
    best_points = points.copy()
    best_radius, _, _, _ = _compute_covering_radius_and_centers(best_points)
    
    for step in range(100):
        if time.time() > deadline:
            break
            
        forces = np.zeros_like(points)
        # O(N^2) repulsion
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
            # If repulsion makes it worse, reduce alpha
            alpha *= 0.9
            if alpha < 0.001:
                break

    return best_points, best_radius

def _local_search_targeted(points, deadline, tol=1e-8):
    """
    Phase 2: Local search targeting the largest facet.
    Moves the vertices of the worst facet towards its circumcenter.
    Includes perturbation to escape local minima.
    """
    points = np.array(points, dtype=float)
    best_points = points.copy()
    best_radius, _, _, _ = _compute_covering_radius_and_centers(best_points)
    
    alpha = 0.05
    stagnation_counter = 0
    max_stagnation = 20
    temperature = 0.1
    
    while time.time() < deadline:
        radius, center, vertices, _ = _compute_covering_radius_and_centers(points)
        
        if radius < best_radius - tol:
            best_radius = radius
            best_points = points.copy()
            stagnation_counter = 0
            # Reset temperature if we found a new best
            temperature = max(temperature * 0.9, 0.01)
        else:
            stagnation_counter += 1
            
        if center is None or vertices is None:
            break
            
        v0, v1, v2 = vertices
        
        improved = False
        
        # Strategy 1: Move all 3 vertices towards circumcenter
        new_points = points.copy()
        new_points[v0] = _move_point_towards(points[v0], center, alpha)
        new_points[v1] = _move_point_towards(points[v1], center, alpha)
        new_points[v2] = _move_point_towards(points[v2], center, alpha)
        
        new_radius, _, _, _ = _compute_covering_radius_and_centers(new_points)
        
        if new_radius < radius - tol:
            points = new_points
            improved = True
        else:
            # Strategy 2: Try moving vertices individually
            for v_idx in [v0, v1, v2]:
                test_points = points.copy()
                test_points[v_idx] = _move_point_towards(points[v_idx], center, alpha)
                test_radius, _, _, _ = _compute_covering_radius_and_centers(test_points)
                if test_radius < radius - tol:
                    points = test_points
                    improved = True
                    break
        
        if not improved:
            # Stuck. Reduce step size and consider perturbation.
            alpha *= 0.8
            if alpha < 0.001:
                # Simulated Annealing style perturbation
                # Accept worse moves with probability exp(-delta/T)
                delta = radius - best_radius
                if delta < 0: delta = 0
                prob = math.exp(-delta / max(temperature, 1e-6))
                
                if np.random.rand() < prob or stagnation_counter > max_stagnation:
                    # Perturb the worst facet's vertices
                    noise_scale = 0.05 + temperature * 0.1
                    new_points = points.copy()
                    new_points[v0] = points[v0] + np.random.randn(3) * noise_scale
                    new_points[v1] = points[v1] + np.random.randn(3) * noise_scale
                    new_points[v2] = points[v2] + np.random.randn(3) * noise_scale
                    
                    # Normalize
                    for idx in [v0, v1, v2]:
                        new_points[idx] = new_points[idx] / np.linalg.norm(new_points[idx])
                    
                    points = new_points
                    alpha = 0.05 # Reset step size
                    temperature *= 0.9 # Cool down
                    stagnation_counter = 0
                else:
                    # If we can't accept a perturbation and are stuck, break
                    if stagnation_counter > max_stagnation * 2:
                        break

    return best_points, best_radius

def _fibonacci_start(n=N):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    best_points = None
    best_radius = 90.0
    
    # Set random seed for reproducibility in perturbations
    np.random.seed(42)
    
    # Start 1: Fibonacci spiral with jitter
    points = _fibonacci_start(N)
    # Add small jitter to break symmetries
    noise = np.random.randn(N, 3) * 0.01
    points = points + noise
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Phase 1: Global Repulsion
    if time.time() < deadline - 30.0:
        rep_points, rep_radius = _global_repulsion(points.copy(), deadline - 30.0)
        if rep_radius < best_radius:
            best_radius = rep_radius
            best_points = rep_points.copy()
        else:
            best_points = points.copy()
            best_radius = rep_radius
            
    # Phase 2: Targeted Local Search
    if time.time() < deadline - 5.0 and best_points is not None:
        refined, radius = _local_search_targeted(best_points.copy(), deadline - 5.0)
        if radius < best_radius:
            best_radius = radius
            best_points = refined.copy()
    
    # If time remains, try a second restart with different jitter
    if time.time() < deadline - 10.0:
        np.random.seed(123)
        points2 = _fibonacci_start(N)
        noise2 = np.random.randn(N, 3) * 0.05
        points2 = points2 + noise2
        points2 = points2 / np.linalg.norm(points2, axis=1, keepdims=True)
        
        # Quick repulsion
        if time.time() < deadline - 15.0:
            rep_points2, rep_radius2 = _global_repulsion(points2.copy(), deadline - 15.0)
            if rep_radius2 < best_radius:
                best_radius = rep_radius2
                best_points = rep_points2.copy()
        
        # Local search on the best of the two so far
        if time.time() < deadline - 5.0:
            refined2, radius2 = _local_search_targeted(best_points.copy(), deadline - 5.0)
            if radius2 < best_radius:
                best_radius = radius2
                best_points = refined2.copy()
    
    # Ensure valid output
    if best_points is None:
        best_points = _fibonacci_start(N)
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
