"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

# EVOLVE-BLOCK-START

def _covering_radius(points):
    """
    Compute the covering radius in radians.
    The covering radius is the maximum angular distance from any point on the sphere
    to the nearest configuration point.
    This is equivalent to the maximum circumradius of the spherical Delaunay triangles
    (faces of the convex hull).
    """
    if len(points) < 3:
        return np.pi / 2
    
    # Ensure points are on the unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        return np.pi
    points = points / norms
    
    # Calculate convex hull of the 3D points
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback if hull fails
        return np.pi / 2

    max_radius = 0.0
    # Hull faces are indices of points
    for face in hull.simplices:
        p0, p1, p2 = points[face[0]], points[face[1]], points[face[2]]
        
        # Calculate 3D circumcenter of triangle p0, p1, p2
        A = p1 - p0
        B = p2 - p0
        A_A = np.dot(A, A)
        A_B = np.dot(A, B)
        B_B = np.dot(B, B)
        
        det = A_A * B_B - A_B * A_B
        if abs(det) < 1e-12:
            centroid = (p0 + p1 + p2) / 3.0
            norm_c = np.linalg.norm(centroid)
            if norm_c < 1e-12:
                continue
            C = centroid / norm_c
        else:
            u = (0.5 * A_A * B_B - 0.5 * A_B * B_B) / det
            v = (0.5 * A_B * A_A - 0.5 * A_A * B_B) / det
            
            O = p0 + u * A + v * B
            
            norm_O = np.linalg.norm(O)
            if norm_O < 1e-12:
                centroid = (p0 + p1 + p2) / 3.0
                norm_c = np.linalg.norm(centroid)
                if norm_c < 1e-12:
                    continue
                C = centroid / norm_c
            else:
                C = O / norm_O
        
        # Calculate angular distance from C to vertices
        d0 = np.arccos(np.clip(np.dot(C, p0), -1, 1))
        d1 = np.arccos(np.clip(np.dot(C, p1), -1, 1))
        d2 = np.arccos(np.clip(np.dot(C, p2), -1, 1))
        r_face = max(d0, d1, d2)
        
        if r_face > max_radius:
            max_radius = r_face
            
    return max_radius

def _optimize_from_start(start_points, time_limit, rng, initial_step=0.1):
    """
    Local optimization via hill-climbing with adaptive step size.
    """
    points = start_points.copy()
    current_cr = _covering_radius(points)
    best_points = points.copy()
    best_cr = current_cr
    
    start_time = time.time()
    
    # Adaptive step size
    step_size = initial_step
    min_step = 1e-5
    max_iter = 200000
    no_improve_count = 0
    no_improve_limit = 500
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random point to move
        idx = rng.integers(N)
        
        # Generate a random direction
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        
        improved = False
        # Try moving in positive and negative direction
        for sign in [1, -1]:
            new_point = points[idx] + sign * step_size * direction
            new_norm = np.linalg.norm(new_point)
            if new_norm < 1e-12:
                continue
            new_point /= new_norm
            
            old_point = points[idx]
            points[idx] = new_point
            
            new_cr = _covering_radius(points)
            
            if new_cr < current_cr - 1e-10:
                # Accept
                current_cr = new_cr
                improved = True
                if new_cr < best_cr:
                    best_cr = new_cr
                    best_points = points.copy()
                # Increase step size slightly if successful
                step_size = min(step_size * 1.05, 0.5)
                no_improve_count = 0
                break
            else:
                # Reject, revert
                points[idx] = old_point
        
        if not improved:
            no_improve_count += 1
            # Decrease step size
            step_size = max(step_size * 0.95, min_step)
            
            # If stuck for a long time, try a larger random jump to escape local minima
            if no_improve_count > no_improve_limit:
                # Random restart of the step size
                step_size = np.random.uniform(0.05, 0.2)
                no_improve_count = 0
                
        # Early stop if step size is very small
        if step_size < min_step * 1.01:
            break
            
    return best_points, best_cr

def construct():
    # Deadline guard
    total_deadline = 140.0  # Leave 10s safety margin for 150s limit
    start_time = time.time()
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_cr = np.pi
    
    # 1. Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimize from Fibonacci
    fib_points, fib_cr = _optimize_from_start(fib_points, 20.0, rng, initial_step=0.1)
    if fib_cr < best_cr:
        best_cr = fib_cr
        best_points = fib_points.copy()
    
    # 2. Multiple random restarts
    # Each restart gets a smaller time budget
    num_restarts = 8
    elapsed = time.time() - start_time
    remaining_time = total_deadline - elapsed - 5.0
    if remaining_time < 1.0:
        if best_points is not None:
            norms = np.linalg.norm(best_points, axis=1, keepdims=True)
            best_points = best_points / norms
        return best_points
    
    time_per_restart = remaining_time / num_restarts
    
    for i_restart in range(num_restarts):
        if time.time() - start_time > total_deadline - 5.0:
            break
            
        # Random start
        random_points = rng.normal(size=(N, 3))
        random_points /= np.linalg.norm(random_points, axis=1, keepdims=True)
        
        # Optimize from random start
        # Use less time for later restarts if running low
        alloc_time = min(time_per_restart, total_deadline - (time.time() - start_time) - 5.0)
        if alloc_time < 1.0:
            break
            
        # Vary initial step size for diversity
        init_step = rng.uniform(0.05, 0.2)
        rand_points, rand_cr = _optimize_from_start(random_points, alloc_time, rng, initial_step=init_step)
        
        if rand_cr < best_cr:
            best_cr = rand_cr
            best_points = rand_points.copy()
    
    # Final normalization
    if best_points is not None:
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        best_points = best_points / norms
    
    return best_points

# EVOLVE-BLOCK-END
