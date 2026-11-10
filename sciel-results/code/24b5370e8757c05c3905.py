"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

# EVOLVE-BLOCK-START

def _covering_radius(points):
    """
    Compute the covering radius in radians using vectorized circumcenter calculation.
    """
    if len(points) < 3:
        return np.pi / 2
    
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        return np.pi
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi / 2
    
    faces = hull.simplices  # shape (F, 3)
    F = len(faces)
    
    # Extract face vertices: shape (F, 3, 3)
    p0 = points[faces[:, 0]]  # (F, 3)
    p1 = points[faces[:, 1]]
    p2 = points[faces[:, 2]]
    
    # Compute circumcenter for each face using vectorized operations
    # A = p1 - p0, B = p2 - p0
    A = p1 - p0  # (F, 3)
    B = p2 - p0  # (F, 3)
    
    # Dot products
    A_A = np.einsum('ij,ij->i', A, A)  # (F,)
    A_B = np.einsum('ij,ij->i', A, B)
    B_B = np.einsum('ij,ij->i', B, B)
    
    det = A_A * B_B - A_B * A_B  # (F,)
    
    # u and v for circumcenter: O = p0 + u*A + v*B
    # u = (0.5*A_A*B_B - 0.5*A_B*B_B) / det
    # v = (0.5*A_B*A_A - 0.5*A_A*B_B) / det
    safe_det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    
    u = (0.5 * A_A * B_B - 0.5 * A_B * B_B) / safe_det
    v = (0.5 * A_B * A_A - 0.5 * A_A * B_B) / safe_det
    
    # O = p0 + u*A + v*B, shape (F, 3)
    O = p0 + u[:, np.newaxis] * A + v[:, np.newaxis] * B
    
    # Normalize O to get spherical circumcenter
    norm_O = np.linalg.norm(O, axis=1, keepdims=True)
    safe_norm = np.where(norm_O < 1e-12, 1e-12, norm_O)
    C = O / safe_norm  # (F, 3)
    
    # For degenerate faces, use centroid as fallback
    degenerate = np.abs(det) < 1e-12
    if np.any(degenerate):
        centroids = (p0 + p1 + p2) / 3.0
        cent_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        safe_cent_norm = np.where(cent_norms < 1e-12, 1e-12, cent_norms)
        C[degenerate] = centroids[degenerate] / safe_cent_norm[degenerate]
    
    # Calculate angular distances from C to each vertex
    # d0 = arccos(C . p0), d1 = arccos(C . p1), d2 = arccos(C . p2)
    d0 = np.arccos(np.clip(np.einsum('ij,ij->i', C, p0), -1, 1))
    d1 = np.arccos(np.clip(np.einsum('ij,ij->i', C, p1), -1, 1))
    d2 = np.arccos(np.clip(np.einsum('ij,ij->i', C, p2), -1, 1))
    
    r_face = np.maximum(np.maximum(d0, d1), d2)  # (F,)
    
    return float(np.max(r_face))

def _optimize_from_start(start_points, time_limit, rng, initial_step=0.1):
    """
    Local optimization via hill-climbing with adaptive step size and escape mechanism.
    """
    points = start_points.copy()
    current_cr = _covering_radius(points)
    best_points = points.copy()
    best_cr = current_cr
    
    start_time = time.time()
    
    step_size = initial_step
    min_step = 1e-5
    max_iter = 500000
    no_improve_count = 0
    no_improve_limit = 200
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        idx = rng.integers(N)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        
        improved = False
        for sign in [1, -1]:
            new_point = points[idx] + sign * step_size * direction
            new_norm = np.linalg.norm(new_point)
            if new_norm < 1e-12:
                continue
            new_point /= new_norm
            
            old_point = points[idx]
            points[idx] = new_point
            
            new_cr = _covering_radius(points)
            
            if new_cr < current_cr - 1e-12:
                current_cr = new_cr
                improved = True
                if new_cr < best_cr:
                    best_cr = new_cr
                    best_points = points.copy()
                step_size = min(step_size * 1.03, 0.3)
                no_improve_count = 0
                break
            else:
                points[idx] = old_point
        
        if not improved:
            no_improve_count += 1
            step_size = max(step_size * 0.95, min_step)
            
            if no_improve_count > no_improve_limit:
                # Escape local minimum with a larger random perturbation
                step_size = rng.uniform(0.05, 0.15)
                no_improve_count = 0
        
        if step_size < min_step * 1.01:
            break
    
    return best_points, best_cr

def construct():
    total_deadline = 145.0
    start_time = time.time()
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_cr = np.pi
    
    # 1. Fibonacci spiral start
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    fib_points, fib_cr = _optimize_from_start(fib_points, 25.0, rng, initial_step=0.1)
    if fib_cr < best_cr:
        best_cr = fib_cr
        best_points = fib_points.copy()
    
    # 2. Random restarts with adaptive time allocation
    num_restarts = 10
    elapsed = time.time() - start_time
    remaining_time = total_deadline - elapsed - 5.0
    if remaining_time < 2.0:
        if best_points is not None:
            norms = np.linalg.norm(best_points, axis=1, keepdims=True)
            best_points = best_points / norms
        return best_points
    
    time_per_restart = remaining_time / num_restarts
    
    for i_restart in range(num_restarts):
        if time.time() - start_time > total_deadline - 5.0:
            break
        
        random_points = rng.normal(size=(N, 3))
        random_points /= np.linalg.norm(random_points, axis=1, keepdims=True)
        
        alloc_time = min(time_per_restart, total_deadline - (time.time() - start_time) - 5.0)
        if alloc_time < 1.5:
            break
        
        init_step = rng.uniform(0.05, 0.2)
        rand_points, rand_cr = _optimize_from_start(random_points, alloc_time, rng, initial_step=init_step)
        
        if rand_cr < best_cr:
            best_cr = rand_cr
            best_points = rand_points.copy()
    
    if best_points is not None:
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        best_points = best_points / norms
    
    return best_points

# EVOLVE-BLOCK-END
