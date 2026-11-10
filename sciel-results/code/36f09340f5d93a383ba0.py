import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def _normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return np.zeros(3)
    return v / norm

def _fibonacci_sphere(n):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_circumcenters(points):
    """
    Compute circumcenters and radii for all faces of the convex hull.
    Returns: (max_radius, face_indices, circumcenters)
    """
    try:
        hull = ConvexHull(points)
        faces = hull.simplices
    except Exception:
        return np.pi, np.array([], dtype=int), np.array([])
    
    max_radius = 0.0
    max_face_idx = -1
    circumcenters = []
    
    for idx, face in enumerate(faces):
        v1, v2, v3 = points[face]
        
        # Calculate circumcenter
        u = v2 - v1
        w = v3 - v1
        n = np.cross(u, w)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            circumcenters.append(np.zeros(3))
            continue
        n = n / n_norm
        
        # Planar circumcenter
        A = np.array([[np.dot(u, u), np.dot(w, u)], [np.dot(u, w), np.dot(w, w)]])
        B = np.array([np.dot(u, u), np.dot(w, w)])
        
        try:
            coeffs = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            circumcenters.append(np.zeros(3))
            continue
        a, b = coeffs
        c_p = v1 + a * u + b * w
        
        c_p_norm = np.linalg.norm(c_p)
        if c_p_norm < 1e-10:
            circumcenters.append(np.zeros(3))
            continue
            
        c_s = c_p / c_p_norm
        cos_R = np.dot(c_s, v1)
        cos_R = np.clip(cos_R, -1.0, 1.0)
        R = np.arccos(cos_R)
        
        circumcenters.append(c_s)
        
        if R > max_radius:
            max_radius = R
            max_face_idx = idx
            
    return max_radius, faces, np.array(circumcenters)

def _repulsive_force(points, k=1.0, power=2.0):
    """
    Compute repulsive forces between all pairs of points.
    Force on point i from point j: k * (1 - dot(pi, pj)) ** power * (pi - pj)
    This pushes points apart when they are close.
    """
    n = len(points)
    forces = np.zeros_like(points)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dot = np.dot(points[i], points[j])
            dot = np.clip(dot, -1.0, 1.0)
            dist_factor = (1.0 - dot) ** power
            force = k * dist_factor * (points[i] - points[j])
            forces[i] += force
    return forces

def _attractive_force_to_circumcenters(points, circumcenters, face_indices, weight=1.0):
    """
    Compute attractive forces pulling vertices toward circumcenters of large faces.
    """
    n = len(points)
    forces = np.zeros_like(points)
    for idx, face in enumerate(face_indices):
        cc = circumcenters[idx]
        if np.linalg.norm(cc) < 1e-10:
            continue
        for i in face:
            # Pull point i toward circumcenter
            forces[i] += weight * (cc - points[i])
    return forces

def _project_to_sphere(points):
    """Normalize points to lie on the unit sphere."""
    norms = np.linalg.norm(points, axis=1)
    norms[norms < 1e-10] = 1.0
    return points / norms[:, np.newaxis]

def _gradient_descent_on_sphere(points, max_time=10.0, lr=0.01):
    """
    Perform gradient descent on the sphere to minimize a surrogate for covering radius.
    We minimize the sum of squared distances to the nearest point, or use repulsive forces.
    Here we use a combination of repulsive forces and attraction to circumcenters.
    """
    start_time = time.time()
    best_points = points.copy()
    best_radius = _compute_circumcenters(points)[0]
    
    while time.time() - start_time < max_time:
        radius, faces, ccs = _compute_circumcenters(points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
        
        # Compute forces
        rep_force = _repulsive_force(points, k=0.1, power=2.0)
        att_force = _attractive_force_to_circumcenters(points, ccs, faces, weight=0.5)
        
        total_force = rep_force + att_force
        
        # Update: move along force direction, then project to sphere
        points = points + lr * total_force
        points = _project_to_sphere(points)
        
        # Adaptive learning rate: decrease if not improving
        if radius >= best_radius:
            lr *= 0.95
        else:
            lr = min(lr * 1.05, 0.05)
        
        # Check convergence
        if lr < 1e-6:
            break
            
    return best_points

def construct():
    start_time = time.time()
    max_time = 115.0
    
    # Start with Fibonacci sphere
    points = _fibonacci_sphere(N)
    
    best_points = points.copy()
    best_radius = _compute_circumcenters(points)[0]
    
    np.random.seed(42)
    
    # Perform multiple rounds of gradient descent with perturbations
    for _ in range(5):
        # Perturb the current best solution
        perturbed = best_points + np.random.normal(0, 0.1, (N, 3))
        perturbed = _project_to_sphere(perturbed)
        
        # Run gradient descent
        elapsed = time.time() - start_time
        remaining = max_time - elapsed
        if remaining < 10:
            break
            
        gd_time = min(15.0, remaining / 2.0)
        refined = _gradient_descent_on_sphere(perturbed, max_time=gd_time)
        
        # Evaluate
        radius, _, _ = _compute_circumcenters(refined)
        
        if radius < best_radius:
            best_radius = radius
            best_points = refined.copy()
            
        # If we found a very good solution, break early
        if best_radius < 0.32:
            break
    
    # Final refinement without perturbation
    elapsed = time.time() - start_time
    remaining = max_time - elapsed
    if remaining > 5:
        final = _gradient_descent_on_sphere(best_points, max_time=min(20.0, remaining))
        radius, _, _ = _compute_circumcenters(final)
        if radius < best_radius:
            best_points = final
    
    return best_points
