import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def compute_covering_radius(points):
    """
    Computes the covering radius.
    points: (N, 3) array of unit vectors.
    Returns:
        max_radius: float
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return np.pi
    
    radii = np.zeros(len(simplices))
    
    for idx, simplex in enumerate(simplices):
        i, j, k = simplex
        a = points[i]
        b = points[j]
        c = points[k]
        
        # Normal vector to the plane of the triangle
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            radii[idx] = np.pi / 2
            continue
            
        u = n / norm_n
        
        # Ensure u points towards the centroid (inside the spherical triangle)
        centroid = (a + b + c) / 3.0
        if np.dot(u, centroid) < 0:
            u = -u
            
        # Circumradius is the angle between u and any vertex
        cos_theta = np.dot(u, a)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii[idx] = np.arccos(cos_theta)
        
    max_radius = np.max(radii) if len(radii) > 0 else np.pi
    return max_radius

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer for safety
    
    best_points = None
    best_radius = np.pi
    
    # Use a Fibonacci sphere as initial guess
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Repulsive Force Optimization
    # Potential: sum(1/r^2) or similar. Gradient points away from other points.
    # We want to maximize minimum distance, so we minimize potential energy.
    
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        if restart == 0:
            points = init_points.copy()
        else:
            np.random.seed(42 + restart)
            # Random initialization on sphere
            vecs = np.random.randn(N, 3)
            points = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        
        # Gradient Descent parameters
        lr = 0.01
        decay = 0.999
        min_lr = 1e-5
        max_steps = 2000
        
        current_points = points.copy()
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                break
            
            # Calculate repulsive forces
            # diff[i, j] = points[i] - points[j]
            # dist[i, j] = ||points[i] - points[j]||
            # Force on i from j is proportional to (points[i] - points[j]) / dist^2
            # We want to move i away from j, so we add force to i.
            
            # Vectorized calculation
            # (N, 1, 3) - (1, N, 3) -> (N, N, 3)
            diff = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            
            # Avoid division by zero for i=j
            np.fill_diagonal(dist_sq, 1.0)
            
            # Force magnitude: 1/dist^2 (inverse square law)
            # We want to minimize sum(1/dist^2), so gradient is -2 * diff / dist^4
            # Let's use a simpler heuristic: move away from neighbors.
            # Force vector on i is sum_j (points[i] - points[j]) / dist_ij^2
            # This corresponds to minimizing sum(1/dist_ij)
            
            # weights[i, j] = 1 / dist_sq[i, j]
            weights = 1.0 / dist_sq
            
            # Set diagonal to 0 to avoid self-interaction
            np.fill_diagonal(weights, 0.0)
            
            # force[i] = sum_j weights[i, j] * (points[i] - points[j])
            # This is equivalent to (weights @ diff) ? No.
            # force[i] = sum_j weights[i, j] * diff[i, j]
            
            # diff is (N, N, 3). weights is (N, N).
            # We want to sum over j (axis 1) for each i.
            # force[i, k] = sum_j weights[i, j] * diff[i, j, k]
            
            # Using einsum: force[i, k] = sum_j weights[i, j] * diff[i, j, k]
            forces = np.einsum('ij,ijk->ik', weights, diff)
            
            # Update positions
            current_points += lr * forces
            
            # Normalize to sphere
            norms = np.linalg.norm(current_points, axis=1, keepdims=True)
            current_points /= norms
            
            # Decay learning rate
            lr *= decay
            if lr < min_lr:
                break
        
        # Evaluate covering radius of the final points of this restart
        radius = compute_covering_radius(current_points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = current_points.copy()
            
    if best_points is None:
        best_points = init_points
        
    return best_points
