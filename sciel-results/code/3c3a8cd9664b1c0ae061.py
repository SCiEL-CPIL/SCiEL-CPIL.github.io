import numpy as np
import time

N = 50
TARGET_RADIUS = 0.1258254895304

def compute_radius(points):
    """
    Computes the maximum radius r such that circles of radius r centered at points
    are inside the unit circle and do not overlap.
    """
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
        
    # Distance to boundary: min(1 - |c_i|)
    dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dists_to_boundary)
    
    if min_boundary <= 0:
        return 0.0
    
    # Minimum distance between centers
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    
    min_dist = np.min(dists)
    
    # The radius is limited by half the minimum distance between centers
    # and the minimum distance to the boundary
    r = min(min_dist / 2.0, min_boundary)
    
    return r if r > 0 else 0.0

def initial_ring_structure():
    """
    Generates a structured initial configuration based on concentric rings.
    This provides a better starting point than random or hexagonal lattices
    for the specific geometry of circle packing in a circle.
    """
    points = np.zeros((N, 2))
    
    # Ring 0: Center
    points[0] = [0.0, 0.0]
    count = 1
    
    # Define rings: (radius, number of circles)
    # Based on typical dense packings: 1 + 6 + 12 + 18 + 13 = 50
    rings = [
        (0.25, 6),
        (0.50, 12),
        (0.75, 18),
        (0.95, 13)
    ]
    
    current_index = 1
    for radius, num_circles in rings:
        if current_index + num_circles > N:
            num_circles = N - current_index
        
        angles = np.linspace(0, 2 * np.pi, num_circles, endpoint=False)
        # Add a small random offset to break symmetry and help escape local minima
        angles += np.random.uniform(0, 2 * np.pi / num_circles)
        
        for k in range(num_circles):
            points[current_index, 0] = radius * np.cos(angles[k])
            points[current_index, 1] = radius * np.sin(angles[k])
            current_index += 1
        
        if current_index >= N:
            break
            
    # Add small random perturbation to avoid perfect symmetry lock-in
    points += np.random.normal(0, 0.01, size=(N, 2))
    
    # Ensure inside container
    norms = np.linalg.norm(points, axis=1)
    mask = norms > 1.0
    points[mask] = points[mask] / norms[mask][:, np.newaxis]
    
    return points

def md_optimize(points, time_limit=60.0):
    """
    Molecular Dynamics optimization with adaptive cooling.
    Uses a soft-sphere repulsive potential to push circles apart and into the container.
    """
    points = points.copy()
    velocities = np.zeros_like(points)
    
    # Parameters
    dt = 0.001
    T = 0.05  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.995
    damping = 0.99  # Velocity damping
    
    # Potential parameters
    # We want to maximize r. 
    # Repulsion between circles: F ~ 1/d^2 if d < 2*r_target
    # Repulsion from wall: F ~ 1/d_wall^2 if d_wall < r_target
    
    r_target = 0.125
    k_repel = 10.0
    k_wall = 10.0
    
    start_time = time.time()
    best_points = points.copy()
    best_r = compute_radius(points)
    
    steps = 0
    while time.time() - start_time < time_limit:
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                
                if dist < 2 * r_target and dist > 1e-8:
                    # Soft sphere repulsion
                    # Force magnitude increases as distance decreases
                    overlap = 2 * r_target - dist
                    force_mag = k_repel * (overlap ** 2)
                    force_vec = (diff / dist) * force_mag
                    forces[i] += force_vec
                    forces[j] -= force_vec
                elif dist < 1e-8:
                    # Coincident points, push randomly
                    rand_vec = np.random.normal(0, 1, 2)
                    forces[i] += rand_vec
                    forces[j] -= rand_vec
        
        # Wall repulsion
        for i in range(N):
            norm = np.linalg.norm(points[i])
            if norm > 1 - r_target:
                # Push towards center
                # Distance from wall = 1 - norm
                dist_wall = 1 - norm
                if dist_wall < 0:
                    dist_wall = 0.01 # Clamp to avoid division by zero or negative forces
                force_mag = k_wall * (r_target - dist_wall) ** 2
                # Direction is towards center: -points[i] / norm
                force_vec = (-points[i] / norm) * force_mag
                forces[i] += force_vec
        
        # Update velocities and positions (Velocity Verlet or simple Euler with damping)
        velocities = velocities * damping + forces * dt
        points += velocities * dt
        
        # Ensure points stay roughly inside to prevent explosion
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.5
        if np.any(mask):
            points[mask] = points[mask] / norms[mask][:, np.newaxis]
            velocities[mask] = 0
            
        steps += 1
        
        # Cool down
        T *= cooling_rate
        
        # Check improvement periodically
        if steps % 100 == 0:
            current_r = compute_radius(points)
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
                
        if T < T_min:
            break
            
    return best_points, best_r

def gradient_refine(points, time_limit=20.0):
    """
    Final polish using gradient descent on a smooth approximation of the objective.
    This helps to settle into the precise local optimum after MD.
    """
    from scipy.optimize import minimize
    
    def objective(x):
        pts = x.reshape(N, 2)
        # Penalty for boundary violation
        norms = np.linalg.norm(pts, axis=1)
        boundary_viol = np.maximum(0, norms - (1 - 0.125))
        # Penalty for overlap
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        overlap_viol = np.maximum(0, 0.25 - dists)
        
        return np.sum(boundary_viol**2) + np.sum(overlap_viol**2)
        
    def gradient(x):
        pts = x.reshape(N, 2)
        grad = np.zeros_like(x)
        
        # Boundary gradient
        norms = np.linalg.norm(pts, axis=1)
        boundary_viol = np.maximum(0, norms - (1 - 0.125))
        for i in range(N):
            if boundary_viol[i] > 0 and norms[i] > 1e-8:
                grad[2*i:2*i+2] = 2 * boundary_viol[i] * pts[i] / norms[i]
                
        # Pairwise gradient
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        overlap_viol = np.maximum(0, 0.25 - dists)
        
        for i in range(N):
            for j in range(i + 1, N):
                if overlap_viol[i, j] > 0 and dists[i, j] > 1e-8:
                    vec = pts[i] - pts[j]
                    d = dists[i, j]
                    grad[2*i:2*i+2] += 2 * overlap_viol[i, j] * vec / d
                    grad[2*j:2*j+2] -= 2 * overlap_viol[i, j] * vec / d
                    
        return grad
        
    x0 = points.flatten()
    
    try:
        res = minimize(
            fun=objective,
            x0=x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'maxfun': 4000}
        )
        if res.success or res.status in [1, 2, 3]:
            new_points = res.x.reshape(N, 2)
            new_r = compute_radius(new_points)
            current_r = compute_radius(points)
            if new_r > current_r:
                return new_points, new_r
    except Exception:
        pass
        
    return points, compute_radius(points)

def construct():
    deadline = time.time() + 145.0
    
    best_overall_r = 0.0
    best_overall_points = None
    
    # Strategy 1: Ring initialization + MD
    try:
        init_pts = initial_ring_structure()
        r_check = compute_radius(init_pts)
        
        if r_check > 0:
            # Run MD for a significant portion of the time
            md_time = min(60.0, max(5.0, deadline - time.time() - 25.0))
            p, r = md_optimize(init_pts, time_limit=md_time)
            if r > best_overall_r:
                best_overall_r = r
                best_overall_points = p
    except Exception:
        pass
    
    if best_overall_points is not None and time.time() < deadline - 10:
        # Final polish
        remaining_time = deadline - time.time() - 2.0
        if remaining_time > 5:
            p, r = gradient_refine(best_overall_points, time_limit=remaining_time)
            if r > best_overall_r:
                best_overall_r = r
                best_overall_points = p

    # Fallback if MD failed or produced garbage
    if best_overall_points is None or best_overall_r <= 0:
        # Try a simple random start with MD
        try:
            pts = np.random.uniform(-0.8, 0.8, size=(N, 2))
            # Project inside
            norms = np.linalg.norm(pts, axis=1)
            mask = norms > 0.8
            pts[mask] = pts[mask] / norms[mask][:, np.newaxis] * 0.8
            
            md_time = min(40.0, max(5.0, deadline - time.time() - 10.0))
            p, r = md_optimize(pts, time_limit=md_time)
            if r > best_overall_r:
                best_overall_r = r
                best_overall_points = p
        except Exception:
            pass
            
    # Final safety check
    if best_overall_points is not None:
        # Ensure validity
        norms = np.linalg.norm(best_overall_points, axis=1)
        if np.any(norms > 1.0):
            best_overall_points = best_overall_points / norms[:, np.newaxis]
            
    if best_overall_points is None:
        # Last resort: return a valid, if not optimal, configuration
        best_overall_points = initial_ring_structure()
        
    return best_overall_points

if __name__ == "__main__":
    pts = construct()
    r = compute_radius(pts)
    print(f"Radius: {r}")
    print(f"Shape: {pts.shape}")
