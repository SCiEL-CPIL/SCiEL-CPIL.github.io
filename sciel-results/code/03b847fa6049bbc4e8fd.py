import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return pts / norms

    def get_covering_radius(pts):
        """
        Computes the covering radius in radians.
        Returns (radius, worst_simplex_indices) or (np.pi, None) on failure.
        """
        try:
            hull = ConvexHull(pts)
            simplices = hull.simplices
            
            # Extract vertices for all simplices
            a = pts[simplices[:, 0]]
            b = pts[simplices[:, 1]]
            c = pts[simplices[:, 2]]
            
            # Compute normals
            # n = cross(b - a, c - a)
            n = np.cross(b - a, c - a)
            n_norms = np.linalg.norm(n, axis=1)
            
            # Mask out degenerate facets
            valid = n_norms > 1e-10
            if not np.any(valid):
                return np.pi, None
            
            n = n / np.maximum(n_norms[:, None], 1e-10)
            
            # Ensure normals point outward (away from origin)
            # The origin is inside the hull. The vector from origin to any vertex is 'outward' relative to the origin.
            # If n . a < 0, n points towards the origin (inward).
            dots = np.einsum('ij,ij->i', n, a)
            signs = np.where(dots < 0, -1.0, 1.0)
            n = n * signs[:, None]
            
            # Recalculate dots with corrected normals
            dots = np.einsum('ij,ij->i', n, a)
            # Clip for arccos stability
            dots = np.clip(dots, -1.0, 1.0)
            
            # Circumradius is arccos(dot)
            radii = np.arccos(dots)
            radii[~valid] = 0.0
            
            max_rad = np.max(radii)
            worst_idx = np.argmax(radii)
            worst_simplex = simplices[worst_idx]
            
            return max_rad, worst_simplex

        except Exception:
            return np.pi, None

    # Initialize with Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = normalize(points)
    
    best_points = points.copy()
    best_radius, _ = get_covering_radius(best_points)
    
    # Simulated Annealing
    current_points = points.copy()
    current_radius, _ = get_covering_radius(current_points)
    
    T = 0.1  # Initial temperature (radians)
    cooling_rate = 0.9995
    min_T = 1e-5
    
    # Random number generator
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            break
            
        # Propose a move: perturb 1 to 3 random points
        num_perturbed = rng.integers(1, 4)
        indices = rng.choice(N, size=num_perturbed, replace=False)
        
        # Random rotation
        # Generate random axis and angle
        angles = rng.uniform(-0.1, 0.1, size=(num_perturbed, 3))
        # Apply rotation approximately by adding small random vector and normalizing
        # This is a simple random walk on the sphere
        perturbations = rng.normal(0, 0.1, size=(num_perturbed, 3))
        
        new_points = current_points.copy()
        new_points[indices] += perturbations
        new_points[indices] = normalize(new_points[indices])
        
        new_radius, _ = get_covering_radius(new_points)
        
        # Metropolis criterion
        delta = new_radius - current_radius
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_points = new_points
            current_radius = new_radius
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = current_points.copy()
        
        T *= cooling_rate
        
    # Final Polish: Lloyd-like iteration for the best points
    points = best_points.copy()
    step_size = 0.05
    for _ in range(100):
        if time.time() - start_time > time_limit - 2:
            break
        current_radius, worst_simplex = get_covering_radius(points)
        if worst_simplex is None:
            break
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        # Move vertices of worst facet towards circumcenter
        # We need the circumcenter direction
        a = points[worst_simplex[0]]
        b = points[worst_simplex[1]]
        c = points[worst_simplex[2]]
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm > 1e-10:
            n /= n_norm
            if n @ a < 0:
                n = -n
            
            for idx in worst_simplex:
                points[idx] = points[idx] + step_size * (n - points[idx])
            points = normalize(points)
            
        step_size *= 0.95

    return best_points
