import numpy as np
from scipy.spatial import ConvexHull

N = 50

def _covering_radius(points):
    """
    Calculate the covering radius for a set of points on the unit sphere.
    Returns the maximum angular distance from any point on the sphere to the nearest point in the set.
    This is equivalent to the maximum circumradius of the convex hull facets.
    """
    try:
        # Ensure points are normalized
        norms = np.linalg.norm(points, axis=1)
        if np.any(norms < 1e-9):
            return np.pi
        points = points / norms[:, np.newaxis]
        
        hull = ConvexHull(points)
        max_radius = 0.0
        
        # Iterate over simplices (facets)
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            
            # Calculate normal to the facet
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm_n = np.linalg.norm(normal)
            
            if norm_n < 1e-9:
                continue
                
            normal = normal / norm_n
            
            # The circumradius of the spherical triangle is the angle between the normal and any vertex
            # Since the triangle is on the unit sphere, the distance from the center to the plane is d = dot(normal, p1)
            # The angular radius R satisfies cos(R) = d
            d = np.dot(normal, p1)
            
            # If d <= 0, the facet contains the origin or is degenerate in a way that implies covering radius >= pi/2 or more
            # Actually, if the origin is inside the hull, d > 0 for all outward normals.
            # If d is very small, R is close to pi/2.
            # If d < 0, it means the normal is pointing inwards or the triangle wraps around? 
            # ConvexHull normals point outwards. For a convex polytope containing the origin, dot(normal, p) > 0.
            # If dot(normal, p) <= 0, the covering radius is at least pi/2.
            
            if d <= 1e-9:
                # If the facet is "flat" or the origin is on the boundary, radius is at least pi/2
                # But we can just use arccos(clip(d))
                pass
                
            cos_R = np.clip(d, -1.0, 1.0)
            R = np.arccos(cos_R)
            
            if R > max_radius:
                max_radius = R
                
        return max_radius
    except Exception:
        return np.pi

def construct():
    """
    Uses Simulated Annealing to minimize the covering radius.
    """
    # Initial guess: Fibonacci spiral is a good starting point
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure normalization
    points = points / np.linalg.norm(points, axis=1)[:, np.newaxis]
    
    current_energy = _covering_radius(points)
    best_points = points.copy()
    best_energy = current_energy
    
    # Simulated Annealing Parameters
    T = 0.1  # Initial temperature
    T_min = 1e-6
    alpha = 0.999 # Cooling factor
    n_iter = 50000 # Number of iterations
    
    # Helper to generate a random perturbation on the sphere
    def perturb_point(p, sigma):
        # Generate random vector in tangent space
        # Random vector in R3
        v = np.random.normal(0, 1, 3)
        # Project onto tangent plane: v_tan = v - (v.p)p
        v_tan = v - np.dot(v, p) * p
        v_tan = v_tan / np.linalg.norm(v_tan)
        # Move along tangent
        new_p = p + sigma * v_tan
        return new_p / np.linalg.norm(new_p)
    
    for it in range(n_iter):
        # Choose a point to move
        idx = np.random.randint(N)
        
        # Determine step size (sigma)
        # Start with larger steps, decrease over time
        sigma = 0.1 * (T / 0.1) + 0.01
        
        # Create new candidate
        new_points = points.copy()
        new_points[idx] = perturb_point(points[idx], sigma)
        
        # Calculate new energy
        new_energy = _covering_radius(new_points)
        
        # Metropolis criterion
        delta = new_energy - current_energy
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            points = new_points
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            
    return best_points
