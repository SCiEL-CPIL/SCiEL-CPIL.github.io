import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _covering_radius_deg(pts):
    """Compute covering radius in degrees via Delaunay triangulation on sphere."""
    try:
        hull = ConvexHull(pts)
        max_circum = 0.0
        for tri in hull.simplices:
            a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            # Compute circumcenter of triangle on sphere
            # The circumcenter is equidistant from a, b, c on the sphere
            # It lies along the normal to the plane of the triangle
            # n = (a - b) x (a - c)
            n = np.cross(a - b, a - c)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            n = n / n_norm
            
            # The circumcenter c_circ satisfies:
            # c_circ . a = c_circ . b = c_circ . c = cos(R)
            # and |c_circ| = 1
            # c_circ is parallel to n (or -n)
            # Let c_circ = alpha * n
            # alpha * n . a = cos(R)
            # alpha * (n . a) = cos(R)
            # Also |alpha * n| = 1 => alpha = 1 (since n is unit)
            # So c_circ = n or c_circ = -n
            # We need the one that is "inside" the spherical triangle
            # Check which one gives a smaller distance to the vertices? No, both give same distance.
            # We need the one that lies in the same hemisphere as the triangle centroid
            centroid = (a + b + c) / 3.0
            if np.dot(n, centroid) < 0:
                n = -n
            
            # Now n is the circumcenter direction
            # Distance from n to a is arccos(n . a)
            dot_val = np.clip(np.dot(n, a), -1.0, 1.0)
            rad = np.arccos(dot_val)
            
            # Check if circumcenter is inside the triangle
            # If not, the max distance in this facet is to a vertex
            # But for covering radius, we take max circumradius of all facets
            # Actually, the covering radius is the max over all facets of the circumradius
            # IF the circumcenter is inside. If outside, the "effective" covering radius for that region
            # is determined by the vertices. But the global covering radius is still the max circumradius
            # of the Delaunay triangulation.
            
            if rad > max_circum:
                max_circum = rad
        
        return np.degrees(max_circum)
    except Exception:
        return 180.0

def _smooth_covering_radius(pts, temperature=0.5):
    """Compute a smooth approximation of the covering radius using soft-min."""
    # For each point on the sphere, we want min distance to any pt_i
    # But we don't have a grid. Instead, we use the fact that the covering radius
    # is related to the maximum over the sphere.
    # Alternative: use the convex hull approach but smooth it?
    # Actually, let's use a different approach: 
    # The covering radius is the smallest r such that the union of caps of radius r covers the sphere.
    # This is hard to optimize directly.
    # Let's stick with the convex hull method for evaluation, but use gradient descent
    # on a surrogate objective.
    
    # Surrogate: minimize the sum of exp(d_i^2 / T) where d_i is distance to nearest point
    # But we need to sample points on the sphere.
    # Let's sample 1000 random points on the sphere and compute the max distance
    # to the nearest configuration point. Then smooth it.
    
    # For efficiency, use a fixed set of sample points
    return None  # We'll use the convex hull method for now

def construct():
    # Fibonacci spiral initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    best_points = points.copy()
    best_rad = _covering_radius_deg(best_points)
    
    start_time = time.time()
    max_time = 140.0  # Safety margin before 150s limit
    
    # Gradient descent with momentum
    # We'll use the convex hull to find the worst facet and move points
    # But we'll also try a more systematic approach
    
    # First, do some random restarts with local optimization
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() - start_time > max_time:
            break
        
        if restart == 0:
            pts = points.copy()
        else:
            # Random initialization with some structure
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + np.random.uniform(0, 2*np.pi, N)
            r = np.sqrt(1.0 - z * z)
            pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            pts = _normalize(pts)
        
        current_rad = _covering_radius_deg(pts)
        if current_rad < best_rad:
            best_rad = current_rad
            best_points = pts.copy()
        
        # Local optimization: move points to reduce the worst facet's circumradius
        step_size = 0.02
        momentum = 0.5
        velocities = np.zeros_like(pts)
        
        for it in range(200):
            if time.time() - start_time > max_time:
                break
            
            try:
                hull = ConvexHull(pts)
            except Exception:
                break
            
            # Find the facet with the largest circumradius
            max_facet_rad = 0.0
            worst_indices = None
            worst_circumcenter = None
            
            for simplex in hull.simplices:
                a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                n = np.cross(a - b, a - c)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-10:
                    continue
                n = n / n_norm
                centroid = (a + b + c) / 3.0
                if np.dot(n, centroid) < 0:
                    n = -n
                dot_val = np.clip(np.dot(n, a), -1.0, 1.0)
                rad = np.arccos(dot_val)
                
                if rad > max_facet_rad:
                    max_facet_rad = rad
                    worst_indices = simplex
                    worst_circumcenter = n
            
            if worst_indices is None:
                break
            
            # Compute gradient: move vertices of worst facet towards circumcenter
            # Also add small random perturbation to escape local minima
            for j in range(3):
                idx = worst_indices[j]
                v = pts[idx]
                # Move towards circumcenter
                grad = worst_circumcenter - v
                grad = grad / np.linalg.norm(grad + 1e-10)
                velocities[idx] = momentum * velocities[idx] + step_size * grad
                new_v = v + velocities[idx]
                pts[idx] = new_v / np.linalg.norm(new_v)
            
            # Add small random noise to all points to escape local minima
            noise = np.random.normal(0, 0.001, pts.shape)
            pts = pts + noise
            pts = _normalize(pts)
            
            # Evaluate
            current_rad = _covering_radius_deg(pts)
            if current_rad < best_rad:
                best_rad = current_rad
                best_points = pts.copy()
            
            # Decay step size
            step_size *= 0.995
    
    return best_points
