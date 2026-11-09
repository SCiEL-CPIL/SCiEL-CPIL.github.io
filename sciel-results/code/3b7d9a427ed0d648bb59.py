import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _covering_radius_and_grad(points):
    """
    Compute the covering radius in radians and the gradient with respect to points.
    The covering radius is the max circumradius of the Delaunay triangles.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        # Initialize gradient
        grad = np.zeros_like(points)
        
        p1 = points[facets[:, 0]]
        p2 = points[facets[:, 1]]
        p3 = points[facets[:, 2]]
        
        # Compute circumcenters and radii for each facet
        # v1 = p1 - p2, v2 = p1 - p3
        v1 = p1 - p2
        v2 = p1 - p3
        c_dir = np.cross(v1, v2)
        norms = np.linalg.norm(c_dir, axis=1)
        
        valid = norms > 1e-12
        if not np.any(valid):
            return np.pi, grad # Degenerate case, return large radius
            
        c = np.zeros_like(c_dir)
        c[valid] = c_dir[valid] / norms[valid, np.newaxis]
        
        # Orient circumcenter towards the centroid of the facet (inside the sphere)
        centroids = (p1 + p2 + p3) / 3.0
        c_norms = np.linalg.norm(centroids, axis=1)
        valid_centroids = c_norms > 1e-12
        centroids[valid_centroids] /= c_norms[valid_centroids, np.newaxis]
        
        dots = np.sum(c * centroids, axis=1)
        c[dots < 0] *= -1
        
        # Compute angular radius theta = arccos(c . p1)
        cos_theta = np.sum(c * p1, axis=1)
        cos_theta = np.clip(cos_theta, -1.0 + 1e-12, 1.0 - 1e-12)
        theta = np.arccos(cos_theta)
        
        # Find the max radius
        max_idx = np.argmax(theta)
        max_radius = theta[max_idx]
        
        if not valid[max_idx]:
            return max_radius, grad
            
        # Compute gradient for the worst facet
        # We want to minimize theta. 
        # theta = arccos(u), u = c . p1
        # d theta / d u = -1 / sqrt(1 - u^2)
        # We need d u / d p_i for i=1,2,3.
        # Note: c depends on p1, p2, p3.
        # This is complex. A simpler, robust heuristic often used in these problems:
        # Move vertices away from the circumcenter if they are "pulling" it, or towards it?
        # Actually, the standard gradient flow for minimizing max circumradius:
        # The circumcenter c is the point equidistant from p1, p2, p3.
        # The gradient of the distance from c to p1 w.r.t p1 is roughly (p1 - c)/||p1-c|| ? No.
        
        # Let's use the property that moving p1 in the direction of the projection of (c - p1) onto the tangent plane at p1 
        # reduces the distance to c. But c moves too.
        
        # A known effective method: 
        # For the worst facet, push the vertices in the direction of the negative gradient of the radius.
        # Approximation: Move vertices towards the circumcenter of the *adjacent* Voronoi cell? No.
        
        # Let's use the analytic gradient of the spherical circumradius.
        # Reference: "Optimization of Spherical Coverings"
        # The gradient of the circumradius R with respect to vertex p_i is:
        # grad R = - (1/sin R) * (projection of (c - p_i) onto tangent space at p_i) ?
        # Actually, if we fix the other two points, moving p1 towards c reduces R.
        # But moving p1 changes c.
        
        # Let's use a simpler, verified heuristic for the gradient step:
        # The force on a vertex in a worst facet should be towards the circumcenter, 
        # but weighted to maintain the triangle shape?
        
        # Actually, there is a simpler geometric insight:
        # If we move all 3 vertices of the worst facet in the direction of the circumcenter, 
        # the triangle shrinks and the circumradius decreases.
        # However, this might increase the radius of neighboring facets.
        # The gradient of the max function is the gradient of the active facet.
        
        # Let's compute the gradient of theta w.r.t p1, p2, p3 for the active facet.
        # theta = arccos( c(p1,p2,p3) . p1 )
        
        # To keep it robust and fast, we will use a "Repulsion from Circumcenter" logic for the worst facet,
        # but actually, we want to *fill* the hole. The hole is the cap centered at c.
        # The vertices are on the boundary.
        # If we move vertices *towards* the centroid of the facet (which is close to c), the triangle gets smaller?
        # No, the circumradius is the distance from c to the vertices.
        # To minimize this distance, we should move vertices towards c.
        
        # Let's try this specific gradient update:
        # For the worst facet (i1, i2, i3) with circumcenter c:
        # g_i = c - p_i  (vector from vertex to center)
        # Normalize and project onto tangent plane at p_i?
        # Since p_i is on the sphere, the tangent direction is (c - p_i) - (p_i . (c - p_i)) p_i.
        # But (c - p_i) is already roughly orthogonal to p_i if c is the circumcenter?
        # c . p_i = cos(theta). p_i . p_i = 1.
        # The component of (c - p_i) along p_i is (c . p_i - 1) = cos(theta) - 1.
        # So the tangent component is (c - p_i) - (cos(theta) - 1) p_i = c - cos(theta) p_i.
        
        # So the direction to move p_i to reduce theta is along (c - cos(theta) p_i).
        
        c_worst = c[max_idx]
        theta_worst = theta[max_idx]
        cos_theta_worst = np.cos(theta_worst)
        
        i1, i2, i3 = facets[max_idx]
        
        # Gradient for p1
        g1 = c_worst - cos_theta_worst * points[i1]
        g2 = c_worst - cos_theta_worst * points[i2]
        g3 = c_worst - cos_theta_worst * points[i3]
        
        # Normalize gradients to have unit length for consistent step size
        n1 = np.linalg.norm(g1)
        n2 = np.linalg.norm(g2)
        n3 = np.linalg.norm(g3)
        
        if n1 > 1e-12: grad[i1] += g1 / n1
        if n2 > 1e-12: grad[i2] += g2 / n2
        if n3 > 1e-12: grad[i3] += g3 / n3
        
        return max_radius, grad
        
    except Exception:
        return np.pi, np.zeros_like(points)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius_and_grad(points)[0]
    
    # Optimization Parameters
    lr = 0.01
    min_lr = 1e-6
    stall_count = 0
    stall_threshold = 50
    
    for iter in range(5000):
        if time.time() > deadline:
            break
            
        radius, grad = _covering_radius_and_grad(points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
            stall_count = 0
        else:
            stall_count += 1
            
        # Adaptive Learning Rate
        if stall_count > stall_threshold:
            lr *= 0.9
            if lr < min_lr:
                # Try a small random perturbation to escape local minima
                points += 0.01 * np.random.randn(N, 3)
                norms = np.linalg.norm(points, axis=1, keepdims=True)
                points = points / norms
                lr = 0.005
                stall_count = 0
                continue
                
        # Apply Gradient Step
        # We want to minimize radius, so we move in the direction of the gradient (which points towards reducing radius)
        points += lr * grad
        
        # Project to Sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
    # Final check
    final_radius, _ = _covering_radius_and_grad(points)
    if final_radius < best_radius:
        best_points = points.copy()
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Covering radius: {_covering_radius_and_grad(pts)[0] * 180.0 / np.pi} degrees")
    print(pts.shape)
