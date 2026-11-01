import numpy as np
import time
from scipy.spatial import ConvexHull
import sys

N = 50

def _circumcenter_spherical(a, b, c):
    """
    Compute the circumcenter of a spherical triangle with vertices a, b, c.
    The circumcenter x satisfies x.a = x.b = x.c.
    This implies x is orthogonal to a-b and b-c.
    So x is parallel to cross(a-b, b-c).
    """
    ab = a - b
    bc = b - c
    x = np.cross(ab, bc)
    norm = np.linalg.norm(x)
    if norm < 1e-10:
        # Degenerate triangle, return centroid
        x = (a + b + c) / 3.0
        norm = np.linalg.norm(x)
        if norm < 1e-10:
            return None
        x = x / norm
        return x
    x = x / norm
    # Ensure x is on the same side as the triangle (dot product with centroid should be positive)
    centroid = (a + b + c) / 3.0
    if np.dot(x, centroid) < 0:
        x = -x
    return x

def _covering_radius_and_gradients(points):
    """
    Compute the covering radius and identify critical faces.
    Returns:
    - covering_radius: float
    - critical_faces: list of (face_indices, circumcenter)
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        # If hull fails, fall back to a simple estimate or return a large value
        # For robustness, if hull fails, we can't proceed properly.
        # But for N=50 well-distributed points, this shouldn't happen often.
        return np.pi, []

    max_dist = 0.0
    critical_faces = []
    
    # Map hull faces to indices
    # hull.simplices gives the indices of vertices for each face
    for simplex in hull.simplices:
        i, j, k = simplex
        a = points[i]
        b = points[j]
        c = points[k]
        
        x = _circumcenter_spherical(a, b, c)
        if x is None:
            continue
            
        # Check if the cap centered at x is empty
        # i.e., the closest points to x are a, b, c
        dists = np.sum(points * x, axis=1) # cos(theta)
        # We want to find if any point is closer to x than a, b, c
        # dist(x, p) < dist(x, a) <=> cos(dist) > cos(dist(x,a))
        # cos(dist(x,a)) = x.a
        cos_xa = np.dot(x, a)
        
        # Find max cos(dist) to x
        max_cos = np.max(dists)
        
        # If the max cos is from a, b, or c, and is equal to cos_xa (within tol), 
        # then the face is empty.
        # Actually, if any other point p has dist(x,p) < dist(x,a), then dist(x,p) < R, 
        # so the cap of radius R is not empty.
        # So we check if min(dist(x, p) for p in P) == dist(x, a).
        # In terms of cos: max(cos(x, p)) == cos(x, a).
        
        # Due to floating point, we check if max_cos is close to cos_xa
        # and that the indices achieving max_cos are subset of {i, j, k}
        
        if max_cos > cos_xa + 1e-7:
            # Some other point is closer to x than a is.
            # So the circumcenter of this face is not the "farthest" point in its cell?
            # Actually, if another point is closer, the Voronoi cell of a,b,c is cut off.
            # The circumcenter of the convex hull face is only a candidate if it's empty.
            continue
            
        # It is empty (or on the boundary)
        # The distance from x to a is the covering radius candidate for this face
        dist = np.arccos(np.clip(cos_xa, -1.0, 1.0))
        
        if dist > max_dist:
            max_dist = dist
            # We can keep all critical faces for the update step, 
            # or just the ones contributing to the max.
            # For a smoother gradient, we might want to include faces with dist close to max_dist.
            critical_faces.append((simplex, x, dist))
            
    # If no critical faces found, something is wrong, return 0 or pi
    if not critical_faces:
        return 0.0, []
        
    return max_dist, critical_faces

def construct():
    start_time = time.time()
    timeout = 150.0 # Safety margin
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Initial covering radius
    current_radius, _ = _covering_radius_and_gradients(points)
    best_points = points.copy()
    best_radius = current_radius
    
    step_size = 0.1
    cooling = 0.999
    
    while time.time() - start_time < timeout - 10: # Leave 10s buffer
        
        # Get critical faces
        radius, critical_faces = _covering_radius_and_gradients(points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
        
        if not critical_faces:
            break
            
        # Calculate gradient-like step
        # We want to move vertices of critical faces towards their circumcenters
        # to reduce the distance.
        # Force on vertex v from face (v, u, w) with circumcenter x:
        # If v is a vertex of the face, move v towards x.
        
        moves = np.zeros_like(points)
        
        for simplex, x, dist in critical_faces:
            # Only consider faces that are close to the maximum radius to focus effort
            if dist > best_radius * 0.95: 
                i, j, k = simplex
                # Move vertices towards x
                # The "error" is the vector from vertex to x
                # We want to reduce the distance, so move towards x.
                moves[i] += (x - points[i])
                moves[j] += (x - points[j])
                moves[k] += (x - points[k])
                
        # Normalize moves to prevent huge jumps
        move_norms = np.linalg.norm(moves, axis=1)
        # Scale moves by step_size
        # If move is 0, don't move
        moves = moves * step_size
        
        # Apply moves
        new_points = points + moves
        
        # Normalize back to sphere
        norms = np.linalg.norm(new_points, axis=1)
        # Avoid division by zero
        norms[norms < 1e-10] = 1.0
        new_points = new_points / norms[:, np.newaxis]
        
        points = new_points
        
        # Cool down step size
        step_size *= cooling
        
        # Optional: Check for convergence
        if step_size < 1e-6:
            break
            
    # Return the best points found
    # Ensure they are normalized
    norms = np.linalg.norm(best_points, axis=1)
    best_points = best_points / norms[:, np.newaxis]
    
    return best_points

# EVOLVE-BLOCK-END
