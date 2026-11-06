"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import math

def _compute_covering_radius_and_worst_point(points):
    """
    Computes the covering radius in radians and the point on the sphere
    that is farthest from the nearest configuration point (center of the largest empty cap).
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback if hull fails (e.g., coplanar or degenerate)
        # Use a brute force grid sampling
        # But for N=50, hull should work unless points are coincident
        # Add small jitter if needed, but we assume valid points
        return np.pi, np.array([1.0, 0.0, 0.0])

    max_rad = 0.0
    worst_point = np.array([1.0, 0.0, 0.0])
    
    for simplex in hull.simplices:
        p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # Calculate circumcenter of the spherical triangle p1, p2, p3
        # The circumcenter c is the normalized vector:
        # c = (p1 * sin(2*A) + p2 * sin(2*B) + p3 * sin(2*C)) ? No.
        # Correct formula for circumcenter of spherical triangle with vertices v1, v2, v3:
        # c = (v1 * sin(l23) * sin(l31) + v2 * sin(l31) * sin(l12) + v3 * sin(l12) * sin(l23)) / norm
        # where l12, l23, l31 are side lengths.
        # Wait, the standard formula is:
        # c = (v1 * sin(2*A) + v2 * sin(2*B) + v3 * sin(2*C)) is for incenter?
        # Let's use the cross product method which is robust:
        # The circumcenter lies on the normal to the plane defined by the triangle? No, it's on the sphere.
        # The vector c is proportional to (v1 x v2) x (v2 x v3) ... no.
        
        # Let's solve the linear system for c = x*v1 + y*v2 + z*v3
        # c . v1 = c . v2 = c . v3 = cos(R)
        # Since |c|=1, c is in the span of v1, v2, v3.
        # Let G be the Gram matrix of v1, v2, v3.
        # v1 . v1 = 1, v2 . v2 = 1, v3 . v3 = 1
        # v1 . v2 = d12, v2 . v3 = d23, v3 . v1 = d31
        # We want c such that c.v1 = c.v2 = c.v3 = k
        # c = alpha * (v1 + v2 + v3) is only true if symmetric.
        # General solution:
        # c = M * w, where M = [v1, v2, v3] and w = [w1, w2, w3]^T
        # c.v1 = w1 + w2*d12 + w3*d31 = k
        # c.v2 = w1*d12 + w2 + w3*d23 = k
        # c.v3 = w1*d31 + w2*d23 + w3 = k
        # Subtracting equations:
        # (1-d12)w1 + (d31-d23)w3 = 0  => w1 = - (d31-d23)/(1-d12) w3
        # (1-d12)w1 - (1-d23)w2 + (d31-d23)w3 = 0 ...
        
        # Easier: Use the formula for circumcenter of spherical triangle:
        # c = (v1 * sin(b) * sin(c) + v2 * sin(a) * sin(c) + v3 * sin(a) * sin(b)) / norm
        # where a, b, c are side lengths opposite to v1, v2, v3?
        # No, standard formula:
        # c = (v1 * sin(2*A) + v2 * sin(2*B) + v3 * sin(2*C)) is incorrect for circumcenter.
        
        # Let's use the property that c is perpendicular to (v1 x v2) x c?
        # Actually, the circumcenter c satisfies (c - v1) . (v1 x v2) = 0? No.
        
        # Reliable method:
        # The circumcenter c is the normalized sum of the three vertices weighted by the sines of the opposite angles?
        # Let's use the explicit vector formula:
        # c = (v1 * sin(l23) * sin(l31) + v2 * sin(l31) * sin(l12) + v3 * sin(l12) * sin(l23)) / norm
        # where l12 = angle(v1, v2), etc.
        
        d12 = np.dot(p1, p2)
        d23 = np.dot(p2, p3)
        d31 = np.dot(p3, p1)
        
        l12 = np.arccos(np.clip(d12, -1, 1))
        l23 = np.arccos(np.clip(d23, -1, 1))
        l31 = np.arccos(np.clip(d31, -1, 1))
        
        # If triangle is degenerate, skip
        if l12 < 1e-6 or l23 < 1e-6 or l31 < 1e-6:
            continue
            
        w1 = np.sin(l23) * np.sin(l31)
        w2 = np.sin(l31) * np.sin(l12)
        w3 = np.sin(l12) * np.sin(l23)
        
        c_vec = w1 * p1 + w2 * p2 + w3 * p3
        c_norm = np.linalg.norm(c_vec)
        
        if c_norm < 1e-10:
            # Degenerate, try centroid
            c_vec = (p1 + p2 + p3) / 3.0
            c_norm = np.linalg.norm(c_vec)
            if c_norm < 1e-10:
                continue
                
        c_point = c_vec / c_norm
        
        # Check if this circumcenter is actually the center of the empty cap.
        # For a convex hull, the circumcenter of a facet is the center of the empty cap IF the facet is part of the Delaunay triangulation.
        # For points on a sphere, the convex hull facets correspond to the Delaunay triangulation.
        # However, we must ensure the circumcenter is on the "outer" side.
        # The normal to the facet is n = (p1 x p2) x (p2 x p3) ... or just p1 x p2 for the edge?
        # The center of the sphere is the origin. The facet is a triangle on the sphere.
        # The circumcenter c_point is a point on the sphere.
        # The distance from c_point to p1 is the circumradius.
        
        cos_R = np.dot(c_point, p1)
        cos_R = np.clip(cos_R, -1, 1)
        R = np.arccos(cos_R)
        
        # We need to make sure this cap is empty.
        # Since it's a convex hull facet, the interior of the sphere is on one side.
        # The "empty cap" is the region on the sphere visible from the origin through this facet?
        # Actually, for a set of points on the sphere, the convex hull is a polyhedron containing the origin.
        # The facets are triangles. The "covering radius" is the max distance from any point on the sphere to the nearest point.
        # The worst point is the circumcenter of the facet with the largest circumradius, provided the circumcenter is in the "empty" region relative to the other points.
        # For a convex hull of points on the sphere, the circumcenter of a facet is the center of the largest empty circle (cap) centered at that facet's circumcenter, IF the facet is Delaunay.
        # All facets of the convex hull are Delaunay.
        
        if R > max_rad:
            max_rad = R
            worst_point = c_point

    return max_rad, worst_point


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Annealing / Iterative Refinement
    # We will use a combination of repulsive forces and targeted moves towards the worst point.
    
    temperature = 0.1
    decay = 0.99
    iterations = 2000
    
    for it in range(iterations):
        # Calculate forces
        # Repulsive force: F_i = sum_{j != i} (p_i - p_j) / ||p_i - p_j||^2
        # This maximizes the minimum distance.
        
        # Vectorized distance calculation
        # points: (N, 3)
        # diff: (N, N, 3)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        
        # Avoid division by zero
        np.fill_diagonal(dist_sq, 1.0)
        
        # Force magnitude: 1 / dist_sq
        # Force vector: diff / dist_sq^1.5 ? No, Coulomb is 1/r^2.
        # F_ij = (p_i - p_j) / ||p_i - p_j||^2
        # This is equivalent to diff / dist_sq^(3/2) * dist_sq? No.
        # ||p_i - p_j|| = sqrt(dist_sq).
        # F = diff / dist_sq.
        
        # We want to move points apart.
        # F_i = sum_{j != i} (p_i - p_j) / ||p_i - p_j||^2
        # Note: (p_i - p_j) / ||p_i - p_j||^2 is the gradient of -1/||p_i - p_j||.
        # So minimizing sum 1/||p_i - p_j|| maximizes distances.
        
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        # Force = sum_j (p_i - p_j) * (1 / ||p_i - p_j||^2)
        # diff is (N, N, 3). inv_dist_sq is (N, N).
        # weights = inv_dist_sq[:, :, np.newaxis]
        # forces = np.sum(diff * weights, axis=1)
        
        weights = inv_dist_sq[:, :, np.newaxis]
        forces = np.sum(diff * weights, axis=1)
        
        # Add noise
        noise = np.random.normal(0, temperature, size=(N, 3))
        
        # Update positions
        points += forces * 0.01 + noise
        
        # Project back to sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        points = points / norms
        
        # Targeted move: Find worst point and push nearest points towards it?
        # No, push nearest points AWAY from the worst point?
        # The worst point is the center of the largest empty cap.
        # To reduce the covering radius, we should move the points defining the cap TOWARDS the center of the cap.
        
        if it % 10 == 0: # Do this periodically to save time
            max_rad, worst_point = _compute_covering_radius_and_worst_point(points)
            
            # Find the 3 nearest points to the worst_point
            dots = np.dot(points, worst_point)
            # We want points that are CLOSE to the worst point?
            # No, the worst point is FAR from all points.
            # The points defining the cap are the ones with the LARGEST angular distance to the worst point?
            # No, the circumcenter is the point. The vertices of the triangle are the points.
            # The distance from the circumcenter to the vertices is the circumradius.
            # To reduce the circumradius, we should move the vertices TOWARDS the circumcenter.
            
            # Identify the 3 points closest to the worst_point?
            # Actually, the points on the hull facet are the ones that define the cap.
            # They are the ones for which the angular distance to the worst_point is roughly the covering radius.
            # Let's find the 3 points with the largest dot product with the worst_point (closest on sphere).
            # Wait, if the worst_point is the center of the empty cap, the vertices of the cap are the points on the boundary.
            # The angular distance from the center to the boundary is the covering radius.
            # So the vertices are the points with distance ~ max_rad from the worst_point.
            # This means dot product is cos(max_rad).
            # Other points are further away (smaller dot product).
            # So we want the 3 points with the LARGEST dot product with worst_point.
            
            indices = np.argsort(dots)[-3:]
            
            # Move these points towards the worst_point
            # Step size proportional to temperature
            step_size = 0.01 * temperature
            for idx in indices:
                # Move p_idx towards worst_point
                # p_new = p_idx + step * (worst_point - p_idx)
                # Then normalize
                new_p = points[idx] + step_size * (worst_point - points[idx])
                points[idx] = new_p / np.linalg.norm(new_p)
        
        temperature *= decay
        if temperature < 1e-6:
            break
            
    # Final cleanup: ensure no coincident points
    # Check for very close points and separate them
    for _ in range(10):
        min_dist = np.inf
        min_i, min_j = 0, 1
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points[i] - points[j])
                if d < min_dist:
                    min_dist = d
                    min_i, min_j = i, j
        if min_dist > 1e-4:
            break
        # Separate them
        mid = (points[min_i] + points[min_j]) / 2
        mid_norm = np.linalg.norm(mid)
        if mid_norm < 1e-6:
            perp = np.array([1.0, 0.0, 0.0])
        else:
            perp = mid / mid_norm
        # Move i and j in opposite directions perpendicular to their midpoint
        # Simple: move i away from j and j away from i
        v = points[min_i] - points[min_j]
        v = v / np.linalg.norm(v)
        points[min_i] += 0.01 * v
        points[min_j] -= 0.01 * v
        points[min_i] /= np.linalg.norm(points[min_i])
        points[min_j] /= np.linalg.norm(points[min_j])

    return points
# EVOLVE-BLOCK-END
