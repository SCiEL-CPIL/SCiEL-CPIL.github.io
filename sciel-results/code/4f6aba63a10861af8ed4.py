"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay
import math


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def _covering_radius(points):
    """Compute the covering radius in degrees.
    
    Uses the Delaunay triangulation on the sphere: the covering radius is the
    maximum circumradius of the spherical triangles (facets of the convex hull).
    """
    try:
        # Project to 2D via stereographic projection or use 3D Delaunay
        # Better: use the fact that for points on the unit sphere, the covering
        # radius equals the max over all facets of the convex hull of the
        # circumradius of that facet's triangle.
        
        # For points on the unit sphere, we can compute the spherical Delaunay
        # triangulation. A simpler approach: for each point, find its 3 nearest
        # neighbors and compute the circumradius of the triangle formed by
        # the point and its two nearest neighbors... No, that's not quite right.
        
        # The correct approach: the covering radius is the maximum over all points
        # on the sphere of the distance to the nearest data point. This equals
        # the maximum circumradius of the triangles in the spherical Delaunay
        # triangulation.
        
        # For points on the unit sphere, the circumradius of a spherical triangle
        # (p1, p2, p3) can be computed as follows:
        # The circumcenter of the triangle in 3D lies along the normal to the plane
        # containing the three points. The angular distance from this circumcenter
        # to any of the three vertices is the circumradius.
        
        # Actually, a simpler and more robust approach: compute the nearest-neighbor
        # distance for a dense set of test points on the sphere.
        
        # But the task says: "the largest circumradius of the facets of the convex hull,
        # each facet checked to have an empty cap"
        
        # Let me use a direct approach: sample many points on the sphere and find
        # the max angular distance to nearest data point.
        
        # Actually, let's use the proper geometric definition. For a set of points
        # on the unit sphere, the covering radius is the supremum over all unit
        # vectors y of min_i angle(y, p_i). This is achieved at a circumcenter of
        # some spherical triangle in the Delaunay triangulation.
        
        # For 130 points, let me use a dense sampling approach for evaluation.
        
        # Create a dense grid of test points
        n_theta = 200
        n_phi = 400
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi)
        
        # Use a more efficient approach: for each data point, the Voronoi cell
        # boundary determines the covering radius. The worst case is at a vertex
        # of the Voronoi diagram (circumcenter of a Delaunay triangle).
        
        # Let me just do dense sampling - it's fast enough for 130 points
        # Actually, let's be smarter. The covering radius for N points on sphere
        # is at most ~sqrt(4*pi/N) radians. For N=130, that's about sqrt(0.096) ≈ 0.31 rad ≈ 17.8 deg.
        
        # Dense sampling:
        # Use Fibonacci spiral for test points too
        n_test = 50000
        i_test = np.arange(n_test) + 0.5
        z_test = 1.0 - 2.0 * i_test / n_test
        phi_test = np.pi * (1.0 + np.sqrt(5.0)) * i_test
        r_test = np.sqrt(np.maximum(0, 1.0 - z_test * z_test))
        test_points = np.column_stack([r_test * np.cos(phi_test), r_test * np.sin(phi_test), z_test])
        
        # Compute distances from each test point to all data points
        # Use matrix multiplication for speed
        # points shape: (N, 3), test_points shape: (M, 3)
        # dot = test_points @ points.T  shape: (M, N)
        dot = np.clip(test_points @ points.T, -1.0, 1.0)
        min_dot = np.max(dot, axis=1)  # max dot product = min angular distance
        max_angle = np.arccos(np.clip(1.0 - min_dot, -1.0, 1.0))
        return np.degrees(np.max(max_angle))
    except Exception:
        return 180.0


def _local_covering_radius_contribution(points, idx):
    """Estimate the local contribution of point idx to the covering radius.
    
    For each point, look at its k nearest neighbors and compute the circumradius
    of the triangles formed.
    """
    p = points[idx]
    # Find nearest neighbors
    dots = points @ p
    order = np.argsort(-dots)
    # Take the 4 nearest neighbors (including self)
    neighbors = [order[1], order[2], order[3]]
    
    max_circ_r = 0.0
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            p1 = points[neighbors[i]]
            p2 = points[neighbors[j]]
            # Compute circumradius of spherical triangle (p, p1, p2)
            # The circumcenter in 3D is along the cross product direction
            # Actually, for three points on the unit sphere, the circumcenter
            # of the spherical triangle is the normalized vector in the plane
            # equidistant from all three.
            
            # The circumcenter c satisfies |c - p| = |c - p1| = |c - p2|
            # and c is on the unit sphere.
            # c is along the direction of (p × p1) × (p × p2) or similar.
            
            # A simpler approach: the circumradius is the angular distance from
            # the circumcenter to any vertex.
            
            # For three points on the unit sphere, the circumcenter can be found
            # as the normalized vector:
            # c = (p * (|p1|^2 * |p2|^2 - (p1·p2)^2) + ...) 
            # This is getting complicated. Let me use a different approach.
            
            # The circumradius of a spherical triangle with side lengths a, b, c
            # (angular distances) can be computed, but it's complex.
            
            # Simpler: the circumradius is at least half the maximum side length,
            # and at most the maximum side length.
            
            # For gradient purposes, let's approximate: the local covering radius
            # contribution is roughly the maximum angular distance to the nearest
            # neighbor among the 3 nearest neighbors.
            
            d1 = np.arccos(np.clip(p @ p1, -1, 1))
            d2 = np.arccos(np.clip(p @ p2, -1, 1))
            d12 = np.arccos(np.clip(p1 @ p2, -1, 1))
            
            # The circumradius of triangle (p, p1, p2) is at least max(d1, d2, d12)/2
            # and the actual circumradius can be larger.
            # For a rough gradient estimate, use the maximum pairwise distance.
            max_d = max(d1, d2, d12)
            max_circ_r = max(max_circ_r, max_d)
    
    return max_circ_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    points = np.array([_normalize(p) for p in points])
    
    best_points = points.copy()
    best_score = _covering_radius(best_points)
    
    # Riemannian gradient descent
    # The objective is to minimize the covering radius.
    # For each point, the gradient of the covering radius with respect to that
    # point points in the direction that increases the local covering radius.
    # So we want to move each point in the direction that decreases the local
    # covering radius, i.e., toward the center of its Voronoi cell.
    
    # A practical approach: for each point, compute the centroid of its nearest
    # neighbors' Voronoi cell approximation, and move toward it.
    
    lr = 0.01  # learning rate
    n_iter = 5000
    
    for iteration in range(n_iter):
        if time.time() > deadline - 5.0:
            break
        
        # Compute gradient for each point
        grads = np.zeros_like(points)
        
        for idx in range(N):
            p = points[idx]
            
            # Find k nearest neighbors
            dots = points @ p
            order = np.argsort(-dots)
            # Take 6 nearest neighbors (excluding self)
            k_neighbors = order[1:7]
            
            # The direction to move: toward the centroid of the Voronoi cell
            # Approximate by moving away from the direction of the largest
            # gap. The largest gap is in the direction opposite to the
            # centroid of the neighbors.
            
            # Compute the weighted centroid of neighbors
            neighbor_points = points[k_neighbors]
            
            # The Voronoi cell center is roughly in the direction of the
            # average of the neighbor directions, but weighted by inverse
            # distance (closer neighbors have more influence on the cell shape).
            
            # Actually, the correct gradient for minimizing the max circumradius
            # is: for each point, move it toward the circumcenter of the
            # Delaunay triangle that has the largest circumradius among those
            # containing this point.
            
            # Simpler heuristic: move each point toward the centroid of its
            # nearest neighbors, projected onto the tangent plane.
            
            centroid = np.mean(neighbor_points, axis=0)
            centroid = _normalize(centroid)
            
            # Project onto tangent plane at p
            tangent = centroid - np.dot(centroid, p) * p
            tangent = _normalize(tangent) if np.linalg.norm(tangent) > 1e-12 else np.zeros(3)
            
            # The magnitude of the step should be proportional to the local
            # covering radius
            local_r = _local_covering_radius_contribution(points, idx)
            
            grads[idx] = tangent * local_r
        
        # Normalize gradients
        grad_norms = np.linalg.norm(grads, axis=1)
        max_grad = np.max(grad_norms) if len(grad_norms) > 0 else 1.0
        if max_grad > 1e-12:
            grads = grads / max_grad
        
        # Riemannian gradient step
        step_size = lr
        for idx in range(N):
            p = points[idx]
            g = grads[idx]
            
            # Move along the geodesic (projected gradient)
            new_p = p + step_size * g
            new_p = _normalize(new_p)
            
            # Check if this improves or maintains the covering radius
            points[idx] = new_p
        
        # Evaluate
        if iteration % 50 == 0 or iteration == n_iter - 1:
            score = _covering_radius(points)
            if score < best_score:
                best_score = score
                best_points = points.copy()
        
        # Adaptive learning rate
        if iteration % 100 == 0 and iteration > 0:
            lr *= 0.95
    
    # Final refinement: local optimization
    if time.time() < deadline - 10.0:
        # Do a few more iterations with smaller steps
        for _ in range(500):
            if time.time() > deadline - 5.0:
                break
            
            grads = np.zeros_like(best_points)
            for idx in range(N):
                p = best_points[idx]
                dots = best_points @ p
                order = np.argsort(-dots)
                k_neighbors = order[1:7]
                neighbor_points = best_points[k_neighbors]
                centroid = np.mean(neighbor_points, axis=0)
                centroid = _normalize(centroid)
                tangent = centroid - np.dot(centroid, p) * p
                tangent = _normalize(tangent) if np.linalg.norm(tangent) > 1e-12 else np.zeros(3)
                local_r = _local_covering_radius_contribution(best_points, idx)
                grads[idx] = tangent * local_r
            
            grad_norms = np.linalg.norm(grads, axis=1)
            max_grad = np.max(grad_norms) if len(grad_norms) > 0 else 1.0
            if max_grad > 1e-12:
                grads = grads / max_grad
            
            for idx in range(N):
                p = best_points[idx]
                g = grads[idx]
                new_p = p + 0.005 * g
                best_points[idx] = _normalize(new_p)
            
            score = _covering_radius(best_points)
            if score < best_score:
                best_score = score
    
    return best_points
# EVOLVE-BLOCK-END
