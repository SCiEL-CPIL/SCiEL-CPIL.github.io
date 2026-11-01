"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import time

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms

def _get_circumradii(points):
    """
    Calculate the circumradii of the Delaunay triangles on the sphere.
    Returns an array of circumradii (in radians).
    """
    # Delaunay triangulation in 3D
    try:
        tri = Delaunay(points)
    except Exception:
        # Fallback if triangulation fails (e.g., points are coplanar or too close)
        # Use pairwise distances as a rough proxy for worst-case gaps
        from scipy.spatial.distance import cdist
        dists = cdist(points, points)
        np.fill_diagonal(dists, 0)
        # Max min distance is not the covering radius, but a lower bound
        # We return a large value to penalize failure
        return np.array([1.0]) 

    radii = []
    # Iterate over simplices (triangles in 3D)
    for simplex in tri.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        # Calculate circumcenter of the triangle p1, p2, p3
        # The circumcenter is the point equidistant from p1, p2, p3.
        # For points on a sphere, the circumcenter of the spherical triangle
        # is the intersection of the sphere with the line passing through
        # the circumcenter of the planar triangle and the origin?
        # No, the circumcenter of the spherical triangle is the point on the sphere
        # that is equidistant from the three vertices.
        # This is the same as the circumcenter of the planar triangle formed by p1, p2, p3
        # ONLY IF the triangle is small. For large triangles, the circumcenter of the
        # planar triangle projected to the sphere is NOT the spherical circumcenter.
        
        # Correct method:
        # Find the point C on the unit sphere such that |C - p1| = |C - p2| = |C - p3|.
        # This implies C is orthogonal to (p1 - p2) and (p1 - p3)? No.
        # C is in the plane defined by p1, p2, p3? No.
        # C is the intersection of the unit sphere with the line passing through the
        # circumcenter of the planar triangle p1p2p3?
        # Let's derive.
        # |C|^2 = 1.
        # |C - p1|^2 = |C|^2 - 2 C.p1 + |p1|^2 = 1 - 2 C.p1 + 1 = 2 - 2 C.p1.
        # So minimizing/maximizing distance is equivalent to maximizing/minimizing C.p1.
        # We want C such that C.p1 = C.p2 = C.p3.
        # C is perpendicular to p1 - p2 and p1 - p3.
        # So C is parallel to the cross product (p1 - p2) x (p1 - p3)?
        # Let v1 = p1 - p2, v2 = p1 - p3.
        # C is orthogonal to v1 and v2.
        # So C is parallel to v1 x v2.
        # Let n = (p1 - p2) x (p1 - p3).
        # Then C = n / ||n|| or C = -n / ||n||.
        # We must choose the sign such that C is on the "same side" as the triangle normal?
        # Actually, there are two such points on the sphere. One is the circumcenter, the other is its antipode.
        # The circumcenter of the spherical triangle (the one inside the triangle) is the one
        # that has a positive dot product with the normal of the triangle plane pointing outwards?
        # Let's check the distance.
        
        v1 = p1 - p2
        v2 = p1 - p3
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            # Degenerate triangle
            continue
            
        c1 = n / norm_n
        c2 = -c1
        
        # Calculate distance from c1 to p1
        d1 = np.arccos(np.clip(np.dot(c1, p1), -1.0, 1.0))
        d2 = np.arccos(np.clip(np.dot(c2, p1), -1.0, 1.0))
        
        # The circumradius is the distance to the nearest vertex.
        # For the spherical triangle, the circumcenter is the one that lies within the triangle.
        # Usually, the circumradius is defined by the center that is "inside".
        # If the triangle is acute, the center is inside. If obtuse, it's outside.
        # The covering radius is determined by the *farthest* point in the cell.
        # The farthest point in a Voronoi cell (which is a spherical polygon) is a vertex of the cell.
        # The vertices of the Voronoi cells are the circumcenters of the Delaunay triangles.
        # So we just need the distance from the circumcenter to the vertices.
        # Both c1 and c2 are equidistant from p1, p2, p3.
        # The distance is the same for both?
        # |c1 - p1| = |c2 - p1|?
        # c1.p1 = -c2.p1.
        # dist^2 = 2 - 2 c.p1.
        # So the distances are different unless c.p1 = 0.
        # One circumcenter is "close" to the triangle, the other is "far" (antipodal).
        # The Voronoi vertex is the one that is "close" to the triangle vertices.
        # So we pick the one with the smaller distance to the vertices?
        # Yes, the Voronoi cell is the set of points closer to p_i than others.
        # The vertex of the Voronoi cell is the circumcenter that is "near" the triangle.
        
        # Let's pick the one with the smaller angular distance to p1.
        if d1 < d2:
            r = d1
        else:
            r = d2
            
        radii.append(r)
        
    if not radii:
        return np.array([1.0])
        
    return np.array(radii)

def _smooth_objective(params, beta=10.0):
    """
    Smooth approximation of the covering radius.
    params: flat array of shape (N*3,)
    """
    points = params.reshape(N, 3)
    points = _normalize(points)
    
    radii = _get_circumradii(points)
    if len(radii) == 0:
        return 1.0
        
    # Use log-sum-exp for stability
    # We want to minimize the max radius.
    # f = (1/beta) * log(sum(exp(beta * r^2)))
    # This approximates max(r^2).
    
    # Clip radii to avoid overflow
    r_sq = radii**2
    max_r_sq = np.max(r_sq)
    if max_r_sq > 50 / beta:
        # Overflow risk, just return max
        return np.sqrt(max_r_sq)
        
    log_sum_exp = np.log(np.sum(np.exp(beta * (r_sq - max_r_sq)))) + beta * max_r_sq
    val = np.sqrt(log_sum_exp / beta)
    
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_score = _smooth_objective(best_points.flatten(), beta=10.0)
    
    # 2. Lloyd's Algorithm to minimize sum of squared distances (variance)
    # This helps find a good local minimum for the covering radius as well.
    for _ in range(20):
        if time.time() > deadline:
            break
            
        # Assign points to nearest center
        # Using cdist is efficient
        from scipy.spatial.distance import cdist
        dists = cdist(points, points)
        # Note: dists[i,j] is distance between points[i] and points[j]
        # We want to assign each point in a dense grid? No, Lloyd's on the points themselves
        # minimizes the energy of the configuration.
        # Standard Lloyd's:
        # 1. Compute Voronoi cells.
        # 2. Move centers to centroids.
        
        # Approximation: Use the Delaunay triangles to find neighbors.
        # For each point, average the positions of its neighbors weighted by distance?
        # Or simply: for each point, find its Voronoi cell centroid.
        # Computing Voronoi centroids on the sphere is complex.
        # Alternative: Repulsive force method.
        
        # Let's use a simple repulsive force step:
        # Force on i is sum over j != i of (p_i - p_j) / ||p_i - p_j||^2
        # This pushes points apart.
        
        forces = np.zeros_like(points)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = points[i] - points[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-10:
                    continue
                # Repulsive force magnitude 1/d^2
                forces[i] += diff / dist_sq
                
        # Gradient step
        # We want to minimize the sum of potential energy.
        # Move points in the direction of the force?
        # The force is the gradient of the repulsive potential.
        # So we move AGAINST the gradient to minimize?
        # Potential V = sum 1/d. Gradient is sum (p_i - p_j)/d^2.
        # So p_i -= alpha * gradient.
        
        # Normalize forces to control step size
        norm_forces = np.linalg.norm(forces, axis=1, keepdims=True)
        norm_forces[norm_forces == 0] = 1.0
        normalized_forces = forces / norm_forces
        
        # Adaptive step size
        step = 0.01
        new_points = points - step * normalized_forces
        new_points = _normalize(new_points)
        
        # Check if it improved the smooth objective
        new_score = _smooth_objective(new_points.flatten(), beta=10.0)
        if new_score < best_score:
            best_score = new_score
            best_points = new_points.copy()
            points = new_points.copy()
        else:
            # Try smaller step or different direction?
            # For now, just keep old points if it didn't improve
            pass

    # 3. Local Optimization using Scipy
    # We use the best points found so far as the starting point.
    x0 = best_points.flatten()
    
    # Define the objective function for minimize
    def objective(x):
        return _smooth_objective(x, beta=20.0) # Higher beta for sharper max approximation
    
    # Use Nelder-Mead or Powell. Powell is often better for this.
    # But Nelder-Mead is derivative-free and robust.
    # Given 150 dimensions, it might be slow.
    # Let's try BFGS with numerical gradients.
    
    try:
        res = minimize(
            objective, 
            x0, 
            method='BFGS', 
            options={'maxiter': 100, 'disp': False, 'gtol': 1e-5}
        )
        
        if res.success and res.fun < best_score:
            best_points = _normalize(res.x.reshape(N, 3))
            best_score = res.fun
            
        if time.time() > deadline:
            return best_points
            
        # Try Nelder-Mead if BFGS didn't help much or failed
        res_nm = minimize(
            objective, 
            best_points.flatten(), 
            method='Nelder-Mead', 
            options={'maxiter': 50, 'disp': False}
        )
        
        if res_nm.fun < best_score:
            best_points = _normalize(res_nm.x.reshape(N, 3))
            best_score = res_nm.fun
            
    except Exception:
        pass
        
    return best_points

# EVOLVE-BLOCK-END
