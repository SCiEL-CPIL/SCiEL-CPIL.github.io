"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
import math

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _get_circumcenters(pts, simplices):
    """
    Compute the circumcenter (farthest point) for each simplex (triangle) on the sphere.
    For a spherical triangle defined by unit vectors v0, v1, v2, the circumcenter c 
    is the point on the unit sphere equidistant from v0, v1, v2.
    c is in the direction of n = v0 x v1 + v1 x v2 + v2 x v0.
    If the circumcenter falls outside the triangle (spherical), the farthest point 
    is one of the edge midpoints. We approximate the 'worst point' for the facet 
    as the circumcenter if it is 'inside' (defined by dot products with normals), 
    otherwise the midpoint of the longest edge.
    """
    centers = np.zeros((len(simplices), 3))
    for i, simplex in enumerate(simplices):
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Vector normal to the plane of the triangle, weighted by side lengths?
        # Actually, the circumcenter direction is proportional to the sum of cross products.
        n = np.cross(v0, v1) + np.cross(v1, v2) + np.cross(v2, v0)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-10:
            # Degenerate case: use centroid
            c = (v0 + v1 + v2) / 3.0
            c = c / np.linalg.norm(c)
        else:
            c = n / n_norm
            
        # Check if c is inside the spherical triangle.
        # A point c is inside if it is in the same hemisphere as the opposite vertex 
        # for each edge.
        # Edge v0-v1: normal n01 = v0 x v1. v2 should be on one side, c on same side.
        n01 = np.cross(v0, v1)
        n12 = np.cross(v1, v2)
        n20 = np.cross(v2, v0)
        
        # Dot products to check side
        s1 = np.dot(v2, n01)
        s2 = np.dot(v0, n12)
        s3 = np.dot(v1, n20)
        
        t1 = np.dot(c, n01)
        t2 = np.dot(c, n12)
        t3 = np.dot(c, n20)
        
        inside = (s1 * t1 >= -1e-12) and (s2 * t2 >= -1e-12) and (s3 * t3 >= -1e-12)
        
        if inside:
            centers[i] = c
        else:
            # If outside, the farthest point is likely near the longest edge midpoint.
            # Calculate distances between vertices to find longest edge
            d01 = np.linalg.norm(v0 - v1)
            d12 = np.linalg.norm(v1 - v2)
            d20 = np.linalg.norm(v2 - v0)
            
            max_d = max(d01, d12, d20)
            if max_d == d01:
                c = (v0 + v1) / np.linalg.norm(v0 + v1)
            elif max_d == d12:
                c = (v1 + v2) / np.linalg.norm(v1 + v2)
            else:
                c = (v2 + v0) / np.linalg.norm(v2 + v0)
            centers[i] = c
            
    return centers

def _covering_radius_and_grad(pts):
    """
    Compute covering radius and its gradient with respect to the points.
    Uses a smooth-max approximation for the maximum distance.
    """
    pts = _normalize(pts)
    
    # Build Delaunay triangulation on the sphere
    # Note: Delaunay in 3D gives tetrahedra. We need the surface triangulation.
    # Using ConvexHull to get the surface simplices is robust.
    try:
        hull = ConvexHull(pts)
        simplices = hull.simplices
    except Exception:
        # Fallback if hull fails: use a grid-based estimate or return large value
        return 180.0, np.zeros_like(pts)

    # Get circumcenters (farthest points) for each facet
    centers = _get_circumcenters(pts, simplices)
    
    # Compute angular distance from each center to its nearest point in the facet
    # Actually, for the covering radius of the facet, the distance is from the center 
    # to any of the vertices (since it's the circumcenter, dist to v0 = dist to v1 = dist to v2).
    # But if we used a midpoint fallback, the distance is to the endpoints of that edge.
    
    # Let's compute the distance from each center to the closest of its 3 vertices.
    # For the smooth max gradient, we need d_i = angular distance from center_i to nearest vertex.
    
    # Vectorized distance calculation
    # centers: (F, 3)
    # pts: (N, 3)
    # For each center, find min distance to its 3 vertices.
    
    # We can just use the distance to the first vertex v0 for the circumcenter case, 
    # but for the midpoint case, it's half the edge length.
    # To be safe and differentiable, let's compute the distance from center to the 
    # centroid of the triangle? No, that's not the covering radius.
    
    # The covering radius of the triangle is the max distance from any point in the 
    # triangle to the nearest vertex. This is exactly the circumradius if the 
    # circumcenter is inside, otherwise it's the half-length of the longest edge.
    
    # Let's compute these values directly for the objective.
    # And for the gradient, we use the subgradient of the max function.
    
    radii = np.zeros(len(simplices))
    # We need to know which vertex was used for the distance calculation to compute gradient.
    # Let's store the index of the vertex that defines the radius for each simplex.
    # Actually, for a circumcenter, distance to v0 is the radius.
    # For a midpoint of v0-v1, distance to v0 is the radius.
    
    # Let's re-evaluate centers and radii together to track the "active" vertex.
    active_verts = np.zeros(len(simplices), dtype=int)
    
    for i, simplex in enumerate(simplices):
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        n = np.cross(v0, v1) + np.cross(v1, v2) + np.cross(v2, v0)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-10:
            c = (v0 + v1 + v2) / 3.0
            c = c / np.linalg.norm(c)
            # Use v0 as active
            active_verts[i] = simplex[0]
            cos_ang = np.clip(np.dot(c, v0), -1.0, 1.0)
            radii[i] = np.arccos(cos_ang)
            continue
            
        c = n / n_norm
        
        n01 = np.cross(v0, v1)
        n12 = np.cross(v1, v2)
        n20 = np.cross(v2, v0)
        
        s1 = np.dot(v2, n01)
        s2 = np.dot(v0, n12)
        s3 = np.dot(v1, n20)
        
        t1 = np.dot(c, n01)
        t2 = np.dot(c, n12)
        t3 = np.dot(c, n20)
        
        inside = (s1 * t1 >= -1e-12) and (s2 * t2 >= -1e-12) and (s3 * t3 >= -1e-12)
        
        if inside:
            active_verts[i] = simplex[0]
            cos_ang = np.clip(np.dot(c, v0), -1.0, 1.0)
            radii[i] = np.arccos(cos_ang)
        else:
            d01 = np.linalg.norm(v0 - v1)
            d12 = np.linalg.norm(v1 - v2)
            d20 = np.linalg.norm(v2 - v0)
            
            max_d = max(d01, d12, d20)
            if max_d == d01:
                active_verts[i] = simplex[0]
                # Distance is half the chord length? No, angular distance.
                # cos(ang) = 1 - d^2/2 approx? 
                # Exact: ang = arccos(v0 . v1) / 2 ? No.
                # If c is midpoint of v0, v1, then c = (v0+v1)/|v0+v1|.
                # c . v0 = (1 + v0.v1) / |v0+v1|.
                # |v0+v1|^2 = 2 + 2 v0.v1 = 2(1 + cos(theta)).
                # c . v0 = (1 + cos(theta)) / sqrt(2(1+cos(theta))) = sqrt((1+cos(theta))/2) = cos(theta/2).
                # So ang = theta/2.
                cos_theta = np.clip(np.dot(v0, v1), -1.0, 1.0)
                radii[i] = 0.5 * np.arccos(cos_theta)
            elif max_d == d12:
                active_verts[i] = simplex[1]
                cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
                radii[i] = 0.5 * np.arccos(cos_theta)
            else:
                active_verts[i] = simplex[2]
                cos_theta = np.clip(np.dot(v2, v0), -1.0, 1.0)
                radii[i] = 0.5 * np.arccos(cos_theta)
                
    # Smooth max of radii
    # f(x) = logsumexp(beta * x) / beta
    beta = 50.0 # Sharpness
    max_r = np.max(radii)
    # Shift for numerical stability
    exp_r = np.exp(beta * (radii - max_r))
    smooth_max = max_r + np.log(np.sum(exp_r)) / beta
    
    # Gradient of smooth max: sum(exp_r / sum(exp_r)) * grad(radii)
    weights = exp_r / np.sum(exp_r)
    
    grad = np.zeros_like(pts)
    
    # Compute gradient of each radius w.r.t its active vertex
    for i in range(len(simplices)):
        w = weights[i]
        if w < 1e-6:
            continue
            
        idx = active_verts[i]
        v = pts[idx]
        
        # We need the gradient of the angular distance from the center to v.
        # This is complex because the center depends on all 3 vertices.
        # Approximation: Treat the center as fixed for the gradient step (Lloyd's algorithm 
        # effectively moves points towards the center, which is equivalent to minimizing 
        # distance to center).
        # If we move v towards c, the distance decreases.
        # Gradient of distance d(c, v) w.r.t v is - (c - v) / |c-v| in Euclidean space?
        # On the sphere, the gradient of the angular distance is in the direction of c - v (projected).
        # Let's use the direction from v to c.
        # grad_v = (c - v) / sin(d) ?
        # Actually, if we want to minimize the distance, we should move v towards c.
        # The vector (c - v) points from v to c.
        # So we add -w * (c - v) to the gradient (since we are minimizing, and we want to move v towards c).
        # Wait, the objective is to minimize the max distance. 
        # If we increase the "pull" towards c, we reduce the distance.
        # So the gradient of the loss w.r.t v should point AWAY from c? 
        # No, if v moves towards c, distance decreases. 
        # So the gradient of the distance function w.r.t v points AWAY from c.
        # We want to minimize, so we move in the direction of -gradient.
        # So -gradient points towards c.
        # Thus, we should add (c - v) to the "force" vector.
        
        # Let's compute the vector from v to c on the sphere.
        # It is simply c - v (tangent space approximation).
        # We project it to the tangent space at v: (c - v) - (v . (c - v)) v = c - (c . v) v.
        # This is the vector perpendicular to v pointing towards c.
        
        # Get the center again (we didn't store it, but we can recompute or store in loop)
        # For efficiency, let's just use the active vertex logic.
        # We need the center c for this simplex.
        # Re-computing is expensive. Let's store centers in the previous step? 
        # No, let's just approximate the direction.
        # Actually, the standard Lloyd step is: move v towards the centroid of the Voronoi cell.
        # Here the "cell" is the triangle, and the "centroid" (farthest point) is c.
        # So we just want to move v towards c.
        
        # Let's re-calculate c for this specific simplex to get the direction.
        simplex = simplices[i]
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        n = np.cross(v0, v1) + np.cross(v1, v2) + np.cross(v2, v0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            c = (v0 + v1 + v2) / 3.0
            c = c / np.linalg.norm(c)
        else:
            c = n / n_norm
            # Check inside again? For gradient direction, even if outside, 
            # the "attractor" is the circumcenter. If it's outside, the attractor 
            # should be the midpoint.
            # For simplicity and robustness in gradient descent, let's always use 
            # the circumcenter direction. If it's outside, the gradient might 
            # push it out, but the smooth max will re-weight.
            # Actually, if it's outside, the distance is determined by the edge.
            # Moving v towards the circumcenter might increase the edge length?
            # Let's stick to the circumcenter direction as a heuristic for "balance".
            
        # Direction from v to c
        dir = c - v
        # Project to tangent space
        dir = dir - np.dot(dir, v) * v
        
        # Add to gradient
        # We want to move v towards c, so the gradient of the loss should be 
        # opposite to the direction of motion.
        # Motion is along dir.
        # So gradient contribution is -w * dir?
        # If we do v_new = v - lr * grad, and grad = -w * dir, then v_new = v + lr * w * dir.
        # This moves v towards c. Correct.
        
        grad[idx] -= w * dir
        
    # Normalize points after gradient step? 
    # The gradient is in the ambient space, but we constrain to sphere.
    # The projection to tangent space handles the constraint.
    
    return smooth_max, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_pts = None
    best_score = float('inf')
    
    # 1. Generate initial candidates
    candidates = []
    
    # Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    candidates.append(fib_pts)
    
    # Random starts
    for seed in range(10):
        rng = np.random.RandomState(seed)
        pts = rng.randn(N, 3)
        pts = _normalize(pts)
        candidates.append(pts)
        
    # Evaluate candidates
    for pts in candidates:
        if time.time() > deadline - 20.0:
            break
        try:
            # Use a fast evaluation for initial ranking
            # Just use the smooth max from the gradient function
            score, _ = _covering_radius_and_grad(pts)
            if score < best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue
            
    if best_pts is None:
        best_pts = fib_pts
        
    # 2. Gradient Descent / Lloyd Refinement
    current_pts = best_pts.copy()
    current_score = best_score
    
    lr = 0.05
    momentum = 0.9
    v = np.zeros_like(current_pts)
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline - 10.0 and no_improve < max_no_improve:
        score, grad = _covering_radius_and_grad(current_pts)
        
        if not np.isfinite(score):
            break
            
        # Update velocity
        v = momentum * v - lr * grad
        
        # Update points
        new_pts = current_pts + v
        
        # Project back to sphere
        new_pts = _normalize(new_pts)
        
        # Check for improvement
        new_score, _ = _covering_radius_and_grad(new_pts)
        
        if new_score < current_score - 1e-7:
            current_pts = new_pts
            current_score = new_score
            no_improve = 0
            if current_score < best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            no_improve += 1
            # Reduce learning rate
            lr *= 0.95
            v *= 0.5 # Dampen momentum
            
    # 3. Final local search with smaller steps
    if time.time() < deadline - 5.0:
        current_pts = best_pts.copy()
        current_score = best_score
        lr = 0.01
        v = np.zeros_like(current_pts)
        no_improve = 0
        
        while time.time() < deadline - 5.0 and no_improve < 50:
            score, grad = _covering_radius_and_grad(current_pts)
            if not np.isfinite(score):
                break
            v = 0.9 * v - lr * grad
            new_pts = _normalize(current_pts + v)
            new_score, _ = _covering_radius_and_grad(new_pts)
            
            if new_score < current_score - 1e-8:
                current_pts = new_pts
                current_score = new_score
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
            else:
                no_improve += 1
                lr *= 0.9
                v *= 0.5

    return best_pts
# EVOLVE-BLOCK-END
