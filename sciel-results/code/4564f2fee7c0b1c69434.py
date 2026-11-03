"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _covering_radius(pts):
    """Compute the covering radius in degrees for points on the unit sphere."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 3):
        return np.inf
    if not np.all(np.isfinite(pts)):
        return np.inf
    # Check for coincident points
    norms = np.linalg.norm(pts, axis=1)
    if np.any(norms < 1e-12):
        return np.inf
    pts = pts / norms[:, None]
    # Check if all points are in a closed hemisphere
    # A set of points on S^2 is in a closed hemisphere iff there exists a vector v
    # such that v · p_i >= 0 for all i. Equivalently, the origin is in the convex hull.
    # We check if the convex hull contains the origin.
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.inf
    # Check if origin is inside the hull
    # For each facet, check if origin is on the same side as the interior
    # The interior is the side where the centroid of the hull lies
    centroid = pts.mean(axis=0)
    for simplex in hull.simplices:
        face_pts = pts[simplex]
        # Compute normal
        v1 = face_pts[1] - face_pts[0]
        v2 = face_pts[2] - face_pts[0]
        normal = np.cross(v1, v2)
        n_norm = np.linalg.norm(normal)
        if n_norm < 1e-12:
            continue
        normal = normal / n_norm
        # Distance from origin to the plane of the face
        # The face plane is: normal · x = d, where d = normal · face_pts[0]
        d = np.dot(normal, face_pts[0])
        # For the origin to be inside, we need normal · 0 <= d (if normal points outward)
        # Actually, let's check the sign consistently
        # The centroid should be on the interior side
        centroid_dist = np.dot(normal, centroid) - d
        origin_dist = 0.0 - d  # normal · 0 - d = -d
        # If centroid and origin are on opposite sides or origin is outside
        if centroid_dist * origin_dist < -1e-10:
            # Origin is outside this face, meaning it's outside the hull
            # Wait, this logic is tricky. Let me use a simpler approach.
            pass
    
    # Simpler hemisphere check: if all points lie in some hemisphere, 
    # then the convex hull does not contain the origin.
    # We can check this by seeing if there's a direction v with v·p_i >= 0 for all i.
    # This is equivalent to checking if the origin is NOT in the convex hull.
    # Use linear programming or just check: if the minimum over i of (v·p_i) can be >= 0 for some v.
    # Actually, a simpler test: the points are in a closed hemisphere iff the origin is not in the interior of the convex hull.
    # Let's just check if the convex hull contains the origin by checking all facets.
    
    # For a convex polytope in R^3, the origin is inside iff for every facet,
    # the origin is on the same side as the interior.
    # The interior side is determined by the centroid.
    inside = True
    for simplex in hull.simplices:
        face_pts = pts[simplex]
        v1 = face_pts[1] - face_pts[0]
        v2 = face_pts[2] - face_pts[0]
        normal = np.cross(v1, v2)
        n_norm = np.linalg.norm(normal)
        if n_norm < 1e-12:
            continue
        normal = normal / n_norm
        # Make sure normal points outward: check against centroid
        centroid_to_face = face_pts[0] - centroid
        if np.dot(normal, centroid_to_face) > 0:
            normal = -normal
        # Now normal points outward
        # Origin is inside if normal · 0 <= normal · face_pts[0], i.e., 0 <= d
        # where d = normal · face_pts[0]
        d = np.dot(normal, face_pts[0])
        if d < -1e-10:
            inside = False
            break
    
    if not inside:
        return np.inf
    
    # Now compute covering radius: max over facets of the circumradius of the spherical triangle
    # For each triangular facet with vertices a, b, c on the unit sphere,
    # the covering radius contribution is the circumradius of the spherical triangle.
    # The circumcenter of the spherical triangle is the point on the sphere equidistant from a, b, c.
    # It can be found as the normalized cross product of (b-a) and (c-a) ... no.
    # Actually, the circumcenter is in the direction of a + b + c only if the triangle is "small".
    # More precisely, the circumcenter is the unit vector in the direction of the vector
    # that is equidistant from a, b, c. This is the direction of the normal to the plane of a,b,c
    # projected... no.
    # The circumcenter of a spherical triangle is the point p on the sphere such that p·a = p·b = p·c.
    # This means p is perpendicular to (a-b) and (a-c), so p is parallel to (a-b) × (a-c).
    # Wait, that's the normal to the plane. But the circumcenter is on the great circle... no.
    # Let me think again. We want p on S^2 such that p·a = p·b = p·c.
    # This means (a-b)·p = 0 and (a-c)·p = 0, so p is parallel to (a-b) × (a-c).
    # So the circumcenter is ±((a-b) × (a-c)) / ||(a-b) × (a-c)||.
    # We pick the sign that gives the correct side (the one on the same side as the triangle's interior).
    
    max_circumradius = 0.0
    for simplex in hull.simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        # Compute circumcenter direction
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        cross_norm = np.linalg.norm(cross)
        if cross_norm < 1e-12:
            continue
        circumcenter_dir = cross / cross_norm
        # Determine the correct sign: the circumcenter should be on the same side as the triangle
        # Check which sign gives a point that is "inside" the spherical triangle
        # We can check by seeing if the dot products are consistent
        # The circumcenter should have the same sign of dot product with the outward normal as... 
        # Actually, let's just compute both and pick the one that gives a valid circumradius
        
        # The angular distance from circumcenter to a vertex is the circumradius
        # cos(circumradius) = circumcenter · a
        
        # Try both signs
        for sign in [1, -1]:
            p = sign * circumcenter_dir
            cos_r = np.dot(p, a)
            cos_r = np.clip(cos_r, -1.0, 1.0)
            r = np.arccos(cos_r)
            # Verify that p is equidistant from b and c
            cos_rb = np.dot(p, b)
            cos_rb = np.clip(cos_rb, -1.0, 1.0)
            rb = np.arccos(cos_rb)
            if abs(r - rb) < 1e-6:
                # This is a valid circumcenter
                # Check that it's on the correct side (interior of the spherical triangle)
                # The circumcenter should be such that the cap around it contains the triangle
                # For a small triangle, the circumcenter is the one closer to the triangle
                # We can check that the angular distance is less than pi/2 for a "small" triangle
                # But for large triangles, this might not hold. Let's just take the minimum of the two valid ones
                max_circumradius = max(max_circumradius, r)
                break
    
    return np.degrees(max_circumradius)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Step 1: Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Step 2: Optimize using soft-min surrogate for pairwise distances
    # We maximize the minimum pairwise distance using a soft-min approximation
    def soft_min_distance(pts_flat, beta=50.0):
        pts = pts_flat.reshape(N, 3)
        pts = pts / np.linalg.norm(pts, axis=1)[:, None]
        # Compute all pairwise dot products
        dots = pts @ pts.T
        # Pairwise angular distances
        cos_dists = np.clip(dots, -1.0, 1.0)
        # We want to maximize the minimum distance, so minimize the maximum of -distance
        # Use soft-min: -soft_min(d_i) = -(-log-sum-exp(-beta * d_i) / beta) = log-sum-exp(-beta * d_i) / beta
        # But we exclude the diagonal
        mask = ~np.eye(N, dtype=bool)
        cos_dists_masked = np.where(mask, cos_dists, 1.0)  # 1.0 corresponds to 0 distance
        # Actually, we want to maximize min distance, so we want to push all distances up
        # Let's use: minimize -log-sum-exp(beta * d_i) / beta, which is a soft-min of d_i
        # But d_i = arccos(cos_d_i), which is non-smooth at the boundary
        # Instead, let's work with cos distances: minimize max cos_d_i (which is equivalent to minimizing max angular distance)
        # No wait, we want to MAXIMIZE min angular distance, which is MINIMIZE max cos(angular distance)
        # So we want to minimize the soft-max of cos_dists
        # soft-max of cos_dists = log-sum-exp(beta * cos_dists) / beta
        cos_dists_for_max = np.where(mask, cos_dists, -1.0)  # -1.0 corresponds to pi distance
        max_cos = np.logsumexp(beta * cos_dists_for_max) / beta
        return max_cos
    
    def soft_min_distance_grad(pts_flat, beta=50.0):
        pts = pts_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1)
        pts_norm = pts / norms[:, None]
        dots = pts_norm @ pts_norm.T
        mask = ~np.eye(N, dtype=bool)
        cos_dists = np.where(mask, dots, -1.0)
        
        # soft-max of cos_dists
        beta_cos = beta * cos_dists
        max_val = np.max(beta_cos)
        exp_vals = np.exp(beta_cos - max_val)
        sum_exp = np.sum(exp_vals)
        softmax = exp_vals / sum_exp
        
        # Gradient of logsumexp(beta * cos_dists) / beta with respect to cos_dists
        # = softmax
        # Gradient of cos_dists with respect to pts:
        # cos_dists[i,j] = pts_norm[i] · pts_norm[j]
        # d(cos_dists[i,j])/d(pts_norm[i]) = pts_norm[j]
        # d(cos_dists[i,j])/d(pts_norm[j]) = pts_norm[i]
        
        grad_cos = np.zeros_like(cos_dists)
        grad_cos[mask] = softmax[mask]
        
        # Gradient with respect to pts_norm
        grad_pts_norm = grad_cos @ pts_norm + grad_cos.T @ pts_norm
        # Symmetrize
        grad_pts_norm = (grad_pts_norm + grad_pts_norm.T) / 2
        
        # Gradient with respect to pts (unnormalized)
        # pts_norm = pts / norms
        # d(pts_norm[i])/d(pts[i]) = (I - pts_norm[i] pts_norm[i]^T) / norms[i]
        grad_pts = np.zeros_like(pts)
        for k in range(N):
            grad_pts[k] = (grad_pts_norm[k] - np.dot(grad_pts_norm[k], pts_norm[k]) * pts_norm[k]) / norms[k]
        
        return grad_pts.flatten()
    
    # Optimize
    best_pts = pts.copy()
    best_score = _covering_radius(best_pts)
    
    for beta in [20.0, 50.0, 100.0]:
        if time.time() > deadline:
            break
        result = minimize(
            soft_min_distance,
            best_pts.flatten(),
            jac=soft_min_distance_grad,
            args=(beta,),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.success and result.fun < np.inf:
            candidate = result.x.reshape(N, 3)
            candidate = candidate / np.linalg.norm(candidate, axis=1)[:, None]
            score = _covering_radius(candidate)
            if score < best_score:
                best_score = score
                best_pts = candidate
    
    # Step 3: Local refinement using convex hull facet circumradius
    if time.time() < deadline - 20.0:
        for iteration in range(100):
            if time.time() > deadline:
                break
            
            # Compute covering radius and identify the worst facet
            pts = best_pts
            try:
                hull = ConvexHull(pts)
            except Exception:
                break
            
            max_circumradius = 0.0
            worst_simplex = None
            worst_circumcenter = None
            
            for simplex in hull.simplices:
                a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                v1 = a - b
                v2 = a - c
                cross = np.cross(v1, v2)
                cross_norm = np.linalg.norm(cross)
                if cross_norm < 1e-12:
                    continue
                circumcenter_dir = cross / cross_norm
                for sign in [1, -1]:
                    p = sign * circumcenter_dir
                    cos_r = np.dot(p, a)
                    cos_r = np.clip(cos_r, -1.0, 1.0)
                    r = np.arccos(cos_r)
                    cos_rb = np.dot(p, b)
                    cos_rb = np.clip(cos_rb, -1.0, 1.0)
                    rb = np.arccos(cos_rb)
                    if abs(r - rb) < 1e-6:
                        if r > max_circumradius:
                            max_circumradius = r
                            worst_simplex = simplex
                            worst_circumcenter = p
                        break
            
            if worst_simplex is None:
                break
            
            # Move the vertex opposite to the worst facet slightly toward the circumcenter
            # Find which vertex of the worst simplex is the "farthest" from the circumcenter
            # and move it
            simplex_pts = pts[worst_simplex]
            dists = np.linalg.norm(simplex_pts - worst_circumcenter, axis=1)
            farthest_idx = np.argmax(dists)
            
            # Move that point slightly toward the circumcenter
            point_idx = worst_simplex[farthest_idx]
            move_direction = worst_circumcenter - pts[point_idx]
            move_norm = np.linalg.norm(move_direction)
            if move_norm < 1e-12:
                break
            move_direction = move_direction / move_norm
            
            # Try a small step
            step_size = 0.01
            new_pts = pts.copy()
            new_pts[point_idx] = pts[point_idx] + step_size * move_direction
            new_pts[point_idx] = new_pts[point_idx] / np.linalg.norm(new_pts[point_idx])
            
            new_score = _covering_radius(new_pts)
            if new_score < best_score:
                best_score = new_score
                best_pts = new_pts
            else:
                # Try smaller step
                step_size = 0.001
                new_pts = pts.copy()
                new_pts[point_idx] = pts[point_idx] + step_size * move_direction
                new_pts[point_idx] = new_pts[point_idx] / np.linalg.norm(new_pts[point_idx])
                new_score = _covering_radius(new_pts)
                if new_score < best_score:
                    best_score = new_score
                    best_pts = new_pts
                else:
                    break
    
    return best_pts
# EVOLVE-BLOCK-END
