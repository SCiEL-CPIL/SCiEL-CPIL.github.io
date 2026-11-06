"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return pts / norms

    def compute_covering_radius_and_worst_indices(pts):
        """
        Computes the exact covering radius and identifies the facet responsible for it.
        Returns (radius_degrees, worst_face_indices, circumcenter).
        """
        try:
            hull = ConvexHull(pts)
            max_r_rad = 0.0
            worst_face = None
            worst_center = None
            
            for face in hull.simplices:
                p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
                
                # Compute circumcenter of the spherical triangle
                v1 = p0 - p1
                v2 = p0 - p2
                c_vec = np.cross(v1, v2)
                
                norm_c = np.linalg.norm(c_vec)
                if norm_c < 1e-10:
                    continue
                    
                c_vec /= norm_c
                
                # Determine correct hemisphere for circumcenter
                centroid = (p0 + p1 + p2) / 3.0
                centroid /= np.linalg.norm(centroid)
                
                if np.dot(c_vec, centroid) < 0:
                    c_vec = -c_vec
                    
                # Check if circumcenter is inside the spherical triangle
                n1 = np.cross(p1, p2)
                n2 = np.cross(p2, p0)
                n3 = np.cross(p0, p1)
                
                # Orient normals towards centroid
                if np.dot(centroid, n1) < 0: n1 = -n1
                if np.dot(centroid, n2) < 0: n2 = -n2
                if np.dot(centroid, n3) < 0: n3 = -n3
                
                if (np.dot(c_vec, n1) >= -1e-8 and 
                    np.dot(c_vec, n2) >= -1e-8 and 
                    np.dot(c_vec, n3) >= -1e-8):
                    
                    dist = np.arccos(np.clip(np.dot(c_vec, p0), -1.0, 1.0))
                    if dist > max_r_rad:
                        max_r_rad = dist
                        worst_face = face
                        worst_center = c_vec
                        
            return np.degrees(max_r_rad), worst_face, worst_center
        except Exception:
            return 180.0, None, None

    def lloyd_relax(pts, steps=20):
        n_grid = 10000
        i = np.arange(n_grid) + 0.5
        z = 1.0 - 2.0 * i / n_grid
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        grid_pts = normalize(grid_pts)
        
        current_pts = pts.copy()
        for _ in range(steps):
            dots = grid_pts @ current_pts.T
            nearest_idx = np.argmax(dots, axis=1)
            new_pts = np.zeros_like(current_pts)
            for k in range(N):
                mask = (nearest_idx == k)
                if np.any(mask):
                    new_pts[k] = np.sum(grid_pts[mask], axis=0)
                else:
                    new_pts[k] = current_pts[k]
            new_pts = normalize(new_pts)
            current_pts = new_pts
        return current_pts

    def refine_worst_triangle(pts, face_indices):
        """
        Refines the 3 vertices of the worst triangle to minimize its circumradius,
        keeping other points fixed.
        """
        if face_indices is None:
            return pts
        
        p0, p1, p2 = pts[face_indices[0]], pts[face_indices[1]], pts[face_indices[2]]
        
        # Initial guess: move vertices towards the circumcenter of the triangle
        # The circumcenter of the triangle on the sphere is the projection of the origin
        # onto the plane defined by p0, p1, p2? No, it's the point equidistant from p0, p1, p2 on the sphere.
        # For a triangle on the unit sphere, the circumcenter is the normalized vector:
        # c = (p0 + p1 + p2) is not generally the circumcenter.
        # The circumcenter is the intersection of the sphere with the line through the origin perpendicular to the plane of the triangle?
        # Actually, for a spherical triangle, the circumcenter is the pole of the plane containing the triangle vertices, IF the triangle is acute.
        # Let's just use the centroid as a starting point for the optimizer, or the normalized sum.
        
        start_vec = p0 + p1 + p2
        start_norm = np.linalg.norm(start_vec)
        if start_norm < 1e-6:
            start_vec = np.array([1.0, 0.0, 0.0])
            start_norm = 1.0
        start_vec /= start_norm
        
        # We want to optimize p0, p1, p2 on the sphere.
        # Parameterize by their current positions? Or just optimize the 3 points directly with constraints?
        # scipy.optimize.minimize doesn't handle sphere constraints well directly.
        # Let's use a local gradient descent on the sphere for these 3 points.
        
        current_p0, current_p1, current_p2 = p0.copy(), p1.copy(), p2.copy()
        
        def objective_and_grad(x):
            q0, q1, q2 = x[:3], x[3:6], x[6:9]
            # Normalize
            n0 = np.linalg.norm(q0)
            n1 = np.linalg.norm(q1)
            n2 = np.linalg.norm(q2)
            if n0 < 1e-6 or n1 < 1e-6 or n2 < 1e-6:
                return 1.0, np.zeros(9)
            q0 /= n0
            q1 /= n1
            q2 /= n2
            
            # Compute circumcenter of q0, q1, q2
            v1 = q0 - q1
            v2 = q0 - q2
            c_vec = np.cross(v1, v2)
            nc = np.linalg.norm(c_vec)
            if nc < 1e-10:
                return 1.0, np.zeros(9)
            c_vec /= nc
            
            # Check if circumcenter is inside the triangle
            # Use the same logic as before
            centroid = (q0 + q1 + q2) / 3.0
            centroid /= np.linalg.norm(centroid)
            if np.dot(c_vec, centroid) < 0:
                c_vec = -c_vec
                
            n1 = np.cross(q1, q2)
            n2 = np.cross(q2, q0)
            n3 = np.cross(q0, q1)
            
            if np.dot(centroid, n1) < 0: n1 = -n1
            if np.dot(centroid, n2) < 0: n2 = -n2
            if np.dot(centroid, n3) < 0: n3 = -n3
            
            if not (np.dot(c_vec, n1) >= -1e-8 and np.dot(c_vec, n2) >= -1e-8 and np.dot(c_vec, n3) >= -1e-8):
                # If circumcenter is outside, the covering radius for this triangle is determined by the longest edge
                # or we should just penalize this configuration heavily.
                # For a valid covering, the circumcenter should be inside.
                # If it's outside, the "worst point" is actually the vertex with the largest angle?
                # No, if circumcenter is outside, the triangle is obtuse. The farthest point on the sphere from the triangle's interior
                # is one of the vertices? No, the circumradius is still defined by the circumcenter if we consider the circumcircle.
                # But for covering, if the circumcenter is outside the triangle, the triangle doesn't cover the circumcenter.
                # The covering radius of the triangle is then the maximum distance to the vertices?
                # Actually, if the circumcenter is outside, the triangle is not "acute" enough to contain its circumcenter.
                # In that case, the covering radius is the distance to the farthest vertex.
                # Let's just return the distance to the circumcenter anyway, as a proxy for "badness".
                pass
            
            dist = np.arccos(np.clip(np.dot(c_vec, q0), -1.0, 1.0))
            
            # Gradient calculation is complex. Let's use numerical gradient or a simple projection method.
            # Given time constraints, let's use a simple gradient descent with numerical gradients for these 3 points.
            # Or better, use scipy minimize on the 9 parameters with a penalty for non-unit norm?
            # Let's stick to a simple iterative projection method:
            # Move each vertex towards the circumcenter.
            
            return dist, np.zeros(9) # Placeholder, we'll use a different method
        
        # Instead of complex gradients, let's use a simple "pull towards circumcenter" iteration.
        # This is effectively a local optimization.
        
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            
            # Compute circumcenter of current triangle
            v1 = current_p0 - current_p1
            v2 = current_p0 - current_p2
            c_vec = np.cross(v1, v2)
            nc = np.linalg.norm(c_vec)
            if nc < 1e-10:
                break
            c_vec /= nc
            
            centroid = (current_p0 + current_p1 + current_p2) / 3.0
            centroid /= np.linalg.norm(centroid)
            if np.dot(c_vec, centroid) < 0:
                c_vec = -c_vec
            
            # Move vertices towards c_vec
            step = 0.05
            new_p0 = current_p0 + step * (c_vec - current_p0)
            new_p1 = current_p1 + step * (c_vec - current_p1)
            new_p2 = current_p2 + step * (c_vec - current_p2)
            
            new_p0 /= np.linalg.norm(new_p0)
            new_p1 /= np.linalg.norm(new_p1)
            new_p2 /= np.linalg.norm(new_p2)
            
            current_p0, current_p1, current_p2 = new_p0, new_p1, new_p2
            
        pts[face_indices[0]] = current_p0
        pts[face_indices[1]] = current_p1
        pts[face_indices[2]] = current_p2
        
        return pts

    # Main Strategy
    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    fib_pts = normalize(fib_pts)
    
    best_pts = fib_pts.copy()
    best_cr, _, _ = compute_covering_radius_and_worst_indices(best_pts)
    
    # 2. Initial Lloyd relaxation to get a decent starting point
    lloyd_pts = lloyd_relax(best_pts, steps=10)
    lloyd_cr, _, _ = compute_covering_radius_and_worst_indices(lloyd_pts)
    if lloyd_cr < best_cr:
        best_cr = lloyd_cr
        best_pts = lloyd_pts

    # 3. Targeted Hull-based refinement
    for _ in range(50):
        if time.time() > deadline - 10:
            break
            
        cr, worst_face, worst_center = compute_covering_radius_and_worst_indices(best_pts)
        
        if worst_face is None:
            break
            
        # Refine the worst triangle
        refined_pts = refine_worst_triangle(best_pts, worst_face)
        
        # Check if the refinement improved the overall covering radius
        new_cr, _, _ = compute_covering_radius_and_worst_indices(refined_pts)
        
        if new_cr < best_cr:
            best_cr = new_cr
            best_pts = refined_pts
        else:
            # If it didn't improve, try a small perturbation or break?
            # To avoid getting stuck, we can try moving the vertices slightly in a random direction on the sphere.
            # Or just break if we've made no progress for a while.
            # For now, let's just break if no improvement to save time.
            # A more robust approach would be to try different step sizes.
            pass

    return best_pts
# EVOLVE-BLOCK-END
