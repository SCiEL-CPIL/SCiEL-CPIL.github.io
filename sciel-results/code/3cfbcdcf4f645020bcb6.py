import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# Pre-compute a dense grid for the smooth objective
def _make_fib_grid(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

GRID_COARSE = _make_fib_grid(2000)
GRID_FINE = _make_fib_grid(8000)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius_exact(pts):
        """Compute covering radius exactly as max circumradius over convex hull faces."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]
            
            # Compute circumcenter for each face
            # For a triangle on the unit sphere, the circumcenter is the normalized sum of the vertices
            # if the triangle is acute. If obtuse, the circumcenter lies outside the triangle
            # and the "covering" point is the midpoint of the longest edge.
            
            # Vector from a to b, b to c, c to a
            ab = b - a
            bc = c - b
            ca = a - c
            
            # Side lengths squared
            d_ab2 = np.sum(ab * ab, axis=1)
            d_bc2 = np.sum(bc * bc, axis=1)
            d_ca2 = np.sum(ca * ca, axis=1)
            
            # Check for degenerate triangles
            valid = (d_ab2 > 1e-10) & (d_bc2 > 1e-10) & (d_ca2 > 1e-10)
            
            radii = np.zeros(len(faces))
            
            if np.any(valid):
                a_v = a[valid]
                b_v = b[valid]
                c_v = c[valid]
                
                # Circumcenter of triangle abc on the sphere
                # The circumcenter is proportional to a + b + c only for equilateral? No.
                # The circumcenter O satisfies |O-a|^2 = |O-b|^2 = |O-c|^2 = R^2
                # |O|^2 - 2 O.a + |a|^2 = |O|^2 - 2 O.b + |b|^2
                # 2 O.(b-a) = |b|^2 - |a|^2 = 0 (since on unit sphere)
                # So O is orthogonal to b-a and c-a.
                # Thus O is parallel to cross(b-a, c-a).
                
                cross = np.cross(b_v - a_v, c_v - a_v)
                norms = np.linalg.norm(cross, axis=1, keepdims=True)
                valid_cross = norms > 1e-10
                cross_normed = cross / norms
                
                # Determine the sign: O should be on the same side as the triangle interior
                # The triangle interior is towards the centroid (a+b+c)/3
                # So dot(O, centroid) should be positive
                centroids = (a_v + b_v + c_v) / 3.0
                dots = np.sum(cross_normed * centroids, axis=1)
                signs = np.sign(dots)
                signs[signs == 0] = 1.0
                cross_normed = cross_normed * signs[:, np.newaxis]
                
                # Check if the circumcenter is inside the triangle
                # The circumcenter is inside if the triangle is acute.
                # A triangle on the sphere is acute if all angles < pi/2.
                # Angle at a: dot(b-a, c-a) > 0 ? No, that's the angle between vectors.
                # The angle at a is arccos( (b-a).c / (|b-a||c|) ) ... wait, tangent vectors.
                # Simpler: The circumcenter is inside the spherical triangle if and only if
                # the circumradius is less than pi/2.
                
                # Let's compute the angular distance from O to a, b, c
                cos_r = np.sum(cross_normed * a_v, axis=1)
                cos_r = np.clip(cos_r, -1.0, 1.0)
                r = np.arccos(cos_r)
                
                # If r > pi/2, the circumcenter is on the "wrong" side or the triangle is obtuse
                # In that case, the covering point is the midpoint of the longest edge
                obtuse = r > np.pi / 2
                if np.any(obtuse):
                    # Find longest edge
                    d_ab2_v = d_ab2[valid]
                    d_bc2_v = d_bc2[valid]
                    d_ca2_v = d_ca2[valid]
                    
                    max_d2 = np.max(np.stack([d_ab2_v, d_bc2_v, d_ca2_v]), axis=0)
                    # Find which edge is longest
                    # 0: ab, 1: bc, 2: ca
                    edge_idx = np.argmax(np.stack([d_ab2_v, d_bc2_v, d_ca2_v]), axis=0)
                    
                    # For each obtuse triangle, find the midpoint of the longest edge
                    midpoints = np.zeros((np.sum(obtuse), 3))
                    for i in range(np.sum(obtuse)):
                        idx = np.where(obtuse)[0][i]
                        e = edge_idx[idx]
                        if e == 0:
                            mid = (a_v[idx] + b_v[idx]) / 2.0
                        elif e == 1:
                            mid = (b_v[idx] + c_v[idx]) / 2.0
                        else:
                            mid = (c_v[idx] + a_v[idx]) / 2.0
                        mid = mid / np.linalg.norm(mid)
                        midpoints[i] = mid
                    
                    # Distance from midpoint to nearest vertex
                    # It's half the length of the longest edge
                    # Angular distance = arccos(1 - max_d2/2)
                    cos_r_obtuse = 1.0 - max_d2[obtuse] / 2.0
                    cos_r_obtuse = np.clip(cos_r_obtuse, -1.0, 1.0)
                    r[obtuse] = np.arccos(cos_r_obtuse)
                
                radii[valid] = r
            
            if np.any(~valid):
                # Degenerate triangles, use max edge distance
                a_d = a[~valid]
                b_d = b[~valid]
                c_d = c[~valid]
                d_ab2_d = d_ab2[~valid]
                d_bc2_d = d_bc2[~valid]
                d_ca2_d = d_ca2[~valid]
                max_d2_d = np.max(np.stack([d_ab2_d, d_bc2_d, d_ca2_d]), axis=0)
                cos_r_d = 1.0 - max_d2_d / 2.0
                cos_r_d = np.clip(cos_r_d, -1.0, 1.0)
                radii[~valid] = np.arccos(cos_r_d)
            
            return float(np.max(radii))
        except Exception:
            # Fallback: dense grid-based estimate
            dot = np.dot(GRID_FINE, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    def repulsion_energy_grad(pts, exponent=2.0):
        """Compute repulsion energy and gradient for uniform distribution."""
        dot = np.dot(pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        
        w = 1.0 / (dist_sq ** exponent)
        np.fill_diagonal(w, 0.0)
        
        w_grad = exponent * (dist_sq ** (-exponent/2 - 1))
        np.fill_diagonal(w_grad, 0.0)
        
        grad = w_grad @ pts
        
        return float(np.sum(w)), grad

    def gradient_descent_repulsion(pts, exponent, alpha_init, decay, max_iter):
        """Optimize points by gradient descent on repulsion energy."""
        alpha = alpha_init
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            energy, grad = repulsion_energy_grad(pts, exponent)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            
            pts = pts + alpha * grad
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            
            alpha *= decay
        
        return pts

    def grid_objective_and_grad(x_flat, grid_pts):
        """
        Smooth objective: soft-max of angular distances from grid points to nearest configuration point.
        Uses a vectorized, differentiable approximation.
        x_flat: flattened (N,3) array of points.
        grid_pts: precomputed grid of sphere points.
        Returns (value, gradient).
        """
        pts = x_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        pts_norm = pts / norms

        M = len(grid_pts)

        # Compute angular distances: M x N
        dot = np.dot(grid_pts, pts_norm.T)  # M x N
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)  # M x N

        # For each grid point, find the min distance to any configuration point
        # Use soft-min for differentiability
        T_min = 50.0
        exp_neg_T_min_d = np.exp(-T_min * ang_dists) # M x N
        sum_exp_neg = np.sum(exp_neg_T_min_d, axis=1) # M
        soft_min_dists = -1.0 / T_min * np.log(sum_exp_neg) # M
        
        # Weights for soft-min: w_jk = exp(-T_min * d_jk) / sum_k exp(-T_min * d_jk)
        w_jk = exp_neg_T_min_d / sum_exp_neg[:, np.newaxis] # M x N
        
        # Gradient of soft_min_dists[j] w.r.t pts_norm[k]
        # d(s_j)/d(pts_norm[k]) = sum_k' w_jk' * (d(d_jk')/d(pts_norm[k']))
        # d(d_jk)/d(pts_norm[k]) = - (grid_pts[j] - pts_norm[k]) / ||grid_pts[j] - pts_norm[k]||
        # Note: d_jk is arccos(dot). d/dx arccos(u) = -1/sqrt(1-u^2) * du/dx.
        # u = dot = grid_pts[j] . pts_norm[k]
        # du/d(pts_norm[k]) = grid_pts[j]
        # So d(d_jk)/d(pts_norm[k]) = - grid_pts[j] / sqrt(1 - dot^2)
        
        sqrt_term = np.sqrt(np.maximum(1.0 - dot * dot, 1e-10)) # M x N
        grad_d = -grid_pts[:, np.newaxis, :] / sqrt_term[:, :, np.newaxis] # M x N x 3
        
        # Now we need sum_k' w_jk' * grad_d_jk'
        # This is a weighted sum over k' for each j
        # w_jk is M x N, grad_d is M x N x 3
        # We want result of shape N x 3
        # grad_s = sum_j [ sum_k w_jk * grad_d_jk ]
        # Wait, the gradient of the objective with respect to pts_norm[k] is:
        # grad_obj[k] = sum_j (d(soft_max)/d(soft_min_dists[j]) * d(soft_min_dists[j])/d(pts_norm[k]))
        
        # First, compute soft-max of soft_min_dists
        T_max = 10.0
        max_s = np.max(soft_min_dists)
        exp_T_max_s = np.exp(T_max * (soft_min_dists - max_s))
        sum_exp_T_max = np.sum(exp_T_max_s)
        soft_max_val = max_s + np.log(sum_exp_T_max) / T_max
        
        # Weights for soft-max: v_j = exp(T_max * (s_j - max_s)) / sum
        v_j = exp_T_max_s / sum_exp_T_max # M
        
        # Now, d(soft_min_dists[j])/d(pts_norm[k]) = sum_k' w_jk' * grad_d_jk'_k
        # Let's compute G[j, k] = sum_k' w_jk' * grad_d_jk'_k
        # This is equivalent to (w_jk @ grad_d_jk) but grad_d is 3D.
        # G[j, :, :] = sum_k' w_jk' * grad_d[j, k', :]
        # We can do this by: G = np.einsum('jk,jkx->jx', w_jk, grad_d)
        G = np.einsum('jk,jkx->jx', w_jk, grad_d) # M x 3
        
        # Now, grad_obj[k] = sum_j v_j * G[j, k] ... wait, G[j, k] is the gradient of s_j w.r.t pts_norm[k]?
        # No, G[j, :] is the gradient of s_j w.r.t the whole vector pts_norm? No.
        # G[j, k] is not quite right.
        # d(s_j)/d(pts_norm) is a vector of shape (N, 3).
        # d(s_j)/d(pts_norm[k]) = sum_k' w_jk' * grad_d_jk'_k
        # This means the component of the gradient vector corresponding to pts_norm[k] is sum_k' w_jk' * grad_d_jk'_k.
        # So G should be of shape M x N x 3? No.
        # Let's re-evaluate.
        # grad_d is M x N x 3. grad_d[j, k, :] is the gradient of d_jk w.r.t pts_norm[k].
        # d(s_j)/d(pts_norm[k]) = sum_k' w_jk' * grad_d[j, k', :]
        # So the full gradient of s_j w.r.t pts_norm is a matrix of shape N x 3, where the k-th row is sum_k' w_jk' * grad_d[j, k', :].
        # Let's call this Grad_s_j.
        # Grad_s_j = np.einsum('jk,jkx->kx', w_jk, grad_d) # N x 3
        # Then grad_obj = sum_j v_j * Grad_s_j
        # grad_obj = np.einsum('j,jkx->kx', v_j, Grad_s)
        
        Grad_s = np.einsum('jk,jkx->kx', w_jk, grad_d) # M x N x 3
        grad_obj = np.einsum('j,jkx->kx', v_j, Grad_s) # N x 3
        
        return float(soft_max_val), grad_obj

    def local_search_facet(pts):
        """Geometric local search based on convex hull facets."""
        # Compute covering radius and find the worst facet
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]
            
            # Compute circumcenters and radii as in compute_covering_radius_exact
            ab = b - a
            bc = c - b
            ca = a - c
            
            d_ab2 = np.sum(ab * ab, axis=1)
            d_bc2 = np.sum(bc * bc, axis=1)
            d_ca2 = np.sum(ca * ca, axis=1)
            
            valid = (d_ab2 > 1e-10) & (d_bc2 > 1e-10) & (d_ca2 > 1e-10)
            
            radii = np.zeros(len(faces))
            
            if np.any(valid):
                a_v = a[valid]
                b_v = b[valid]
                c_v = c[valid]
                
                cross = np.cross(b_v - a_v, c_v - a_v)
                norms = np.linalg.norm(cross, axis=1, keepdims=True)
                valid_cross = norms > 1e-10
                cross_normed = cross / norms
                
                centroids = (a_v + b_v + c_v) / 3.0
                dots = np.sum(cross_normed * centroids, axis=1)
                signs = np.sign(dots)
                signs[signs == 0] = 1.0
                cross_normed = cross_normed * signs[:, np.newaxis]
                
                cos_r = np.sum(cross_normed * a_v, axis=1)
                cos_r = np.clip(cos_r, -1.0, 1.0)
                r = np.arccos(cos_r)
                
                obtuse = r > np.pi / 2
                if np.any(obtuse):
                    d_ab2_v = d_ab2[valid]
                    d_bc2_v = d_bc2[valid]
                    d_ca2_v = d_ca2[valid]
                    
                    max_d2 = np.max(np.stack([d_ab2_v, d_bc2_v, d_ca2_v]), axis=0)
                    edge_idx = np.argmax(np.stack([d_ab2_v, d_bc2_v, d_ca2_v]), axis=0)
                    
                    for i in range(np.sum(obtuse)):
                        idx = np.where(obtuse)[0][i]
                        e = edge_idx[idx]
                        if e == 0:
                            mid = (a_v[idx] + b_v[idx]) / 2.0
                        elif e == 1:
                            mid = (b_v[idx] + c_v[idx]) / 2.0
                        else:
                            mid = (c_v[idx] + a_v[idx]) / 2.0
                        mid = mid / np.linalg.norm(mid)
                        
                        cos_r_obtuse = 1.0 - max_d2[idx] / 2.0
                        cos_r_obtuse = np.clip(cos_r_obtuse, -1.0, 1.0)
                        r[idx] = np.arccos(cos_r_obtuse)
                
                radii[valid] = r
            
            if np.any(~valid):
                a_d = a[~valid]
                b_d = b[~valid]
                c_d = c[~valid]
                d_ab2_d = d_ab2[~valid]
                d_bc2_d = d_bc2[~valid]
                d_ca2_d = d_ca2[~valid]
                max_d2_d = np.max(np.stack([d_ab2_d, d_bc2_d, d_ca2_d]), axis=0)
                cos_r_d = 1.0 - max_d2_d / 2.0
                cos_r_d = np.clip(cos_r_d, -1.0, 1.0)
                radii[~valid] = np.arccos(cos_r_d)
            
            # Find the facet with the largest radius
            worst_idx = np.argmax(radii)
            worst_radius = radii[worst_idx]
            
            # Move the points of the worst facet slightly towards the circumcenter
            # to reduce the radius
            face = faces[worst_idx]
            p_a = pts[face[0]]
            p_b = pts[face[1]]
            p_c = pts[face[2]]
            
            # Compute circumcenter
            cross = np.cross(p_b - p_a, p_c - p_a)
            norm = np.linalg.norm(cross)
            if norm > 1e-10:
                cross = cross / norm
                centroid = (p_a + p_b + p_c) / 3.0
                if np.dot(cross, centroid) < 0:
                    cross = -cross
                
                # Move points towards circumcenter
                step = 0.01
                pts[face[0]] = p_a + step * (cross - p_a)
                pts[face[1]] = p_b + step * (cross - p_b)
                pts[face[2]] = p_c + step * (cross - p_c)
                
                # Normalize
                pts[face[0]] = pts[face[0]] / np.linalg.norm(pts[face[0]])
                pts[face[1]] = pts[face[1]] / np.linalg.norm(pts[face[1]])
                pts[face[2]] = pts[face[2]] / np.linalg.norm(pts[face[2]])
            
            return pts
        except Exception:
            return pts

    # Initial points using Fibonacci spiral
    points = fibonacci_spiral(N)
    
    # Phase 1: Repulsion-based optimization
    points = gradient_descent_repulsion(points, 2.0, 0.01, 0.999, 5000)
    
    # Phase 2: Grid-based gradient descent
    x_flat = points.flatten()
    for i in range(200):
        if time.time() > deadline - 10.0:
            break
        obj, grad = grid_objective_and_grad(x_flat, GRID_COARSE)
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        x_flat = x_flat + 0.01 * grad
        points = x_flat.reshape(N, 3)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
    
    # Phase 3: Local search on facets
    for i in range(100):
        if time.time() > deadline - 5.0:
            break
        points = local_search_facet(points)
    
    # Final projection
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    return points
# EVOLVE-BLOCK-END
