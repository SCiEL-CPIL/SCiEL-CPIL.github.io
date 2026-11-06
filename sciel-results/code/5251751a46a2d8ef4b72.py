"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return pts / norms

    def compute_covering_radius(pts):
        """
        Compute the exact covering radius via convex hull facets.
        """
        try:
            pts = normalize(pts)
            if np.any(np.linalg.norm(pts, axis=1) < 1e-10):
                return 180.0
            
            # Check for hemisphere degeneracy: origin not in convex hull
            # If the convex hull does not contain the origin, points are in a hemisphere
            # We can check by trying to find a separating hyperplane or just trust the hull
            # But a simpler check: if all dot products with some vector are positive, it's in a hemisphere
            # A robust check is to see if the minimum distance from origin to hull is > 0
            # However, ConvexHull in 3D for points on the sphere: if origin is inside, 
            # the hull is a polyhedron containing origin.
            # Let's just compute the hull and check facets.
            
            hull = ConvexHull(pts)
            
            max_r = 0.0
            for face in hull.simplices:
                p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
                
                # Compute circumcenter of spherical triangle
                # The circumcenter is proportional to the cross product of (p1-p0) x (p2-p0) ? No.
                # The circumcenter c satisfies c.p0 = c.p1 = c.p2 = r
                # c = w0*p0 + w1*p1 + w2*p2
                # w is solution to G*w = [1,1,1] where G[i,j] = pi.pj
                p_face = np.array([p0, p1, p2])
                G = p_face @ p_face.T
                rhs = np.array([1.0, 1.0, 1.0])
                
                try:
                    w = np.linalg.solve(G, rhs)
                except np.linalg.LinAlgError:
                    # Degenerate triangle (collinear on great circle)
                    # Circumcenter is undefined or at infinity, but on sphere it's ambiguous.
                    # Usually this means the triangle is very large (near 180 deg)
                    # Let's approximate by moving towards the antipode of the midpoint?
                    # Or just skip and let other facets handle it?
                    # Actually, if G is singular, the points are linearly dependent.
                    # On the sphere, this means they lie on a great circle.
                    # The "circumcenter" is not unique. The covering radius for this facet
                    # is determined by the largest gap.
                    # For now, skip or estimate.
                    continue
                    
                c = np.sum(w[:, None] * p_face, axis=0)
                c_norm = np.linalg.norm(c)
                if c_norm < 1e-10:
                    continue
                    
                c /= c_norm
                
                # Check if c is inside the spherical triangle
                # The circumcenter is inside if all w_i > 0
                # If any w_i < 0, the circumcenter is outside, and the max distance
                # is to one of the vertices.
                if np.min(w) < 0:
                    # Max distance is to the farthest vertex
                    # Since it's a triangle on the sphere, the farthest point from c (which is outside)
                    # would be one of the vertices.
                    # Actually, if c is outside, the function d(y) = dist(y, nearest pt) 
                    # on this facet is maximized at a vertex.
                    # So we check vertices.
                    d0 = np.arccos(np.clip(p0 @ p0, -1, 1)) # 0
                    # Wait, the distance from a point y on the facet to the nearest of p0,p1,p2.
                    # If the circumcenter is outside, the "Voronoi" cell doesn't contain the circumcenter.
                    # The maximum of min(dist(y,p0), dist(y,p1), dist(y,p2)) over the triangle
                    # occurs at a vertex?
                    # Let's check the vertices.
                    # dist(p0, p0)=0, dist(p0, p1)=d01, dist(p0, p2)=d02. Min is 0.
                    # So vertices are local minima of the distance function.
                    # The maximum must be on the edges or at the circumcenter (if inside).
                    # If circumcenter is outside, the maximum is on the boundary (edges).
                    # The midpoint of the longest edge is a candidate.
                    d01 = np.arccos(np.clip(p0 @ p1, -1, 1))
                    d02 = np.arccos(np.clip(p0 @ p2, -1, 1))
                    d12 = np.arccos(np.clip(p1 @ p2, -1, 1))
                    
                    # Check midpoints of edges
                    # Midpoint of p0,p1
                    m01 = (p0 + p1) / np.linalg.norm(p0 + p1) if np.linalg.norm(p0+p1)>1e-10 else p0
                    # Distance from m01 to nearest of p0,p1,p2
                    # dist(m01, p0) = d01/2
                    # dist(m01, p1) = d01/2
                    # dist(m01, p2) = ?
                    d_m01_p2 = np.arccos(np.clip(m01 @ p2, -1, 1))
                    min_d_m01 = min(d01/2, d_m01_p2)
                    
                    m02 = (p0 + p2) / np.linalg.norm(p0 + p2) if np.linalg.norm(p0+p2)>1e-10 else p0
                    d_m02_p1 = np.arccos(np.clip(m02 @ p1, -1, 1))
                    min_d_m02 = min(d02/2, d_m02_p1)
                    
                    m12 = (p1 + p2) / np.linalg.norm(p1 + p2) if np.linalg.norm(p1+p2)>1e-10 else p1
                    d_m12_p0 = np.arccos(np.clip(m12 @ p0, -1, 1))
                    min_d_m12 = min(d12/2, d_m12_p0)
                    
                    max_r_facet = max(min_d_m01, min_d_m02, min_d_m12)
                else:
                    # Circumcenter is inside
                    cos_d = np.clip(c @ p0, -1.0, 1.0)
                    d = np.arccos(cos_d)
                    max_r_facet = d
                    
                if max_r_facet > max_r:
                    max_r = max_r_facet
                    
            return np.degrees(max_r)
            
        except Exception:
            return 180.0

    def lloyd_polish(pts, max_iters=50, step_size=0.1):
        """
        Lloyd's algorithm for sphere covering.
        """
        pts = normalize(pts)
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            try:
                hull = ConvexHull(pts)
            except:
                break
                
            sum_centers = np.zeros_like(pts)
            counts = np.zeros(N)
            
            for face in hull.simplices:
                p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
                p_face = np.array([p0, p1, p2])
                G = p_face @ p_face.T
                rhs = np.array([1.0, 1.0, 1.0])
                try:
                    w = np.linalg.solve(G, rhs)
                    # Only use circumcenter if it's inside the triangle (w_i > 0)
                    if np.min(w) < 0:
                        # If outside, use the centroid or skip?
                        # For Lloyd, we usually move towards the centroid of the Voronoi cell.
                        # If the circumcenter is outside, the Voronoi vertex is not there.
                        # The "center" of the cell is tricky.
                        # Let's just use the average of the vertices for a stable update,
                        # or skip this facet's contribution to the specific vertex if it's not the closest?
                        # Actually, if the circumcenter is outside, the closest point on the sphere
                        # to the Voronoi cell center is one of the vertices.
                        # For simplicity, let's just add the vertices themselves?
                        # Or better: only update if circumcenter is inside.
                        # If outside, the point is likely "stuck" or the triangle is obtuse.
                        # Let's just use the centroid of the triangle as a fallback target.
                        c = (p0 + p1 + p2) / 3.0
                        c = c / np.linalg.norm(c)
                    else:
                        c = np.sum(w[:, None] * p_face, axis=0)
                        c_norm = np.linalg.norm(c)
                        if c_norm > 1e-10:
                            c /= c_norm
                        else:
                            continue
                            
                    for v_idx in face:
                        sum_centers[v_idx] += c
                        counts[v_idx] += 1
                except:
                    pass
            
            # Update points
            moved = False
            for i in range(N):
                if counts[i] > 0:
                    target = sum_centers[i] / counts[i]
                    target_norm = np.linalg.norm(target)
                    if target_norm > 1e-10:
                        target /= target_norm
                        new_pt = pts[i] + step_size * (target - pts[i])
                        new_pt_norm = np.linalg.norm(new_pt)
                        if new_pt_norm > 1e-10:
                            pts[i] = new_pt / new_pt_norm
                            moved = True
            
            if not moved:
                break
                
        return pts

    def covering_energy(x, beta=100.0):
        """
        Soft-max approximation of the covering radius.
        """
        pts = x.reshape(-1, 3)
        pts = normalize(pts)
        
        if not hasattr(covering_energy, 'test_pts'):
            # Generate Fibonacci sphere points
            n_test = 10000
            i = np.arange(n_test) + 0.5
            z = 1.0 - 2.0 * i / n_test
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            covering_energy.test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            
        test_pts = covering_energy.test_pts
        
        dots = test_pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        
        min_dists = np.min(dists, axis=1)
        
        max_d = np.max(min_dists)
        if max_d == 0:
            return 0.0
        exp_dists = np.exp(beta * (min_dists - max_d))
        soft_max = max_d + (1.0 / beta) * np.log(np.sum(exp_dists))
        
        return soft_max

    def covering_energy_grad(x, beta=100.0):
        """
        Gradient of the soft-max covering energy.
        """
        pts = x.reshape(-1, 3)
        pts = normalize(pts)
        
        test_pts = covering_energy.test_pts
        n_test = len(test_pts)
        
        dots = test_pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        
        min_idx = np.argmin(dists, axis=1)
        min_dists = np.take_along_axis(dists, min_idx[:, None], axis=1).flatten()
        
        max_d = np.max(min_dists)
        if max_d == 0:
            return np.zeros_like(x)
            
        exp_dists = np.exp(beta * (min_dists - max_d))
        weights = exp_dists / np.sum(exp_dists)
        
        grad_pts = np.zeros_like(pts)
        
        for i in range(n_test):
            j = min_idx[i]
            w = weights[i]
            if w == 0:
                continue
            
            u = dots[i, j]
            denom = np.sqrt(1.0 - u * u + 1e-12)
            grad_u = -1.0 / denom
            grad_pts[j] += w * beta * grad_u * test_pts[i]
            
        v = x.reshape(-1, 3)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        
        grad_v = np.zeros_like(v)
        for k in range(N):
            n_k = norms[k, 0]
            v_k = v[k]
            dot_vg = np.dot(v_k, grad_pts[k])
            grad_v[k] = grad_pts[k] / n_k - v_k * (dot_vg / (n_k ** 3))
            
        return grad_v.flatten()

    # Strategy: Multi-start with different initializations
    # 1. Fibonacci Sphere
    # 2. Random Points (multiple tries)
    
    best_cr = 180.0
    best_pts = None
    
    # Initial Configuration 1: Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimization 1
    if time.time() < deadline - 15:
        try:
            res = minimize(
                covering_energy, 
                init_pts.flatten(), 
                jac=covering_energy_grad, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            
            if res.fun < np.inf:
                pts_opt = res.x.reshape(-1, 3)
                pts_opt = normalize(pts_opt)
                cr_opt = compute_covering_radius(pts_opt)
                
                if cr_opt < best_cr:
                    best_cr = cr_opt
                    best_pts = pts_opt
                    
                # Lloyd Polish
                if time.time() < deadline - 10:
                    pts_lloyd = lloyd_polish(pts_opt, max_iters=50, step_size=0.1)
                    cr_lloyd = compute_covering_radius(pts_lloyd)
                    if cr_lloyd < best_cr:
                        best_cr = cr_lloyd
                        best_pts = pts_lloyd
        except Exception:
            pass

    # Initial Configuration 2: Random Points (Seed 1)
    if time.time() < deadline - 15:
        np.random.seed(42)
        init_pts2 = np.random.randn(N, 3)
        init_pts2 = normalize(init_pts2)
        
        try:
            res = minimize(
                covering_energy, 
                init_pts2.flatten(), 
                jac=covering_energy_grad, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            
            if res.fun < np.inf:
                pts_opt = res.x.reshape(-1, 3)
                pts_opt = normalize(pts_opt)
                cr_opt = compute_covering_radius(pts_opt)
                
                if cr_opt < best_cr:
                    best_cr = cr_opt
                    best_pts = pts_opt
                    
                if time.time() < deadline - 10:
                    pts_lloyd = lloyd_polish(pts_opt, max_iters=50, step_size=0.1)
                    cr_lloyd = compute_covering_radius(pts_lloyd)
                    if cr_lloyd < best_cr:
                        best_cr = cr_lloyd
                        best_pts = pts_lloyd
        except Exception:
            pass

    # Initial Configuration 3: Random Points (Seed 2)
    if time.time() < deadline - 15:
        np.random.seed(123)
        init_pts3 = np.random.randn(N, 3)
        init_pts3 = normalize(init_pts3)
        
        try:
            res = minimize(
                covering_energy, 
                init_pts3.flatten(), 
                jac=covering_energy_grad, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            
            if res.fun < np.inf:
                pts_opt = res.x.reshape(-1, 3)
                pts_opt = normalize(pts_opt)
                cr_opt = compute_covering_radius(pts_opt)
                
                if cr_opt < best_cr:
                    best_cr = cr_opt
                    best_pts = pts_opt
                    
                if time.time() < deadline - 10:
                    pts_lloyd = lloyd_polish(pts_opt, max_iters=50, step_size=0.1)
                    cr_lloyd = compute_covering_radius(pts_lloyd)
                    if cr_lloyd < best_cr:
                        best_cr = cr_lloyd
                        best_pts = pts_lloyd
        except Exception:
            pass

    # Final Polish on the best found
    if best_pts is not None and time.time() < deadline - 5:
        # Try a few different step sizes for Lloyd
        for step in [0.2, 0.05, 0.01]:
            if time.time() > deadline - 2:
                break
            pts_final = lloyd_polish(best_pts, max_iters=30, step_size=step)
            cr_final = compute_covering_radius(pts_final)
            if cr_final < best_cr:
                best_cr = cr_final
                best_pts = pts_final

    # If no valid points found (shouldn't happen), return Fibonacci
    if best_pts is None:
        return init_pts

    return normalize(best_pts)
# EVOLVE-BLOCK-END
