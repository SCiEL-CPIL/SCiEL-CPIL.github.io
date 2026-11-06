"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return pts / norms

    def compute_covering_radius(pts):
        """
        Compute the exact covering radius based on the convex hull facets.
        """
        try:
            pts = normalize(pts)
            if np.any(np.linalg.norm(pts, axis=1) < 1e-10):
                return 180.0
            
            hull = ConvexHull(pts)
            max_r = 0.0
            
            for face in hull.simplices:
                p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
                p_face = np.array([p0, p1, p2])
                G = p_face @ p_face.T
                rhs = np.array([1.0, 1.0, 1.0])
                
                try:
                    w = np.linalg.solve(G, rhs)
                except np.linalg.LinAlgError:
                    # Degenerate
                    centroid = np.sum(p_face, axis=0)
                    centroid /= np.linalg.norm(centroid)
                    dists = np.arccos(np.clip(pts @ centroid, -1.0, 1.0))
                    min_dist = np.min(dists)
                    if min_dist > max_r:
                        max_r = min_dist
                    continue
                    
                c = np.sum(w[:, None] * p_face, axis=0)
                c_norm = np.linalg.norm(c)
                if c_norm < 1e-10:
                    continue
                    
                c /= c_norm
                
                # Check if circumcenter is inside the triangle
                # For a convex hull of points on a sphere, the circumcenter of a face 
                # is the center of the minimal cap covering the face vertices.
                # However, we must ensure it's the "active" constraint.
                # If the circumcenter is far from the face centroid, it might be outside.
                # A robust check: The distance from c to the plane of the triangle?
                # Actually, for the covering radius, we just need the max circumradius 
                # of the Delaunay triangles. The ConvexHull gives the Delaunay triangulation 
                # for points on a sphere.
                
                cos_d = np.clip(c @ p0, -1.0, 1.0)
                d = np.arccos(cos_d)
                
                if d > max_r:
                    max_r = d
                    
            return np.degrees(max_r)
            
        except Exception:
            return 180.0

    def facet_circumradii(pts):
        """
        Compute circumradius for each convex hull facet.
        Returns array of circumradii (in radians) and indices of points involved.
        """
        try:
            pts = normalize(pts)
            hull = ConvexHull(pts)
            radii = []
            for face in hull.simplices:
                p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
                p_face = np.array([p0, p1, p2])
                G = p_face @ p_face.T
                rhs = np.array([1.0, 1.0, 1.0])
                
                try:
                    w = np.linalg.solve(G, rhs)
                except np.linalg.LinAlgError:
                    radii.append(np.pi/2) # Fallback
                    continue
                    
                c = np.sum(w[:, None] * p_face, axis=0)
                c_norm = np.linalg.norm(c)
                if c_norm < 1e-10:
                    radii.append(np.pi/2)
                    continue
                    
                c /= c_norm
                cos_d = np.clip(c @ p0, -1.0, 1.0)
                d = np.arccos(cos_d)
                radii.append(d)
            return np.array(radii)
        except Exception:
            return np.array([np.pi/2] * 100) # Fallback

    def soft_max_energy(x, beta=100.0):
        """
        Energy: Soft-max of the circumradii of the convex hull facets.
        This directly targets the covering radius definition.
        """
        pts = x.reshape(-1, 3)
        pts = normalize(pts)
        
        radii = facet_circumradii(pts)
        if len(radii) == 0:
            return 1.0
            
        # To handle the non-differentiability of the hull structure,
        # we rely on the fact that locally, the hull structure is stable.
        # We use a soft-max to smooth the max operation.
        max_r = np.max(radii)
        if max_r == 0:
            return 0.0
        exp_r = np.exp(beta * (radii - max_r))
        soft_max = max_r + (1.0 / beta) * np.log(np.sum(exp_r))
        return soft_max

    def soft_max_grad(x, beta=100.0):
        """
        Gradient of the soft-max energy.
        We use finite differences for the hull structure changes? No, too slow.
        Instead, we approximate the gradient by assuming the hull structure is fixed 
        for a small step, and only the circumradii change.
        However, the circumradius depends on the points.
        
        dR/dp_i for a facet (i,j,k):
        R = arccos(c . p_i)
        c = (sum w_m p_m) / |sum w_m p_m|
        w solves G w = 1.
        
        This is complex. Let's use a simpler heuristic gradient:
        Push points in the direction that reduces the distance to the circumcenter of their facets.
        """
        pts = x.reshape(-1, 3)
        pts = normalize(pts)
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            return np.zeros_like(x)
            
        grad_pts = np.zeros_like(pts)
        
        radii = []
        circumcenters = []
        faces_list = []
        
        for face in hull.simplices:
            p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
            p_face = np.array([p0, p1, p2])
            G = p_face @ p_face.T
            rhs = np.array([1.0, 1.0, 1.0])
            
            try:
                w = np.linalg.solve(G, rhs)
            except np.linalg.LinAlgError:
                continue
                
            c = np.sum(w[:, None] * p_face, axis=0)
            c_norm = np.linalg.norm(c)
            if c_norm < 1e-10:
                continue
            c /= c_norm
            
            cos_d = np.clip(c @ p0, -1.0, 1.0)
            d = np.arccos(cos_d)
            
            radii.append(d)
            circumcenters.append(c)
            faces_list.append(face)
            
        if len(radii) == 0:
            return np.zeros_like(x)
            
        radii = np.array(radii)
        max_r = np.max(radii)
        exp_r = np.exp(beta * (radii - max_r))
        weights = exp_r / np.sum(exp_r)
        
        # Gradient accumulation
        # For each facet f with weight w_f:
        # dE/dp_i += w_f * (dE/dR_f) * (dR_f/dp_i)
        # dE/dR_f = 1 (since E approx max R)
        # dR/dp_i = d(arccos(c.p_i))/dp_i
        # This requires d c / d p_i as well.
        
        # Approximation:
        # Moving p_i towards the circumcenter c reduces the distance.
        # But c also moves.
        # A robust heuristic: 
        # Force points to move in the direction of the circumcenter of their worst facets.
        
        for idx, face in enumerate(faces_list):
            w_f = weights[idx]
            if w_f < 1e-6:
                continue
                
            c = circumcenters[idx]
            p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
            
            # Direction to move p0 to reduce distance to c
            # grad_d/dp0 = -c / sin(d)
            d0 = np.arccos(np.clip(c @ p0, -1.0, 1.0))
            sin_d0 = np.sin(d0)
            if sin_d0 > 1e-6:
                grad_pts[face[0]] += w_f * (-c / sin_d0)
            
            d1 = np.arccos(np.clip(c @ p1, -1.0, 1.0))
            sin_d1 = np.sin(d1)
            if sin_d1 > 1e-6:
                grad_pts[face[1]] += w_f * (-c / sin_d1)
                
            d2 = np.arccos(np.clip(c @ p2, -1.0, 1.0))
            sin_d2 = np.sin(d2)
            if sin_d2 > 1e-6:
                grad_pts[face[2]] += w_f * (-c / sin_d2)
                
        # Project gradient to tangent space
        # grad_tangent = grad - (grad . p) p
        for i in range(N):
            proj = np.dot(grad_pts[i], pts[i])
            grad_pts[i] -= proj * pts[i]
            
        # Backprop through normalization (approximate: assume norms are 1)
        return grad_pts.flatten()

    # Initial configurations
    # 1. Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = init_pts.copy()
    best_cr = compute_covering_radius(init_pts)
    
    # 2. Random restarts with optimization
    rng = np.random.default_rng(42)
    
    # Optimization settings
    # We use L-BFGS-B with the custom gradient.
    # Note: The gradient is approximate/heuristic, so we might need to be careful with step size.
    
    attempts = [
        init_pts,
        rng.normal(size=(N, 3)),
        rng.normal(size=(N, 3)),
        rng.normal(size=(N, 3))
    ]
    
    for init in attempts:
        if time.time() > deadline - 10:
            break
            
        pts_init = normalize(init)
        x0 = pts_init.flatten()
        
        try:
            res = minimize(
                soft_max_energy, 
                x0, 
                jac=soft_max_grad, 
                method='L-BFGS-B', 
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            
            if res.fun < np.inf:
                pts_opt = res.x.reshape(-1, 3)
                pts_opt = normalize(pts_opt)
                cr_opt = compute_covering_radius(pts_opt)
                
                if cr_opt < best_cr:
                    best_cr = cr_opt
                    best_pts = pts_opt
                    
        except Exception:
            pass

    # Final polish: Small random perturbations of the best solution
    if time.time() < deadline - 5:
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            perturbed = best_pts + 0.01 * rng.normal(size=(N, 3))
            perturbed = normalize(perturbed)
            x0 = perturbed.flatten()
            try:
                res = minimize(
                    soft_max_energy, 
                    x0, 
                    jac=soft_max_grad, 
                    method='L-BFGS-B', 
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
                if res.fun < np.inf:
                    pts_opt = res.x.reshape(-1, 3)
                    pts_opt = normalize(pts_opt)
                    cr_opt = compute_covering_radius(pts_opt)
                    if cr_opt < best_cr:
                        best_cr = cr_opt
                        best_pts = pts_opt
            except Exception:
                pass

    return best_pts
# EVOLVE-BLOCK-END
