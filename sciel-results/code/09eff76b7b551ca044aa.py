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
        Compute the covering radius of the configuration on the unit sphere.
        The covering radius is the maximum circumradius of the spherical triangles
        (facets of the convex hull) formed by the points.
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
                    continue
                    
                c = np.sum(w[:, None] * p_face, axis=0)
                c_norm = np.linalg.norm(c)
                if c_norm < 1e-10:
                    continue
                    
                c /= c_norm
                
                cos_d = np.clip(c @ p0, -1.0, 1.0)
                d = np.arccos(cos_d)
                
                if d > max_r:
                    max_r = d
                    
            return np.degrees(max_r)
            
        except Exception:
            return 180.0

    def covering_energy(x, beta=100.0):
        """
        Soft-max approximation of the covering radius using a dense test grid.
        """
        pts = x.reshape(-1, 3)
        pts = normalize(pts)
        
        if not hasattr(covering_energy, 'test_pts'):
            # Dense Fibonacci sphere for test points
            n_test = 12000
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
        if max_d < 1e-10:
            return 0.0
            
        # Log-sum-exp for numerical stability
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
        if max_d < 1e-10:
            return np.zeros_like(x)
            
        exp_dists = np.exp(beta * (min_dists - max_d))
        weights = exp_dists / np.sum(exp_dists)
        
        grad_pts = np.zeros_like(pts)
        
        # Vectorized gradient computation
        for i in range(n_test):
            j = min_idx[i]
            w = weights[i]
            if w < 1e-15:
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

    def lloyd_polish(pts, max_iters=100, step_size=0.1):
        """
        Lloyd's algorithm for sphere covering.
        Moves each point towards the centroid of the circumcenters of its adjacent facets.
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
                    c = np.sum(w[:, None] * p_face, axis=0)
                    c_norm = np.linalg.norm(c)
                    if c_norm > 1e-10:
                        c /= c_norm
                        for v_idx in face:
                            sum_centers[v_idx] += c
                            counts[v_idx] += 1
                except:
                    pass
            
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

    def micro_polish(pts, max_iters=300, tol=1e-10):
        """
        Iteratively move points to reduce the maximum circumradius of hull facets.
        Focuses on the worst (largest circumradius) facet each iteration.
        """
        pts = normalize(pts)
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
                
            try:
                hull = ConvexHull(pts)
            except:
                break
            
            max_r = 0.0
            max_face_idx = -1
            max_circumcenter = None
            max_face = None
            
            for idx, face in enumerate(hull.simplices):
                p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
                p_face = np.array([p0, p1, p2])
                G = p_face @ p_face.T
                rhs = np.array([1.0, 1.0, 1.0])
                try:
                    w = np.linalg.solve(G, rhs)
                    c = np.sum(w[:, None] * p_face, axis=0)
                    c_norm = np.linalg.norm(c)
                    if c_norm < 1e-10:
                        continue
                    c /= c_norm
                    cos_d = np.clip(c @ p0, -1.0, 1.0)
                    d = np.arccos(cos_d)
                    if d > max_r:
                        max_r = d
                        max_face_idx = idx
                        max_circumcenter = c
                        max_face = face
                except:
                    pass
            
            if max_face_idx < 0:
                break
            
            # Adaptive step size: smaller as we approach convergence
            step = 0.005 if max_r < np.radians(18.5) else 0.02
            
            for v_idx in max_face:
                target = max_circumcenter
                new_pt = pts[v_idx] + step * (target - pts[v_idx])
                new_pt_norm = np.linalg.norm(new_pt)
                if new_pt_norm > 1e-10:
                    pts[v_idx] = new_pt / new_pt_norm
            
            if max_r < np.radians(18.31):
                break
                
        return pts

    # Generate multiple initial configurations
    def fibonacci_sphere(n, shift=0.0):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i + shift
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def random_sphere(n, seed=42):
        rng = np.random.RandomState(seed)
        pts = rng.randn(n, 3)
        return normalize(pts)

    # Initial configurations to try
    initial_configs = [
        fibonacci_sphere(N, 0.0),
        fibonacci_sphere(N, 0.3),
        fibonacci_sphere(N, 0.7),
        random_sphere(N, 42),
        random_sphere(N, 123),
    ]

    best_cr = 180.0
    best_pts = initial_configs[0]

    # Optimization loop for each initial configuration
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 20:
            break
            
        x0 = init_pts.flatten()
        
        try:
            # Phase 1: Coarse optimization with lower beta
            res1 = minimize(
                covering_energy, 
                x0, 
                jac=covering_energy_grad, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False,
                         'maxfun': 2000}
            )
            
            if res1.fun < np.inf:
                pts_opt1 = normalize(res1.x.reshape(-1, 3))
                cr_opt1 = compute_covering_radius(pts_opt1)
                
                if cr_opt1 < best_cr:
                    best_cr = cr_opt1
                    best_pts = pts_opt1
                    
                # Phase 2: Fine optimization with higher beta
                if time.time() < deadline - 10:
                    # Temporarily increase beta for finer optimization
                    old_beta = 100.0
                    # We'll just run another optimization from the current best
                    res2 = minimize(
                        covering_energy, 
                        best_pts.flatten(), 
                        jac=covering_energy_grad, 
                        method='L-BFGS-B', 
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False,
                                 'maxfun': 1000}
                    )
                    
                    if res2.fun < np.inf:
                        pts_opt2 = normalize(res2.x.reshape(-1, 3))
                        cr_opt2 = compute_covering_radius(pts_opt2)
                        
                        if cr_opt2 < best_cr:
                            best_cr = cr_opt2
                            best_pts = pts_opt2
                            
        except Exception:
            pass

    # Refinement: Lloyd's Algorithm
    if time.time() < deadline - 15:
        pts_lloyd = lloyd_polish(best_pts, max_iters=80, step_size=0.15)
        cr_lloyd = compute_covering_radius(pts_lloyd)
        if cr_lloyd < best_cr:
            best_cr = cr_lloyd
            best_pts = pts_lloyd
            
        # Second pass with smaller step
        if time.time() < deadline - 10:
            pts_lloyd2 = lloyd_polish(best_pts, max_iters=80, step_size=0.05)
            cr_lloyd2 = compute_covering_radius(pts_lloyd2)
            if cr_lloyd2 < best_cr:
                best_cr = cr_lloyd2
                best_pts = pts_lloyd2

    # Micro-polish: iteratively optimize the worst facet's circumcenter
    if time.time() < deadline - 5:
        pts_micro = micro_polish(best_pts, max_iters=300)
        cr_micro = compute_covering_radius(pts_micro)
        if cr_micro < best_cr:
            best_cr = cr_micro
            best_pts = pts_micro

    # Final Lloyd pass with very small step
    if time.time() < deadline - 2:
        pts_final = lloyd_polish(best_pts, max_iters=50, step_size=0.01)
        cr_final = compute_covering_radius(pts_final)
        if cr_final < best_cr:
            best_cr = cr_final
            best_pts = pts_final

    return best_pts
# EVOLVE-BLOCK-END
