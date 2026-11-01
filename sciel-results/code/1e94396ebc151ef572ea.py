import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

    def fibonacci_initial(phase=0.0):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + phase)
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def greedy_initial(seed_val):
        rng = np.random.RandomState(seed_val)
        pts = [rng.randn(3)]
        pts[0] = pts[0] / np.linalg.norm(pts[0])
        for _ in range(N - 1):
            candidates = rng.randn(10000, 3)
            candidates = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
            dots = candidates @ np.array(pts).T
            max_dots = np.max(dots, axis=1)
            idx = np.argmin(max_dots)
            pts.append(candidates[idx])
        return np.array(pts)

    def compute_hull_data(p):
        try:
            hull = ConvexHull(p)
            facets = hull.simplices
        except Exception:
            return None, None, None

        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None

        a = p[facets[:, 0]]
        b = p[facets[:, 1]]
        c = p[facets[:, 2]]

        v1 = a - b
        v2 = b - c
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n, axis=1)

        valid = norm_n > 1e-12
        circumcenters = np.zeros((n_facets, 3))
        radii = np.zeros(n_facets)

        if np.any(valid):
            cc = n[valid] / norm_n[valid, np.newaxis]
            # Ensure normal points outward
            dots = np.sum(cc * a[valid], axis=1)
            cc[dots < 0] *= -1
            circumcenters[valid] = cc
            cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
            radii[valid] = np.arccos(cos_r)

        return facets, circumcenters, radii

    def get_max_radius(pts):
        facets, circumcenters, radii = compute_hull_data(pts)
        if radii is None:
            return float('inf')
        return np.max(radii)

    def soft_max_objective(params, beta=200.0):
        X = params.reshape(N, 3)
        X = normalize(X)
        
        facets, circumcenters, radii = compute_hull_data(X)
        if facets is None:
            return 1e10

        r_max = np.max(radii)
        if r_max < 1e-15:
            return r_max
        shifted = beta * (radii - r_max)
        shifted = np.clip(shifted, -500, 500)
        log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + r_max
        return log_sum_exp

    def pressure_relax(pts, n_iters=500, alpha_start=0.05):
        """
        Pressure minimization: move vertices to equalize facet circumradii.
        This treats the hull as a soap film minimizing surface energy (related to radius).
        """
        pts = pts.copy()
        alpha = alpha_start
        
        for it in range(n_iters):
            if time.time() > deadline - 5.0:
                break
                
            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break
            
            r_mean = np.mean(radii)
            r_max = np.max(radii)
            
            # Compute forces: pull vertices towards circumcenters of large facets
            # Force on vertex i = sum over facets f containing i of w_f * (cc_f - p_i)
            # w_f is proportional to how much larger facet f is than average
            forces = np.zeros_like(pts)
            
            for k in range(len(facets)):
                rad = radii[k]
                if rad < 1e-10:
                    continue
                
                # Weight: emphasize facets with large radii
                # Using a smooth weight to avoid discontinuities
                w = np.exp(5.0 * (rad - r_mean))
                
                cc = circumcenters[k]
                for idx in facets[k]:
                    # Direction from vertex to circumcenter, projected to tangent
                    v = pts[idx]
                    dir_vec = cc - v
                    radial_comp = np.dot(dir_vec, v)
                    dir_vec = dir_vec - radial_comp * v
                    norm_dir = np.linalg.norm(dir_vec)
                    if norm_dir > 1e-10:
                        dir_vec /= norm_dir
                        forces[idx] += w * dir_vec
            
            # Normalize forces to prevent explosion
            norm_forces = np.linalg.norm(forces, axis=1, keepdims=True)
            norm_forces[norm_forces < 1e-12] = 1.0
            forces = forces / norm_forces
            
            # Line search
            improved = False
            step = alpha
            current_radius = get_max_radius(pts)
            
            for _ in range(5):
                new_pts = pts + step * forces
                new_pts = normalize(new_pts)
                new_radius = get_max_radius(new_pts)
                
                if new_radius < current_radius:
                    pts = new_pts
                    current_radius = new_radius
                    improved = True
                    break
                step *= 0.5
            
            if improved:
                alpha = min(alpha * 1.05, 0.2)
            else:
                alpha *= 0.8
            
            if alpha < 1e-6:
                break
                
        return pts, get_max_radius(pts)

    def lloyd_relax(pts, n_iters=200):
        """
        Lloyd iteration: Move each point towards the circumcenter of the 
        largest facet it belongs to.
        """
        pts = pts.copy()
        
        for it in range(n_iters):
            if time.time() > deadline - 5.0:
                break
                
            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break
            
            # For each point, find the facet it belongs to with the largest radius
            # and move towards its circumcenter
            targets = pts.copy()
            counts = np.zeros(N)
            
            for k in range(len(facets)):
                rad = radii[k]
                if rad < 1e-10:
                    continue
                
                cc = circumcenters[k]
                # We want to move points towards the centers of their "worst" facets
                # A simple heuristic: average of all circumcenters weighted by radius
                # Or just pick the max radius facet for each point.
                # Let's use a weighted average for stability
                weight = rad**2 
                
                for idx in facets[k]:
                    targets[idx] += weight * cc
                    counts[idx] += weight
            
            # Apply update
            mask = counts > 1e-10
            if np.any(mask):
                # Normalize target vectors
                target_norms = np.linalg.norm(targets[mask], axis=1, keepdims=True)
                target_norms[target_norms < 1e-12] = 1.0
                targets[mask] /= target_norms
                
                # Interpolate towards target
                step_size = 0.5
                pts[mask] = (1 - step_size) * pts[mask] + step_size * targets[mask]
                pts = normalize(pts)
                
        return pts, get_max_radius(pts)

    def tangent_gradient_descent(pts, obj_func, n_iters=500, lr=0.05):
        """
        Gradient descent on the sphere using numerical gradients of the soft-max objective.
        """
        pts = pts.copy()
        
        for it in range(n_iters):
            if time.time() > deadline - 5.0:
                break
                
            # Compute numerical gradient
            grad = np.zeros_like(pts)
            eps = 1e-5
            
            for i in range(N):
                for j in range(3):
                    p_plus = pts.copy()
                    p_plus[i, j] += eps
                    p_plus = normalize(p_plus)
                    
                    p_minus = pts.copy()
                    p_minus[i, j] -= eps
                    p_minus = normalize(p_minus)
                    
                    f_plus = obj_func(p_plus.flatten())
                    f_minus = obj_func(p_minus.flatten())
                    
                    grad[i, j] = (f_plus - f_minus) / (2 * eps)
            
            # Project gradient to tangent space
            grad_tan = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
            
            # Normalize gradient
            norm_grad = np.linalg.norm(grad_tan)
            if norm_grad < 1e-10:
                break
                
            # Line search
            improved = False
            step = lr
            current_val = obj_func(pts.flatten())
            
            for _ in range(5):
                new_pts = pts - step * grad_tan
                new_pts = normalize(new_pts)
                new_val = obj_func(new_pts.flatten())
                
                if new_val < current_val:
                    pts = new_pts
                    current_val = new_val
                    improved = True
                    break
                step *= 0.5
            
            if improved:
                lr = min(lr * 1.05, 0.5)
            else:
                lr *= 0.8
                
        return pts

    # Generate initial configurations
    initial_configs = [
        fibonacci_initial(0.0),
        fibonacci_initial(0.37),
        fibonacci_initial(0.618),
        greedy_initial(101),
        greedy_initial(202),
        greedy_initial(303),
    ]

    best_pts = initial_configs[0].copy()
    best_radius = get_max_radius(best_pts)

    # Phase 1: L-BFGS-B on soft max objective with progressive beta
    betas = [50, 100, 200, 500, 1000]
    
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 30.0:
            break

        current_pts = init_pts.copy()
        current_radius = get_max_radius(current_pts)
        
        for beta in betas:
            if time.time() > deadline - 20.0:
                break
                
            def obj_sm(params, _beta=beta):
                return soft_max_objective(params, beta=_beta)

            try:
                res = minimize(obj_sm, current_pts.flatten(), method='L-BFGS-B',
                               options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100})
                if res.fun < obj_sm(current_pts.flatten()):
                    new_pts = normalize(res.x.reshape(N, 3))
                    new_radius = get_max_radius(new_pts)
                    if new_radius < current_radius:
                        current_pts = new_pts
                        current_radius = new_radius
            except Exception:
                pass
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = current_pts.copy()

    # Phase 2: Pressure Relaxation
    if time.time() < deadline - 15.0:
        pts, radius = pressure_relax(best_pts, n_iters=300)
        if radius < best_radius:
            best_radius = radius
            best_pts = pts.copy()

    # Phase 3: Lloyd Relaxation
    if time.time() < deadline - 10.0:
        pts, radius = lloyd_relax(best_pts, n_iters=200)
        if radius < best_radius:
            best_radius = radius
            best_pts = pts.copy()

    # Phase 4: Final Polish with High Beta L-BFGS
    if time.time() < deadline - 5.0:
        def obj_final(params):
            return soft_max_objective(params, beta=2000.0)

        try:
            res = minimize(obj_final, best_pts.flatten(), method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200})
            if res.fun < obj_final(best_pts.flatten()):
                new_pts = normalize(res.x.reshape(N, 3))
                new_radius = get_max_radius(new_pts)
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_pts = new_pts.copy()
        except Exception:
            pass

    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        best_pts = fibonacci_initial()

    best_pts = normalize(best_pts)
    return best_pts
