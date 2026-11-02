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
            # Generate many candidates
            candidates = rng.randn(10000, 3)
            candidates = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
            # Calculate min distance to existing points for each candidate
            # Using dot products: distance is arccos(dot). Min distance = max dot? No.
            # We want to maximize the minimum distance.
            # d(c, p) is large if dot(c, p) is small (close to -1).
            # So we want to minimize the max dot product with existing points?
            # Actually, we want the candidate c that maximizes min_{p in pts} d(c, p).
            # d(c, p) = arccos(c.p). Max d is min c.p.
            # So we want to find c that minimizes max_{p in pts} (c.p).
            # Wait. If c is close to p, c.p is large. If c is far, c.p is small.
            # We want c to be far from ALL p. So we want c.p to be small for all p.
            # So we want to minimize the maximum of (c.p) over p?
            # No, if c.p is -1, distance is pi. If c.p is 1, distance is 0.
            # We want distance to be large. So we want c.p to be small (negative).
            # So for a candidate c, its "quality" is min_{p in pts} d(c, p) = arccos(max_{p in pts} c.p).
            # To maximize this, we need to minimize max_{p in pts} c.p.
            
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
        # Handle overflow in exp
        shifted = np.clip(shifted, -500, 500)
        exp_terms = np.exp(shifted)
        sum_exp = np.sum(exp_terms)
        if sum_exp < 1e-300:
            return r_max
        log_sum_exp = np.log(sum_exp) / beta + r_max
        return log_sum_exp

    def compute_forces(p, beta=50.0):
        facets, circumcenters, radii = compute_hull_data(p)
        if facets is None:
            return None

        n_facets = len(facets)
        if n_facets == 0:
            return None

        r2 = radii * radii
        max_r2 = np.max(r2)
        # Soft max of squared radii
        shifted = beta * (r2 - max_r2)
        shifted = np.clip(shifted, -500, 500)
        exp_terms = np.exp(shifted)
        sum_exp = np.sum(exp_terms)
        if sum_exp < 1e-300:
            return None
        # Gradient of soft-max of r^2 w.r.t. r is 2*r*exp(beta*(r^2-rmax^2))/sum
        weights = (2.0 * radii * exp_terms) / sum_exp

        forces = np.zeros_like(p)

        for k in range(n_facets):
            w = weights[k]
            if w < 1e-15:
                continue
            cc = circumcenters[k]
            for idx in facets[k]:
                # Move vertex towards circumcenter
                # Project (cc - p[idx]) onto tangent space at p[idx]
                v = p[idx]
                dir_vec = cc - v
                radial_comp = np.dot(dir_vec, v)
                dir_vec = dir_vec - radial_comp * v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    forces[idx] += w * dir_vec

        return forces

    def optimize_worst_facet(pts, iterations=500):
        pts = pts.copy()
        best_radius = get_max_radius(pts)
        
        for _ in range(iterations):
            if time.time() > deadline - 2.0:
                break
                
            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break
                
            current_radius = np.max(radii)
            if current_radius < best_radius:
                best_radius = current_radius
                
            worst_idx = np.argmax(radii)
            worst_center = circumcenters[worst_idx]
            worst_facet = facets[worst_idx]
            
            # Try to move vertices of the worst facet towards the circumcenter
            # Use a line search to find a good step size
            improved = False
            for step_factor in [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]:
                new_pts = pts.copy()
                for idx in worst_facet:
                    v = new_pts[idx]
                    dir_vec = worst_center - v
                    radial_comp = np.dot(dir_vec, v)
                    dir_vec = dir_vec - radial_comp * v
                    norm_dir = np.linalg.norm(dir_vec)
                    if norm_dir > 1e-10:
                        dir_vec /= norm_dir
                        new_pts[idx] = v + step_factor * dir_vec
                        new_pts[idx] /= np.linalg.norm(new_pts[idx])
                        
                new_radius = get_max_radius(new_pts)
                if new_radius < current_radius:
                    pts = new_pts
                    improved = True
                    break
            
            if not improved:
                # Try moving neighbors of the worst facet
                # Find all facets sharing an edge with the worst facet
                neighbor_indices = set()
                for f in range(len(facets)):
                    if f == worst_idx:
                        continue
                    common = set(facets[worst_idx]) & set(facets[f])
                    if len(common) >= 2:
                        neighbor_indices.add(f)
                
                # Try moving vertices of neighbor facets
                for nf in sorted(neighbor_indices):
                    nc = circumcenters[nf]
                    nf_facet = facets[nf]
                    for step_factor in [0.05, 0.02, 0.01, 0.005]:
                        new_pts = pts.copy()
                        for idx in nf_facet:
                            v = new_pts[idx]
                            dir_vec = nc - v
                            radial_comp = np.dot(dir_vec, v)
                            dir_vec = dir_vec - radial_comp * v
                            norm_dir = np.linalg.norm(dir_vec)
                            if norm_dir > 1e-10:
                                dir_vec /= norm_dir
                                new_pts[idx] = v + step_factor * dir_vec
                                new_pts[idx] /= np.linalg.norm(new_pts[idx])
                                
                        new_radius = get_max_radius(new_pts)
                        if new_radius < current_radius:
                            pts = new_pts
                            improved = True
                            break
                    if improved:
                        break
                
                if not improved:
                    break
                    
        return pts, get_max_radius(pts)

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

    # Phase 1: L-BFGS-B on soft max objective
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 20.0:
            break

        def obj_sm(params, _beta=150.0):
            return soft_max_objective(params, beta=_beta)

        try:
            res = minimize(obj_sm, init_pts.flatten(), method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100})
            if res.fun < obj_sm(init_pts.flatten()):
                new_pts = normalize(res.x.reshape(N, 3))
                current_radius = get_max_radius(new_pts)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = new_pts.copy()
        except Exception:
            pass

    # Phase 2: Gradient descent with forces
    if time.time() < deadline - 10.0:
        pts = best_pts.copy()
        eta = 0.05
        beta = 50.0
        
        for step in range(3000):
            if time.time() > deadline - 5.0:
                break
                
            forces = compute_forces(pts, beta)
            if forces is None:
                break
                
            pts += eta * forces
            pts = normalize(pts)
            
            current_radius = get_max_radius(pts)
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
            
            # Adaptive step size
            if step % 100 == 0:
                eta *= 0.92
                beta += 3.0

    # Phase 3: Worst facet optimization
    if time.time() < deadline - 5.0:
        for _ in range(15): # Multiple passes of worst facet optimization
            if time.time() > deadline - 2.0:
                break
            new_pts, new_radius = optimize_worst_facet(best_pts, iterations=150)
            if new_radius < best_radius:
                best_radius = new_radius
                best_pts = new_pts
            else:
                break

    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        best_pts = fibonacci_initial()

    best_pts = normalize(best_pts)
    
    # Final high-precision polish: short L-BFGS with tighter tolerances
    if time.time() < deadline - 3.0:
        def obj_final(params):
            X = params.reshape(N, 3)
            X = normalize(X)
            facets, circumcenters, radii = compute_hull_data(X)
            if facets is None:
                return 1e10
            return np.max(radii)
        
        try:
            res = minimize(obj_final, best_pts.flatten(), method='L-BFGS-B',
                           options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            new_pts = normalize(res.x.reshape(N, 3))
            new_radius = get_max_radius(new_pts)
            if new_radius < best_radius:
                best_radius = new_radius
                best_pts = new_pts.copy()
        except Exception:
            pass
    
    best_pts = normalize(best_pts)
    return best_pts
