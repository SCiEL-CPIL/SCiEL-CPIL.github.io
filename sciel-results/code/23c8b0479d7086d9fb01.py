import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    def fibonacci_initial():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

    def compute_hull_data(p):
        """Compute hull facets and their circumcenters/radii."""
        try:
            hull = ConvexHull(p)
            facets = hull.simplices
        except Exception:
            return None, None, None
        
        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None
        
        circumcenters = np.zeros((n_facets, 3))
        radii = np.zeros(n_facets)
        
        for k, facet in enumerate(facets):
            a = p[facet[0]]
            b = p[facet[1]]
            c = p[facet[2]]
            v1 = a - b
            v2 = b - c
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-12:
                circumcenters[k] = a
                radii[k] = 0.0
                continue
            cc = n / norm_n
            if np.dot(cc, a) < 0:
                cc = -cc
            circumcenters[k] = cc
            cos_r = np.clip(np.dot(cc, a), -1.0, 1.0)
            radii[k] = np.arccos(cos_r)
        
        return facets, circumcenters, radii

    def compute_forces(p, beta=50.0):
        """Compute smooth forces from soft-max energy + repulsion."""
        facets, circumcenters, radii = compute_hull_data(p)
        if facets is None:
            return None, None, None, None
        
        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None, None
        
        # Soft-max energy: E = (1/beta) * log(sum(exp(beta * r_i^2)))
        r2 = radii * radii
        max_r2 = np.max(r2)
        exp_terms = np.exp(beta * (r2 - max_r2))
        sum_exp = np.sum(exp_terms)
        weights = (2.0 / beta) * radii * exp_terms / sum_exp

        forces = np.zeros_like(p)
        
        for k in range(n_facets):
            w = weights[k]
            if w < 1e-15:
                continue
            cc = circumcenters[k]
            for idx in facets[k]:
                forces[idx] += w * (cc - p[idx])
        
        # Repulsive forces (vectorized)
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diffs * diffs, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        forces_rep = np.sum(diffs * inv_dist_sq[:, :, np.newaxis], axis=1)
        forces += 0.1 * forces_rep
        
        return forces, facets, circumcenters, radii

    def optimize_worst_facet(pts, step_size, iterations=100):
        """Iteratively move vertices of worst facet toward its circumcenter."""
        pts = pts.copy()
        prev_radius = np.pi
        
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break
            
            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break
            
            current_radius = np.max(radii)
            worst_idx = np.argmax(radii)
            worst_center = circumcenters[worst_idx]
            
            if current_radius >= prev_radius:
                step_size *= 0.5
                if step_size < 1e-8:
                    break
            else:
                prev_radius = current_radius
            
            # Find all facets near the worst center
            target_indices_set = set()
            for k in range(len(facets)):
                if np.dot(circumcenters[k], worst_center) > 0.999:
                    for idx in facets[k]:
                        target_indices_set.add(idx)
            
            if not target_indices_set:
                break
            
            for idx in target_indices_set:
                v = pts[idx]
                dir_vec = worst_center - v
                radial_comp = np.dot(dir_vec, v)
                dir_vec = dir_vec - radial_comp * v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    pts[idx] = v + step_size * dir_vec
                    pts[idx] /= np.linalg.norm(pts[idx])
        
        return pts

    best_pts = fibonacci_initial()
    best_radius = float('inf')
    
    # Phase 1: Coarse optimization with soft-max forces
    pts = best_pts.copy()
    eta = 0.02
    beta = 30.0
    
    for step in range(1500):
        if time.time() > deadline - 30.0:
            break
        
        forces, facets, circumcenters, radii = compute_forces(pts, beta)
        if forces is None:
            break
        
        pts += eta * forces
        pts = normalize(pts)
        
        if radii is not None:
            current_radius = np.max(radii)
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
        
        eta *= 0.998
        beta += 0.5
    
    # Phase 2: L-BFGS-B on sum of squared radii
    if time.time() < deadline - 25.0:
        pts = best_pts.copy()
        
        def objective_sum_sq(params):
            X = params.reshape(N, 3)
            X = normalize(X)
            
            facets, circumcenters, radii = compute_hull_data(X)
            if facets is None:
                return 1e10
            
            r = radii
            return np.sum(r**2) + 0.01 * np.var(r)
        
        try:
            res = minimize(objective_sum_sq, pts.flatten(), method='L-BFGS-B', 
                           options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100})
            
            if res.success or res.fun < objective_sum_sq(pts.flatten()):
                new_pts = normalize(res.x.reshape(N, 3))
                
                facets, circumcenters, radii = compute_hull_data(new_pts)
                if radii is not None:
                    current_radius = np.max(radii)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = new_pts.copy()
                        pts = new_pts.copy()
        except Exception:
            pass

    # Phase 3: Adaptive worst-facet optimization with multiple step sizes
    if time.time() < deadline - 15.0:
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        for step in step_sizes:
            if time.time() > deadline - 10:
                break
            
            current = optimize_worst_facet(best_pts.copy(), step, iterations=80)
            facets, circumcenters, radii = compute_hull_data(current)
            if radii is not None:
                radius = np.max(radii)
                if radius < best_radius:
                    best_radius = radius
                    best_pts = current.copy()
    
    # Phase 4: Random restarts with small perturbations
    if time.time() < deadline - 10.0:
        rng = np.random.RandomState(42)
        for restart in range(30):
            if time.time() > deadline - 5:
                break
            
            # Perturb best solution
            noise = rng.randn(N, 3) * 0.05
            perturbed = best_pts + noise
            perturbed = normalize(perturbed)
            
            # Quick optimization from perturbed start
            pts = perturbed.copy()
            eta = 0.01
            beta = 80.0
            
            for step in range(300):
                if time.time() > deadline - 3:
                    break
                
                forces, facets, circumcenters, radii = compute_forces(pts, beta)
                if forces is None:
                    break
                
                pts += eta * forces
                pts = normalize(pts)
                
                if radii is not None:
                    current_radius = np.max(radii)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = pts.copy()
                
                eta *= 0.995
                beta += 0.3
            
            # Also try worst-facet polish from this restart
            if time.time() < deadline - 3:
                current = optimize_worst_facet(pts, 0.01, iterations=50)
                facets, circumcenters, radii = compute_hull_data(current)
                if radii is not None:
                    radius = np.max(radii)
                    if radius < best_radius:
                        best_radius = radius
                        best_pts = current.copy()
    
    # Phase 5: Final fine polish with soft-max gradient
    if time.time() < deadline - 1.0:
        pts = best_pts.copy()
        eta = 0.002
        beta = 150.0
        
        for step in range(200):
            if time.time() > deadline - 0.5:
                break
            
            forces, facets, circumcenters, radii = compute_forces(pts, beta)
            if forces is None:
                break
            
            pts += eta * forces
            pts = normalize(pts)
            
            if radii is not None:
                current_radius = np.max(radii)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = pts.copy()
            
            eta *= 0.995
    
    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        best_pts = fibonacci_initial()
    
    best_pts = normalize(best_pts)
    
    return best_pts
