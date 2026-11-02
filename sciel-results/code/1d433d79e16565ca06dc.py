import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2s safety margin

    def fibonacci_initial(phase=0.0):
        i = np.arange(N) + 0.5 + phase
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

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
            dots = np.sum(cc * a[valid], axis=1)
            cc[dots < 0] *= -1
            circumcenters[valid] = cc
            cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
            radii[valid] = np.arccos(cos_r)

        return facets, circumcenters, radii

    def compute_covering_radius(p):
        facets, circumcenters, radii = compute_hull_data(p)
        if facets is None:
            return float('inf')
        return np.max(radii)

    def compute_forces(p, beta=50.0, repulsion_strength=0.0):
        facets, circumcenters, radii = compute_hull_data(p)
        if facets is None:
            return None, None, None, None

        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None, None

        r2 = radii * radii
        max_r2 = np.max(r2)
        exp_terms = np.exp(beta * (r2 - max_r2))
        sum_exp = np.sum(exp_terms)
        
        # Weights for soft-max gradient approximation
        # Gradient of (1/beta) log sum exp(beta r_f^2) is (2/beta) sum w_f r_f grad(r_f)
        # w_f = exp(beta r_f^2) / sum exp
        # We use w_f * r_f * (C_f - p_i) as the force direction
        weights = (2.0 / beta) * radii * exp_terms / sum_exp

        forces = np.zeros_like(p)
        vertex_indices = facets.ravel()
        facet_indices = np.repeat(np.arange(n_facets), 3)
        w_expanded = weights[facet_indices]
        cc_expanded = circumcenters[facet_indices]
        p_expanded = p[vertex_indices]

        # Force direction is towards circumcenter
        contributions = w_expanded[:, np.newaxis] * (cc_expanded - p_expanded)
        np.add.at(forces, vertex_indices, contributions)

        if repulsion_strength > 0:
            diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dist_sq = np.sum(diffs * diffs, axis=2)
            np.fill_diagonal(dist_sq, 1.0)
            inv_dist_sq = 1.0 / dist_sq
            np.fill_diagonal(inv_dist_sq, 0.0)
            forces_rep = np.sum(diffs * inv_dist_sq[:, :, np.newaxis], axis=1)
            forces += repulsion_strength * forces_rep

        return forces, facets, circumcenters, radii

    def optimize_gradient_descent(pts_init, time_budget):
        best_pts = pts_init.copy()
        best_radius = compute_covering_radius(best_pts)
        end_time = start_time + time_budget
        
        pts = best_pts.copy()
        
        # Phase 1: Coarse GD
        beta = 20.0
        eta = 0.05
        n_steps = 2000
        
        for step in range(n_steps):
            if time.time() > end_time - 10.0:
                break
            
            forces, facets, circumcenters, radii = compute_forces(pts, beta, repulsion_strength=0.01)
            if forces is None:
                break
            
            # Project forces to tangent plane
            dot_fp = np.sum(forces * pts, axis=1)
            forces_tan = forces - dot_fp[:, np.newaxis] * pts
            
            # Line search
            step_size = eta
            improved = False
            for _ in range(5):
                new_pts = pts + step_size * forces_tan
                new_pts = normalize(new_pts)
                
                # Check radius
                new_radius = compute_covering_radius(new_pts)
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_pts = new_pts.copy()
                    pts = new_pts
                    improved = True
                    break
                step_size *= 0.5
            
            if not improved:
                eta *= 0.9
            else:
                eta *= 1.05
            
            eta = min(eta, 0.1)
            beta += 0.1 # Slowly increase beta to focus on max

        # Phase 2: Local Facet Moves
        if time.time() < end_time - 5.0:
            pts = best_pts.copy()
            for _ in range(500):
                if time.time() > end_time - 2.0:
                    break
                
                facets, circumcenters, radii = compute_hull_data(pts)
                if facets is None:
                    break
                
                current_radius = np.max(radii)
                
                # Find top 5 worst facets
                worst_idx = np.argsort(radii)[-5:]
                
                improved_this_iter = False
                for idx in worst_idx:
                    if time.time() > end_time - 2.0:
                        break
                    
                    facet = facets[idx]
                    cc = circumcenters[idx]
                    
                    # Try moving each vertex of this facet towards cc
                    for v_idx in facet:
                        for step_factor in [0.05, 0.02, 0.01]:
                            new_pts = pts.copy()
                            new_pts[v_idx] = pts[v_idx] + step_factor * (cc - pts[v_idx])
                            new_pts = normalize(new_pts)
                            
                            new_radius = compute_covering_radius(new_pts)
                            if new_radius < current_radius - 1e-9:
                                pts = new_pts
                                current_radius = new_radius
                                improved_this_iter = True
                                break
                        if improved_this_iter:
                            break
                
                if not improved_this_iter:
                    # Try small random perturbation to escape local min
                    if np.random.rand() < 0.1:
                        pts = pts + np.random.normal(0, 0.01, (N, 3))
                        pts = normalize(pts)

                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = pts.copy()

        # Phase 3: Fine GD
        if time.time() < end_time - 1.0:
            pts = best_pts.copy()
            beta = 100.0
            eta = 0.01
            n_steps = 1000
            
            for step in range(n_steps):
                if time.time() > end_time - 0.5:
                    break
                
                forces, facets, circumcenters, radii = compute_forces(pts, beta, repulsion_strength=0.001)
                if forces is None:
                    break
                
                dot_fp = np.sum(forces * pts, axis=1)
                forces_tan = forces - dot_fp[:, np.newaxis] * pts
                
                step_size = eta
                improved = False
                for _ in range(5):
                    new_pts = pts + step_size * forces_tan
                    new_pts = normalize(new_pts)
                    new_radius = compute_covering_radius(new_pts)
                    if new_radius < best_radius:
                        best_radius = new_radius
                        best_pts = new_pts.copy()
                        pts = new_pts
                        improved = True
                        break
                    step_size *= 0.5
                
                if not improved:
                    eta *= 0.9
                beta += 0.5

        return best_pts, best_radius

    # Multi-restart
    fib_pts = fibonacci_initial(0.0)
    best_pts, best_radius = optimize_gradient_descent(fib_pts, 100.0)

    if time.time() < deadline - 30.0:
        fib2 = fibonacci_initial(0.5)
        pts2, rad2 = optimize_gradient_descent(fib2, 20.0)
        if rad2 < best_radius:
            best_radius = rad2
            best_pts = pts2

    if time.time() < deadline - 15.0:
        rng = np.random.default_rng(42)
        perturbed = best_pts + rng.normal(0, 0.05, (N, 3))
        perturbed = normalize(perturbed)
        pts3, rad3 = optimize_gradient_descent(perturbed, 10.0)
        if rad3 < best_radius:
            best_radius = rad3
            best_pts = pts3

    if time.time() < deadline - 5.0:
        rng = np.random.default_rng(123)
        random_pts = rng.normal(0, 1, (N, 3))
        random_pts = normalize(random_pts)
        pts4, rad4 = optimize_gradient_descent(random_pts, 5.0)
        if rad4 < best_radius:
            best_radius = rad4
            best_pts = pts4

    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        best_pts = fibonacci_initial(0.0)
        best_radius = compute_covering_radius(best_pts)

    best_pts = normalize(best_pts)
    return best_pts
