import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    def compute_hull_data(p):
        """Compute hull facets and their circumcenters/radii (vectorized)."""
        try:
            hull = ConvexHull(p)
            facets = hull.simplices
        except Exception:
            return None, None, None
        
        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None
        
        # Vectorized circumcenter computation
        a = p[facets[:, 0]]  # (F, 3)
        b = p[facets[:, 1]]
        c = p[facets[:, 2]]
        
        v1 = a - b  # (F, 3)
        v2 = b - c
        n = np.cross(v1, v2)  # (F, 3)
        norm_n = np.linalg.norm(n, axis=1)  # (F,)
        
        # Handle degenerate facets
        valid = norm_n > 1e-12
        circumcenters = np.zeros((n_facets, 3))
        radii = np.zeros(n_facets)
        
        if np.any(valid):
            cc = n[valid] / norm_n[valid, np.newaxis]
            # Ensure circumcenter faces outward (toward 'a')
            dots = np.sum(cc * a[valid], axis=1)
            cc[dots < 0] *= -1
            circumcenters[valid] = cc
            
            cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
            radii[valid] = np.arccos(cos_r)
        
        return facets, circumcenters, radii

    def compute_forces(p, beta=50.0, repulsion_weight=0.1):
        """Compute smooth forces from soft-max energy + repulsion."""
        facets, circumcenters, radii = compute_hull_data(p)
        if facets is None:
            return None, None, None, None
        
        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None, None
        
        # Soft-max energy: E = (1/beta) * log(sum(exp(beta * r_i^2)))
        # Gradient: dE/dr_i = r_i * exp(beta * r_i^2) / sum(exp(beta * r_j^2))
        r2 = radii * radii
        max_r2 = np.max(r2)
        exp_terms = np.exp(beta * (r2 - max_r2))
        sum_exp = np.sum(exp_terms)
        weights = radii * exp_terms / sum_exp  # dE/dr_i

        forces = np.zeros_like(p)
        
        # Attractive forces from facet circumcenters
        for k in range(n_facets):
            w = weights[k]
            if w < 1e-15:
                continue
            cc = circumcenters[k]
            for idx in facets[k]:
                forces[idx] += w * (cc - p[idx])
        
        # Repulsive forces (vectorized) - helps avoid coincident points
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diffs * diffs, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        forces_rep = np.sum(diffs * inv_dist_sq[:, :, np.newaxis], axis=1)
        forces += repulsion_weight * forces_rep
        
        return forces, facets, circumcenters, radii

    def optimize_from_start(pts_init, time_budget):
        """Run the full optimization pipeline from a starting configuration."""
        pts = pts_init.copy()
        best_pts_local = pts.copy()
        best_radius_local = float('inf')
        
        # Phase 1: Coarse optimization with adaptive beta
        eta = 0.015
        beta = 20.0
        rep_weight = 0.15
        
        for step in range(1200):
            if time.time() > time_budget - 8.0:
                break
            
            forces, facets, circumcenters, radii = compute_forces(pts, beta, rep_weight)
            if forces is None:
                break
            
            pts += eta * forces
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            pts = pts / norms
            
            if radii is not None:
                current_radius = np.max(radii)
                if current_radius < best_radius_local:
                    best_radius_local = current_radius
                    best_pts_local = pts.copy()
            
            eta *= 0.999
            beta += 0.3
            rep_weight *= 0.999
        
        # Phase 2: L-BFGS-B Polish on sum of squared radii
        if time.time() < time_budget - 6.0:
            def objective_sum_sq(params):
                X = params.reshape(N, 3)
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                X = X / norms
                facets, circumcenters, radii = compute_hull_data(X)
                if facets is None:
                    return 1e10
                return np.sum(radii**2)
            
            try:
                res = minimize(objective_sum_sq, best_pts_local.flatten(), method='L-BFGS-B',
                               options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200})
                if res.fun < objective_sum_sq(best_pts_local.flatten()):
                    new_pts = res.x.reshape(N, 3)
                    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                    norms[norms < 1e-12] = 1.0
                    new_pts = new_pts / norms
                    facets, circumcenters, radii = compute_hull_data(new_pts)
                    if radii is not None:
                        current_radius = np.max(radii)
                        if current_radius < best_radius_local:
                            best_radius_local = current_radius
                            best_pts_local = new_pts.copy()
            except Exception:
                pass
        
        # Phase 3: Fine Polish with higher beta
        if time.time() < time_budget - 3.0:
            pts = best_pts_local.copy()
            eta = 0.003
            beta = 80.0
            rep_weight = 0.05
            
            for step in range(400):
                if time.time() > time_budget - 2.0:
                    break
                forces, facets, circumcenters, radii = compute_forces(pts, beta, rep_weight)
                if forces is None:
                    break
                pts += eta * forces
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                pts = pts / norms
                if radii is not None:
                    current_radius = np.max(radii)
                    if current_radius < best_radius_local:
                        best_radius_local = current_radius
                        best_pts_local = pts.copy()
                eta *= 0.997
        
        # Phase 4: Final L-BFGS-B with very tight tolerances
        if time.time() < time_budget - 1.5:
            def objective_final(params):
                X = params.reshape(N, 3)
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                X = X / norms
                facets, circumcenters, radii = compute_hull_data(X)
                if facets is None:
                    return 1e10
                return np.sum(radii**2)
            
            try:
                res = minimize(objective_final, best_pts_local.flatten(), method='L-BFGS-B',
                               options={'maxiter': 30, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 300})
                if res.fun < objective_final(best_pts_local.flatten()):
                    new_pts = res.x.reshape(N, 3)
                    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                    norms[norms < 1e-12] = 1.0
                    new_pts = new_pts / norms
                    facets, circumcenters, radii = compute_hull_data(new_pts)
                    if radii is not None:
                        current_radius = np.max(radii)
                        if current_radius < best_radius_local:
                            best_radius_local = current_radius
                            best_pts_local = new_pts.copy()
            except Exception:
                pass
        
        return best_pts_local, best_radius_local

    # Generate multiple initial configurations with better diversity
    # Config 1: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Config 2: Fibonacci with different offset
    i2 = np.arange(N) + 0.5
    z2 = 1.0 - 2.0 * i2 / N
    phi2 = np.pi * (1.0 + np.sqrt(5.0)) * (i2 + 0.37)
    r2_ = np.sqrt(1.0 - z2 * z2)
    fib2_pts = np.column_stack([r2_ * np.cos(phi2), r2_ * np.sin(phi2), z2])
    
    # Config 3: Different golden ratio phase
    i3 = np.arange(N) + 0.5
    z3 = 1.0 - 2.0 * i3 / N
    phi3 = np.pi * (1.0 + np.sqrt(5.0)) * (i3 + 0.71)
    r3 = np.sqrt(1.0 - z3 * z3)
    fib3_pts = np.column_stack([r3 * np.cos(phi3), r3 * np.sin(phi3), z3])
    
    # Config 4: Spherical t-design-like initial (use scipy if available, else fallback)
    try:
        from scipy.spatial.transform import Rotation
        # Create points with better initial spread using rotation groups
        base_pts = []
        # Add icosahedron vertices
        phi_golden = (1 + np.sqrt(5)) / 2
        ico = np.array([
            [0, 1, phi_golden], [0, -1, phi_golden], [0, 1, -phi_golden], [0, -1, -phi_golden],
            [1, phi_golden, 0], [-1, phi_golden, 0], [1, -phi_golden, 0], [-1, -phi_golden, 0],
            [phi_golden, 0, 1], [-phi_golden, 0, 1], [phi_golden, 0, -1], [-phi_golden, 0, -1],
            [1, 0, -phi_golden], [-1, 0, -phi_golden], [1, 0, phi_golden], [-1, 0, phi_golden]
        ], dtype=float)
        ico /= np.linalg.norm(ico, axis=1, keepdims=True)
        base_pts.append(ico)
        
        # Add more points via rotations
        all_pts = ico.copy()
        for angle in [np.pi/5, 2*np.pi/5, 3*np.pi/5]:
            rot = Rotation.from_rotvec([0, 0, angle])
            rotated = rot.apply(ico)
            all_pts = np.vstack([all_pts, rotated])
        
        # Remove duplicates and take first N
        unique_pts = []
        for p in all_pts:
            is_dup = False
            for u in unique_pts:
                if np.dot(p, u) > 0.9999:
                    is_dup = True
                    break
            if not is_dup:
                unique_pts.append(p)
            if len(unique_pts) >= N:
                break
        
        if len(unique_pts) < N:
            # Pad with remaining fibonacci points
            for p in fib_pts[len(unique_pts):]:
                unique_pts.append(p)
                if len(unique_pts) >= N:
                    break
        
        ico_pts = np.array(unique_pts[:N])
        ico_pts /= np.linalg.norm(ico_pts, axis=1, keepdims=True)
    except Exception:
        ico_pts = fib_pts.copy()
    
    initial_configs = [fib_pts, fib2_pts, fib3_pts, ico_pts]
    
    best_pts = fib_pts.copy()
    best_radius = float('inf')
    
    # Allocate time budget across restarts with 4 configs
    n_configs = len(initial_configs)
    
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 2.0:
            break
        
        # Give each restart a portion of remaining time, decreasing
        remaining = deadline - time.time() - 2.0
        if remaining < 5.0:
            break
        
        # Allocate time proportionally, first gets most
        if idx == 0:
            tb = deadline - 20.0
        elif idx == 1:
            tb = deadline - 12.0
        elif idx == 2:
            tb = deadline - 6.0
        else:
            tb = deadline - 2.5
        
        if time.time() > tb - 1.0:
            continue
        
        pts_res, rad_res = optimize_from_start(init_pts, tb)
        
        if rad_res < best_radius:
            best_radius = rad_res
            best_pts = pts_res.copy()
    
    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Final high-precision polish
    if time.time() < deadline - 1.5:
        # Use soft-max surrogate with high beta for final minimax polish
        def objective_softmax(params, beta=150.0):
            X = params.reshape(N, 3)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            X = X / norms
            facets, circumcenters, radii = compute_hull_data(X)
            if facets is None:
                return 1e10
            r2 = radii * radii
            max_r2 = np.max(r2)
            return (1.0 / beta) * np.log(np.sum(np.exp(beta * (r2 - max_r2)))) + max_r2 / beta
        
        try:
            res = minimize(objective_softmax, best_pts.flatten(), method='L-BFGS-B',
                           options={'maxiter': 40, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 300})
            if res.fun < objective_softmax(best_pts.flatten()):
                new_pts = res.x.reshape(N, 3)
                norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                new_pts = new_pts / norms
                facets, circumcenters, radii = compute_hull_data(new_pts)
                if radii is not None:
                    current_radius = np.max(radii)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = new_pts.copy()
        except Exception:
            pass
        
        # One more sum-sq polish
        if time.time() < deadline - 1.0:
            def objective_sum_sq_final(params):
                X = params.reshape(N, 3)
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                X = X / norms
                facets, circumcenters, radii = compute_hull_data(X)
                if facets is None:
                    return 1e10
                return np.sum(radii**2)
            
            try:
                res = minimize(objective_sum_sq_final, best_pts.flatten(), method='L-BFGS-B',
                               options={'maxiter': 20, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 200})
                if res.fun < objective_sum_sq_final(best_pts.flatten()):
                    new_pts = res.x.reshape(N, 3)
                    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                    norms[norms < 1e-12] = 1.0
                    new_pts = new_pts / norms
                    facets, circumcenters, radii = compute_hull_data(new_pts)
                    if radii is not None:
                        current_radius = np.max(radii)
                        if current_radius < best_radius:
                            best_radius = current_radius
                            best_pts = new_pts.copy()
            except Exception:
                pass
    
    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return best_pts
