import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

    def fibonacci_initial():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    # State for caching hull data
    hull_cache = {'equations': None, 'simplices': None}

    def compute_hull_data(p, use_cache=True):
        """Compute hull facets and their circumcenters/radii (vectorized)."""
        try:
            if use_cache and hull_cache['equations'] is not None:
                # Check if current points satisfy the previous hull inequalities
                # n . x + d <= 0  (scipy uses n . x + d = 0 for boundary, interior is < 0)
                # Wait, scipy equations are n . x + d = 0.
                # For convex hull, all points must satisfy n . x + d <= 0 (if n is outward normal? or inward?)
                # Let's check scipy docs: "equations : (M, 4) array. The equations of the hyperplanes... n . x + d = 0"
                # The normal points OUTWARD from the hull.
                # So for any point p in the hull, n . p + d <= 0.
                
                eqs = hull_cache['equations']
                dots = np.dot(p, eqs[:, :3].T) + eqs[:, 3]
                if np.all(dots <= 1e-7):
                    # Hull structure is likely unchanged.
                    # Verify that the vertices of the cached simplices are still the ones defining the facets?
                    # The check above ensures all points are inside the old hull.
                    # It does NOT guarantee that the old facets are still the facets of the new hull.
                    # However, if points move slightly, the combinatorial structure usually persists.
                    # We assume it's valid if all points are inside.
                    # We must still compute the new circumcenters/radii based on the NEW positions of the vertices.
                    facets = hull_cache['simplices']
                    return _compute_radii_from_facets(p, facets)
            
            # Recompute hull
            hull = ConvexHull(p)
            facets = hull.simplices
            equations = hull.equations
            
            # Update cache
            hull_cache['simplices'] = facets
            hull_cache['equations'] = equations
            
            return _compute_radii_from_facets(p, facets)
            
        except Exception:
            return None, None, None

    def _compute_radii_from_facets(p, facets):
        """Vectorized circumcenter computation from facets."""
        if facets is None or len(facets) == 0:
            return None, None, None
            
        n_facets = len(facets)
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

    def compute_covering_radius(p):
        """Compute the covering radius from hull data."""
        facets, circumcenters, radii = compute_hull_data(p)
        if facets is None:
            return float('inf')
        return np.max(radii)

    def lloyd_iteration(p):
        """
        Perform one Lloyd iteration.
        """
        facets, circumcenters, radii = compute_hull_data(p)
        if facets is None:
            return p
        
        # For each circumcenter, find the nearest site (max dot product with p)
        dots_all = circumcenters @ p.T
        nearest_idx = np.argmax(dots_all, axis=1)
        
        # Accumulate targets for each site
        sums = np.zeros((N, 3))
        counts = np.zeros(N)
        
        np.add.at(sums, nearest_idx, circumcenters)
        np.add.at(counts, nearest_idx, 1)
        
        new_p = p.copy()
        for i in range(N):
            if counts[i] > 0:
                target = sums[i]
                n = np.linalg.norm(target)
                if n > 1e-10:
                    new_p[i] = target / n
                else:
                    new_p[i] = p[i]
            else:
                new_p[i] = p[i]
                
        return normalize(new_p)

    def repulsion_relax(points, alpha_init=0.05, decay=0.995, max_iter=200, deadline=None):
        """
        Repulsion Relaxation (Thomson Problem solver).
        """
        points = points.copy()
        alpha = alpha_init
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            if time.time() > deadline - 5.0:
                break

            # Calculate forces: F_i = sum_j (p_i - p_j) / ||p_i - p_j||^2
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            
            mask = np.eye(N, dtype=bool)
            inv_dist_sq = np.zeros_like(dist_sq)
            np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)
            
            forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
            
            # Project onto tangent plane: F_tan = F - (F . p) p
            dot_fp = np.sum(forces * points, axis=1)
            forces_tan = forces - dot_fp[:, np.newaxis] * points
            
            # Update
            points = points + alpha * forces_tan
            points = normalize(points)
            
            # Decay alpha
            alpha *= decay
            
        return points

    def soft_max_objective(params, beta=200.0):
        """
        Soft-max surrogate: approximates max(radii) via log-sum-exp.
        """
        X = params.reshape(N, 3)
        X = normalize(X)
        facets, circumcenters, radii = compute_hull_data(X)
        if facets is None:
            return 1e10
            
        r_max = np.max(radii)
        if r_max < 1e-15:
            return r_max
        shifted = beta * (radii - r_max)
        log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + r_max
        return log_sum_exp

    def optimize_from_initial(pts_init, time_budget):
        """Run multi-phase optimization from a given initial configuration."""
        best_pts = pts_init.copy()
        best_radius = compute_covering_radius(best_pts)
        end_time = start_time + time_budget

        # Phase 1: Lloyd's Algorithm
        pts = best_pts.copy()
        lloyd_steps = 0
        while time.time() < end_time - 10.0 and lloyd_steps < 100:
            new_pts = lloyd_iteration(pts)
            if np.allclose(new_pts, pts, atol=1e-8):
                break
            pts = new_pts
            lloyd_steps += 1
            
            current_radius = compute_covering_radius(pts)
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
        
        # Phase 2: L-BFGS-B on Soft-Max Objective (Refinement)
        # We use the cached hull to speed this up significantly.
        if time.time() < end_time - 2.0:
            pts = best_pts.copy()
            
            # Use a high beta for precision, but not too high to avoid numerical issues
            # Since the objective is now faster due to caching, we can afford higher beta
            beta_lbfgs = 500.0
            
            def obj_lbfgs(params):
                return soft_max_objective(params, beta=beta_lbfgs)

            try:
                # Initial evaluation to warm up cache
                obj_lbfgs(pts.flatten())
                
                res = minimize(obj_lbfgs, pts.flatten(), method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
                if res.success or res.fun < obj_lbfgs(pts.flatten()):
                    new_pts = normalize(res.x.reshape(N, 3))
                    current_radius = compute_covering_radius(new_pts)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = new_pts.copy()
            except Exception:
                pass

        # Phase 3: Final Lloyd's Polish
        if time.time() < end_time - 1.0:
            pts = best_pts.copy()
            for _ in range(20):
                if time.time() > end_time - 0.5:
                    break
                new_pts = lloyd_iteration(pts)
                if np.allclose(new_pts, pts, atol=1e-9):
                    break
                pts = new_pts
                current_radius = compute_covering_radius(pts)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = pts.copy()

        return best_pts, best_radius

    # Multi-restart strategy
    
    # Restart 1: Fibonacci
    fib_pts = fibonacci_initial()
    best_pts, best_radius = optimize_from_initial(fib_pts, 60.0)

    # Restart 2: Repulsion-relaxed random start
    if time.time() < deadline - 40.0:
        rng = np.random.default_rng(42)
        random_pts = rng.normal(0, 1, (N, 3))
        random_pts = normalize(random_pts)
        # Relax to spread out
        relaxed_pts = repulsion_relax(random_pts, max_iter=100, deadline=deadline - 30.0)
        pts2, rad2 = optimize_from_initial(relaxed_pts, 30.0)
        if rad2 < best_radius:
            best_radius = rad2
            best_pts = pts2

    # Restart 3: Perturbed best
    if time.time() < deadline - 20.0:
        rng = np.random.default_rng(123)
        perturbed = best_pts + rng.normal(0, 0.03, (N, 3))
        perturbed = normalize(perturbed)
        pts3, rad3 = optimize_from_initial(perturbed, 15.0)
        if rad3 < best_radius:
            best_radius = rad3
            best_pts = pts3

    # Restart 4: Different random relaxed
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(999)
        random_pts = rng.normal(0, 1, (N, 3))
        random_pts = normalize(random_pts)
        relaxed_pts = repulsion_relax(random_pts, max_iter=100, deadline=deadline - 5.0)
        pts4, rad4 = optimize_from_initial(relaxed_pts, 5.0)
        if rad4 < best_radius:
            best_radius = rad4
            best_pts = pts4

    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        best_pts = fibonacci_initial()
        best_radius = compute_covering_radius(best_pts)

    best_pts = normalize(best_pts)
    return best_pts
