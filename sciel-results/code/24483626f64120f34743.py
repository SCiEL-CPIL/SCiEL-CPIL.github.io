"""Program for covering the sphere with 130 equal caps using facet-based optimization with CVT refinement."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

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
            
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            norms = np.linalg.norm(c_dir, axis=1)
            
            radii = np.zeros(len(faces))
            valid = norms > 1e-10
            
            if np.any(valid):
                c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
                cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r
            
            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
            
            return float(np.max(radii))
        except Exception:
            # Fallback: dense grid-based estimate
            pts = np.array(pts)
            grid = fibonacci_spiral(5000)
            dot = np.dot(grid, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    def get_facets(pts):
        """Get convex hull facets and their circumcenters/radii."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]
            
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            norms = np.linalg.norm(c_dir, axis=1)
            
            centers = np.zeros((len(faces), 3))
            radii = np.zeros(len(faces))
            valid = norms > 1e-10
            
            if np.any(valid):
                centers[valid] = c_dir[valid] / norms[valid, np.newaxis]
                cos_r = np.clip(np.sum(centers[valid] * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r
            
            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
                # For degenerate, use centroid
                centers[~valid] = (a_deg + b_deg + c_deg) / 3.0
                centers[~valid] /= np.linalg.norm(centers[~valid], axis=1, keepdims=True)
            
            return faces, centers, radii
        except Exception:
            return None, None, None

    def cvt_step(pts, n_iter=10, alpha=0.5):
        """
        Centroidal Voronoi Tiling step: move each point towards the center of its largest adjacent facet.
        This is a gradient-free optimization that directly targets the covering radius.
        """
        pts = pts.copy()
        for _ in range(n_iter):
            if time.time() > deadline - 5.0:
                break
            
            faces, centers, radii = get_facets(pts)
            if faces is None:
                break
            
            # For each point, find the largest facet it belongs to
            # and move it towards that facet's center
            new_pts = pts.copy()
            
            for i in range(N):
                # Find facets containing point i
                mask = np.any(faces == i, axis=1)
                if not np.any(mask):
                    continue
                
                facet_indices = np.where(mask)[0]
                # Find the facet with the largest radius
                max_idx = facet_indices[np.argmax(radii[facet_indices])]
                
                # Move towards the facet center
                center = centers[max_idx]
                # Ensure center is on the sphere
                center_norm = np.linalg.norm(center)
                if center_norm < 1e-10:
                    continue
                center = center / center_norm
                
                # Interpolate
                new_pts[i] = (1 - alpha) * pts[i] + alpha * center
                new_pts[i] /= np.linalg.norm(new_pts[i])
            
            # Check if points are distinct
            new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
            
            # Verify improvement
            old_rad = compute_covering_radius_exact(pts)
            new_rad = compute_covering_radius_exact(new_pts)
            
            if new_rad < old_rad:
                pts = new_pts
            else:
                # If no improvement, reduce alpha
                alpha *= 0.5
                if alpha < 0.01:
                    break
        
        return pts

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

    def facet_objective_and_grad(x_flat):
        """
        Objective: soft-max of facet circumradii (covering radius approximation).
        Uses log-sum-exp for numerical stability.
        """
        pts = x_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        pts_norm = pts / norms
        
        faces, centers, radii = get_facets(pts_norm)
        if faces is None:
            return 1.0, np.zeros_like(x_flat)
        
        # Soft-max of radii
        T = 20.0
        max_r = np.max(radii)
        shifted = radii - max_r
        exp_shifted = np.exp(shifted * T)
        sum_exp = np.sum(exp_shifted)
        val = max_r + np.log(sum_exp) / T
        
        # Gradient: weighted average of gradients of radii
        w = exp_shifted / sum_exp  # weights for each facet
        
        # Gradient of radii w.r.t pts
        # This is complex; use finite differences or approximate
        # For simplicity, use a gradient-free approach in the main loop
        # and use this only for value evaluation
        
        return float(val), np.zeros_like(x_flat)

    def local_search(pts, max_rounds=10, step_sizes=None):
        """Local search: try moving each point in random directions."""
        if step_sizes is None:
            step_sizes = [0.05, 0.02, 0.01, 0.005]
        
        improved = True
        round_count = 0
        best_rad = compute_covering_radius_exact(pts)
        
        while improved and round_count < max_rounds and time.time() < deadline - 5.0:
            improved = False
            round_count += 1
            
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                
                old_p = pts[i].copy()
                rng = np.random.RandomState(i * 1000 + round_count * 100)
                
                for trial in range(6):
                    direction = rng.randn(3)
                    direction -= np.dot(direction, old_p) * old_p
                    dir_norm = np.linalg.norm(direction)
                    if dir_norm < 1e-10:
                        continue
                    direction /= dir_norm
                    
                    for step in step_sizes:
                        new_p = old_p + step * direction
                        new_p /= np.linalg.norm(new_p)
                        pts[i] = new_p
                        
                        rad = compute_covering_radius_exact(pts)
                        if rad < best_rad - 1e-10:
                            best_rad = rad
                            improved = True
                            old_p = new_p.copy()
                            break
                        else:
                            pts[i] = old_p
                    else:
                        continue
                    break
        
        return pts, best_rad

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple starts with repulsion + CVT
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 60.0:
            break
        
        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            rng = np.random.RandomState(42 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(100 + restart)
            points = points + 0.3 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            rng = np.random.RandomState(200 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is not None:
                rng = np.random.RandomState(500 + restart)
                points = best_points + 0.05 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        # Step 1: Repulsion to get good separation
        points = gradient_descent_repulsion(points, 3.0, 0.08, 0.985, 200)
        
        # Step 2: CVT optimization
        points = cvt_step(points, n_iter=15, alpha=0.6)
        
        # Step 3: Local search
        points, rad = local_search(points, max_rounds=5)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Refinement with more CVT iterations
    if best_points is not None and time.time() < deadline - 20.0:
        for trial in range(3):
            if time.time() > deadline - 15.0:
                break
            
            points = best_points.copy()
            points = cvt_step(points, n_iter=20, alpha=0.4)
            points, rad = local_search(points, max_rounds=8, step_sizes=[0.02, 0.01, 0.005, 0.002])
            
            if rad < best_rad:
                best_rad = rad
                best_points = points.copy()

    # Phase 3: Final polish
    if best_points is not None and time.time() < deadline - 5.0:
        final = best_points.copy()
        final = cvt_step(final, n_iter=10, alpha=0.3)
        final, rad = local_search(final, max_rounds=5, step_sizes=[0.01, 0.005, 0.002, 0.001])
        
        if rad < best_rad:
            best_rad = rad
            best_points = final.copy()

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    # Ensure all points are on the unit sphere
    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)

    return best_points

# EVOLVE-BLOCK-END
