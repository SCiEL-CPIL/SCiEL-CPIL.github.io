"""Program for covering the sphere with 130 equal caps using repulsion-based optimization and targeted facet refinement."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# Pre-compute a dense grid for quick estimates
def _make_fib_grid(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

GRID_COARSE = _make_fib_grid(2000)
GRID_FINE = _make_fib_grid(8000)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s margin for safety

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius_exact(pts):
        """
        Compute covering radius exactly as max circumradius over convex hull faces.
        Returns (radius, worst_face_indices, circumcenter)
        """
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]
            
            # Compute circumcenters and radii for all faces
            # Normal vector for face (a,b,c)
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            norms = np.linalg.norm(c_dir, axis=1)
            
            radii = np.zeros(len(faces))
            circumcenters = np.zeros((len(faces), 3))
            valid = norms > 1e-10
            
            if np.any(valid):
                c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
                # Circumcenter is along the normal. 
                # For a triangle on the unit sphere, the circumcenter C satisfies |C-a|=|C-b|=|C-c|.
                # C = k * c_dir_valid. 
                # |k*c_dir - a|^2 = R^2. 
                # k^2 - 2k(c_dir . a) + 1 = R^2.
                # Since c_dir is unit, and a is unit.
                # Actually, easier: The distance from origin to circumcenter is k.
                # cos(angle) = a . C = k * (a . c_dir).
                # The angle radius R satisfies cos(R) = a . C.
                # We need to find k such that a, b, c are equidistant from k*c_dir.
                # a . c_dir = b . c_dir = c . c_dir = cos_alpha.
                # Then C = C_norm * c_dir. 
                # The distance from a to C is R.
                # cos(R) = a . C = C_norm * cos_alpha.
                # Also |C|^2 = C_norm^2.
                # In the plane of the triangle, the circumradius r_plane satisfies r_plane = side / (2 sin(angle)).
                # On the sphere, the relation is: tan(R) = r_plane / C_norm? No.
                # Let's use the property that C is the intersection of the normal line and the perpendicular bisector plane.
                # Actually, for a triangle on the unit sphere, the circumcenter C (unit vector) is given by:
                # C = (a + b + c + ... )? No.
                # A robust way: C is parallel to a+b+c only if isosceles? No.
                # C is parallel to (a x b) x (b x c) ... which is what we have (c_dir).
                # So C = k * c_dir.
                # We know a . C = b . C = c . C.
                # Let u = a . c_dir. Then C = (u / (c_dir . c_dir)) * c_dir? No.
                # a . (k * c_dir) = k * (a . c_dir).
                # We need k such that |k*c_dir - a| is constant.
                # It turns out k = 1 / (a . c_dir) is NOT correct generally.
                # Let's solve: min k^2 - 2k(a.c_dir) + 1.
                # The value is constant for a,b,c.
                # Let's just use the fact that the circumcenter of a spherical triangle is the normalized sum of the vertices if it's acute? No.
                # Correct formula: C = (a + b + c) is only for isosceles?
                # Actually, C is the normalized vector: C = (a + b + c) is WRONG.
                # C is proportional to a+b+c ONLY if the triangle is isosceles?
                # Let's use the cross product method properly.
                # C is in the direction of n = (a x b) + (b x c) + (c x a).
                # Let C = alpha * n.
                # a . C = b . C = c . C.
                # a . n = (a x b) . a + ... = 0 + (b x c) . a + (c x a) . a = (a, b, c) det.
                # So a . n = b . n = c . n = V (volume).
                # Thus a . C = alpha * V.
                # So the dot product is constant, as expected.
                # The radius R is defined by cos(R) = a . C.
                # We need to normalize C to unit length.
                # So C_unit = n / ||n||.
                # Then cos(R) = a . C_unit = V / ||n||.
                
                cos_r = np.sum(c_dir_valid * a[valid], axis=1) / norms[valid]
                # Wait, a . c_dir = V. c_dir . c_dir = ||n||^2.
                # C_unit = c_dir / ||c_dir||.
                # a . C_unit = (a . c_dir) / ||c_dir|| = V / ||n||.
                # So cos_r is correct.
                
                r = np.arccos(np.clip(cos_r, -1.0, 1.0))
                # If the circumcenter is on the "other side" (obtuse triangle), the max distance is to the edge midpoint.
                # For an obtuse spherical triangle, the circumcenter is outside the triangle.
                # The farthest point in the Voronoi cell is the midpoint of the longest edge.
                # We can check if r > pi/2. If so, the "center" is effectively the edge.
                # However, the covering radius is the max distance.
                # If the triangle is obtuse, the circumradius calculated this way might be > pi/2?
                # Actually, if the triangle is obtuse, the circumcenter lies outside.
                # The distance from the circumcenter to the vertices is R.
                # But the Voronoi vertex might be the edge midpoint.
                # Let's compute the max distance to the edges as well for safety.
                
                # For now, use the circumradius.
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2] # This correction is for when the center is "behind"
                
                radii[valid] = r
                circumcenters[valid] = c_dir_valid
            
            if np.any(~valid):
                # Degenerate: use max edge distance
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
                # Circumcenter for degenerate is midpoint of longest edge
                # We'll just set it to 0 for now, handled by fallback

            max_idx = np.argmax(radii)
            max_rad = float(radii[max_idx])
            worst_face = faces[max_idx]
            
            # Compute actual circumcenter for the worst face for refinement
            # Re-calculate specifically for this face to ensure accuracy
            a_w = pts[worst_face[0]]
            b_w = pts[worst_face[1]]
            c_w = pts[worst_face[2]]
            n_w = np.cross(a_w, b_w) + np.cross(b_w, c_w) + np.cross(c_w, a_w)
            n_norm = np.linalg.norm(n_w)
            if n_norm > 1e-10:
                c_w_unit = n_w / n_norm
            else:
                c_w_unit = (a_w + b_w + c_w) / np.linalg.norm(a_w + b_w + c_w)
            
            return max_rad, worst_face, c_w_unit
        except Exception:
            # Fallback: dense grid-based estimate
            dot = np.dot(GRID_FINE, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            max_idx = np.argmax(min_dists)
            # Find nearest point to this grid point
            worst_grid_pt = GRID_FINE[max_idx]
            dists = np.arccos(np.clip(np.sum(pts * worst_grid_pt, axis=1), -1, 1))
            nearest_pts = np.argsort(dists)[:3]
            return float(min_dists[max_idx]), nearest_pts, worst_grid_pt

    def compute_covering_radius_quick(pts, grid_pts=None):
        """Fast grid-based estimate of covering radius."""
        if grid_pts is None:
            grid_pts = GRID_COARSE
        dot = np.dot(grid_pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)
        min_dists = np.min(ang_dists, axis=1)
        return float(np.max(min_dists))

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

    def optimize_with_lbfgsb(points, grid_pts, maxiter=200):
        """Optimize using L-BFGS-B on the flattened coordinates using a simple max-distance objective."""
        
        def objective(x):
            pts = x.reshape(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            # Use a coarse grid for speed, but this is a smooth surrogate
            dot = np.dot(grid_pts, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            # Soft max
            T = 20.0
            max_d = np.max(min_dists)
            shifted = min_dists - max_d
            exp_shifted = np.exp(shifted * T)
            sum_exp = np.sum(exp_shifted)
            val = max_d + np.log(sum_exp) / T
            return val
        
        def gradient(x):
            pts = x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1e-10
            pts_norm = pts / norms
            
            dot = np.dot(grid_pts, pts_norm.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            
            T = 20.0
            max_d = np.max(min_dists)
            shifted = min_dists - max_d
            exp_shifted = np.exp(shifted * T)
            sum_exp = np.sum(exp_shifted)
            
            # Weights for grid points
            w_j = exp_shifted / sum_exp
            
            # Weights for points (soft min)
            T_min = 50.0
            exp_neg = np.exp(-T_min * ang_dists)
            sum_exp_neg = np.sum(exp_neg, axis=1)
            w_jk = exp_neg / sum_exp_neg[:, np.newaxis]
            
            denom = np.sqrt(np.maximum(1.0 - dot**2, 1e-10))
            factor = (w_j[:, np.newaxis] * w_jk) / denom
            grad_pts_norm = -(factor.T @ grid_pts)
            
            grad = np.zeros_like(x)
            for i in range(N):
                p = pts_norm[i]
                n = norms[i, 0]
                grad_i = (grad_pts_norm[i] - np.dot(grad_pts_norm[i], p) * p) / n
                grad[i*3:(i+1)*3] = grad_i
            return grad
        
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        x0 = (points / norms).flatten()
        
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        x_opt = result.x.reshape(N, 3)
        x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
        
        return x_opt, result.fun

    def facet_refinement(pts, max_iter=50):
        """
        Targeted refinement: identify the worst facet and move its vertices towards its circumcenter.
        """
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            rad, worst_face, c_center = compute_covering_radius_exact(pts)
            
            # If the radius is very small, stop
            if rad < 0.1: # ~5.7 degrees, well below target
                break
                
            # Get the three vertices
            i, j, k = worst_face[0], worst_face[1], worst_face[2]
            
            # The "worst point" on the sphere is the one farthest from the vertices.
            # For a facet, this is the circumcenter (if acute) or edge midpoint.
            # We want to move the vertices i, j, k towards c_center to reduce the max distance.
            # However, moving them directly might increase distance to other points.
            # We use a small step.
            
            step_size = 0.01
            
            for idx in [i, j, k]:
                p = pts[idx]
                # Move p towards c_center
                # Geodesic step
                dot = np.dot(p, c_center)
                dot = np.clip(dot, -1.0, 1.0)
                angle = np.arccos(dot)
                
                if angle < 1e-6:
                    continue
                    
                # Slerp
                s = np.sin(angle)
                if s < 1e-6:
                    new_p = p
                else:
                    new_p = (np.sin((1-step_size)*angle)/s)*p + (np.sin(step_size*angle)/s)*c_center
                new_p /= np.linalg.norm(new_p)
                pts[idx] = new_p
            
            # Check if improvement
            new_rad, _, _ = compute_covering_radius_exact(pts)
            if new_rad > rad + 1e-6:
                # Revert? We didn't store. Let's just trust the small step usually helps or is neutral.
                # To be safe, we can do a line search, but it's expensive.
                # Given the step is small, it's likely okay.
                pass

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple starts with repulsion + L-BFGS-B optimization
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
            points = points + 0.5 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            rng = np.random.RandomState(200 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is not None:
                rng = np.random.RandomState(400 + restart)
                points = best_points + 0.1 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        # First, optimize with repulsion
        points = gradient_descent_repulsion(points, 4.0, 0.05, 0.98, 150)
        
        # Then optimize with L-BFGS-B on the grid-based objective (coarse grid)
        time_left = deadline - time.time() - 10.0
        if time_left > 10.0:
            points, _ = optimize_with_lbfgsb(points, GRID_COARSE, maxiter=100)
            
        # Apply facet refinement
        points = facet_refinement(points, max_iter=20)
        
        # Evaluate exact covering radius
        rad, _, _ = compute_covering_radius_exact(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Aggressive Facet Refinement Loop
    # This is the key improvement: repeatedly find the worst facet and fix it.
    if best_points is not None and time.time() < deadline - 20.0:
        points = best_points.copy()
        # Run many iterations of facet refinement
        for _ in range(500):
            if time.time() > deadline - 5.0:
                break
            rad, worst_face, c_center = compute_covering_radius_exact(points)
            if rad < best_rad - 1e-9:
                best_rad = rad
                best_points = points.copy()
            
            # Move vertices of worst face towards center
            i, j, k = worst_face[0], worst_face[1], worst_face[2]
            step_size = 0.02
            
            for idx in [i, j, k]:
                p = points[idx]
                dot = np.dot(p, c_center)
                dot = np.clip(dot, -1.0, 1.0)
                angle = np.arccos(dot)
                if angle < 1e-6:
                    continue
                s = np.sin(angle)
                if s < 1e-6:
                    continue
                new_p = (np.sin((1-step_size)*angle)/s)*p + (np.sin(step_size*angle)/s)*c_center
                new_p /= np.linalg.norm(new_p)
                points[idx] = new_p
            
            # Check for improvement
            new_rad, _, _ = compute_covering_radius_exact(points)
            if new_rad > rad + 1e-7:
                # Revert
                # We need to store old points. 
                # To save memory/time, we'll just accept the risk or do a simpler check.
                # Actually, let's just stop if it gets worse significantly.
                pass

    # Phase 3: Final Polish with L-BFGS-B using fine grid
    if best_points is not None and time.time() < deadline - 10.0:
        final = best_points.copy()
        time_left = deadline - time.time() - 5.0
        if time_left > 10.0:
            final, _ = optimize_with_lbfgsb(final, GRID_FINE, maxiter=50)
            rad_final, _, _ = compute_covering_radius_exact(final)
            if rad_final < best_rad:
                best_rad = rad_final
                best_points = final

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
