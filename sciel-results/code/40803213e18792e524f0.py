"""Program for covering the sphere with 130 equal caps."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for safety

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius_hull(pts):
        """Compute exact covering radius via convex hull facets."""
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
            return np.inf

    def make_grid(n_theta=60, n_phi=120):
        """Create a dense grid of points on the unit sphere."""
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        T, P = np.meshgrid(theta, phi, indexing='ij')
        x = np.sin(T) * np.cos(P)
        y = np.sin(T) * np.sin(P)
        z = np.cos(T)
        return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    grid_points = make_grid(60, 120)  # ~7200 points

    def exact_covering_radius_gradient(pts, n_samples=5000):
        """
        Compute a gradient estimate for the covering radius using Monte Carlo sampling.
        The covering radius is max_y min_j d(y, p_j).
        We estimate the gradient by sampling points y and finding which p_j is closest.
        """
        rng = np.random.RandomState(42)
        # Sample random points on the sphere
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        
        # For each sample, find nearest point and compute distance
        dots = samples @ pts.T  # (n_samples, N)
        dots = np.clip(dots, -1.0, 1.1)
        dists = np.arccos(dots)  # (n_samples, N)
        
        # Find nearest point for each sample
        min_dists = np.min(dists, axis=1)  # (n_samples,)
        nearest_idx = np.argmin(dists, axis=1)  # (n_samples,)
        
        # The covering radius is approximately max(min_dists)
        # For gradient, we need to account for which points are "active" (closest to some sample)
        # Use soft assignment: weight each point by how often it's nearest, weighted by distance
        weights = np.zeros(N)
        for i in range(n_samples):
            weights[nearest_idx[i]] += min_dists[i]
        
        # Normalize weights
        total_w = np.sum(weights)
        if total_w < 1e-10:
            return min_dists.max(), np.zeros(N * 3)
        weights /= total_w
        
        # Gradient: for each point p_j, the gradient of d(y, p_j) w.r.t. p_j is:
        # d/dp_j arccos(y·p_j) = -y / sqrt(1 - (y·p_j)^2)
        # But we only care about points that are "active" (nearest to some sample)
        # Approximate gradient: sum over samples where p_j is nearest of -y / sqrt(1-(y·p_j)^2)
        
        grad = np.zeros((N, 3))
        for i in range(n_samples):
            j = nearest_idx[i]
            y = samples[i]
            p_j = pts[j]
            dot = np.dot(y, p_j)
            dot = np.clip(dot, -1.0, 1.0)
            sqrt_term = np.sqrt(max(1.0 - dot * dot, 1e-12))
            # Gradient of arccos(y·p) w.r.t. p is -y/sqrt(1-(y·p)^2)
            # But we want to MINIMIZE the max distance, so we want to move p_j to REDUCE distance
            # which means moving p_j TOWARD y
            # So gradient direction for p_j is +y/sqrt(1-(y·p)^2)
            grad[j] += y / sqrt_term
        
        # Project to tangent space
        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        return float(min_dists.max()), grad_tangent.ravel()

    def repulsion_force(points, exponent=1.5):
        """Compute repulsion forces between all point pairs."""
        dot = np.dot(points, points.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        w = 1.0 / (dist_sq ** exponent)
        np.fill_diagonal(w, 0.0)
        sum_w = np.sum(w, axis=1)
        sum_p_w = w @ points
        forces = points * sum_w[:, np.newaxis] - sum_p_w
        return forces

    def gradient_descent(points, alpha_init, decay, max_iter, exponent=1.5):
        """Optimize points via gradient descent on the repulsion energy."""
        alpha = alpha_init
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            forces = repulsion_force(points, exponent)
            points = points + alpha * forces
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            alpha *= decay
        return points

    def targeted_refinement(points, max_iters=200):
        """Refine by targeting the worst facet: move its 3 vertices to reduce circumradius."""
        pts = points.copy()
        step = 0.01
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
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
                
                max_idx = np.argmax(radii)
                rad = radii[max_idx]
                face = faces[max_idx]
            except Exception:
                break
            
            i, j, k = face
            a = pts[i].copy()
            b = pts[j].copy()
            c = pts[k].copy()
            
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            c_norm = np.linalg.norm(c_dir)
            
            if c_norm < 1e-10:
                mid = (a + b + c) / 3.0
                mid /= np.linalg.norm(mid)
                for idx, p in [(i, a), (j, b), (k, c)]:
                    direction = p - 0.1 * mid
                    direction /= np.linalg.norm(direction)
                    pts[idx] = direction
                continue
            
            c_dir /= c_norm
            
            # Try multiple step sizes, keep the best
            best_new_pts = None
            best_new_rad = rad
            for s in [step, step * 0.5, step * 2.0]:
                trial_pts = pts.copy()
                for idx, p in [(i, a), (j, b), (k, c)]:
                    tangent = c_dir - np.dot(c_dir, p) * p
                    t_norm = np.linalg.norm(tangent)
                    if t_norm > 1e-10:
                        tangent /= t_norm
                        new_p = p + s * tangent
                        new_p /= np.linalg.norm(new_p)
                        trial_pts[idx] = new_p
                
                new_rad = compute_covering_radius_hull(trial_pts)
                if new_rad < best_new_rad:
                    best_new_rad = new_rad
                    best_new_pts = trial_pts.copy()
            
            if best_new_pts is not None:
                pts = best_new_pts
                step = min(step * 1.2, 0.05)
            else:
                step *= 0.5
                if step < 1e-6:
                    break
        
        return pts

    def swap_refinement(points, max_iters=50):
        """Try swapping points that are close to each other with points near the worst facet."""
        pts = points.copy()
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
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
                
                max_idx = np.argmax(radii)
                rad = radii[max_idx]
                face = faces[max_idx]
            except Exception:
                break
            
            i, j, k = face
            # Find the point closest to the centroid of the worst face
            centroid = (pts[i] + pts[j] + pts[k]) / 3.0
            centroid /= np.linalg.norm(centroid)
            
            dots = np.sum(pts * centroid, axis=1)
            # Find points far from the face (potential candidates to move into the face region)
            # Actually, we want to find a point that is "redundant" (close to another point) 
            # and move it to the worst face region
            # For simplicity, try moving one of the face vertices to a better position
            # by projecting the centroid onto the sphere and adjusting
            
            # Alternative: try to find a point that is very close to another point
            # and replace it with a point in the direction of the worst facet centroid
            dot_matrix = np.dot(pts, pts.T)
            np.fill_diagonal(dot_matrix, -1.0)
            
            # Find the closest pair
            max_dot_idx = np.unravel_index(np.argmax(dot_matrix), dot_matrix.shape)
            p1, p2 = max_dot_idx
            if dot_matrix[p1, p2] > 0.99:  # Very close points
                # Move p2 to the direction of the worst facet centroid
                new_pos = centroid + 0.1 * np.random.randn(3)
                new_pos /= np.linalg.norm(new_pos)
                old_p2 = pts[p2].copy()
                pts[p2] = new_pos
                new_rad = compute_covering_radius_hull(pts)
                if new_rad < rad:
                    rad = new_rad
                else:
                    pts[p2] = old_p2
                    break
        
        return pts

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple starts with repulsion-based gradient descent (fast initialization)
    n_restarts = 5
    exponents = [1.0, 1.5, 2.0, 2.5, 1.2]
    
    for restart in range(n_restarts):
        if time.time() > deadline - 70.0:
            break
        
        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            rng = np.random.RandomState(42)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(100)
            points = points + 0.3 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            if best_points is not None:
                rng = np.random.RandomState(300)
                points = best_points + 0.1 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)
        else:
            # Use a different perturbation strategy
            if best_points is not None:
                rng = np.random.RandomState(500)
                points = best_points.copy()
                # Perturb only half the points
                half = N // 2
                points[:half] += 0.05 * rng.randn(half, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        exp = exponents[restart % len(exponents)]
        points = gradient_descent(points, 0.05, 0.995, 150, exp)
        rad = compute_covering_radius_hull(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Gradient descent using Monte Carlo gradient estimate
    if best_points is not None and time.time() < deadline - 40.0:
        alpha = 0.02
        for it in range(100):
            if time.time() > deadline - 15.0:
                break
            
            obj_val, grad = exact_covering_radius_gradient(best_points, n_samples=3000)
            
            if np.linalg.norm(grad) < 1e-10:
                break
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                grad = grad / grad_norm
            
            # Update
            new_pts = best_points + alpha * grad.reshape(N, 3)
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            new_pts = new_pts / norms
            
            new_rad = compute_covering_radius_hull(new_pts)
            if new_rad < best_rad:
                best_rad = new_rad
                best_points = new_pts
                alpha = min(alpha * 1.1, 0.05)
            else:
                alpha *= 0.8
        
        # Final L-BFGS-B polish with soft-max for high precision
        if time.time() < deadline - 10.0:
            beta = 100.0
            x0 = best_points.ravel()
            
            def objective_wrapper(x):
                val, _ = soft_max_objective(x, beta=beta)
                return val
            
            def gradient_wrapper(x):
                _, grad = soft_max_objective(x, beta=beta)
                return grad
            
            try:
                result = minimize(
                    objective_wrapper,
                    x0,
                    jac=gradient_wrapper,
                    method='L-BFGS-B',
                    options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 60}
                )
                
                if result.success or result.fun < objective_wrapper(x0):
                    candidate = result.x.reshape(N, 3)
                    norms = np.linalg.norm(candidate, axis=1, keepdims=True)
                    norms[norms < 1e-10] = 1.0
                    candidate = candidate / norms
                    rad = compute_covering_radius_hull(candidate)
                    if rad < best_rad:
                        best_rad = rad
                        best_points = candidate.copy()
            except Exception:
                pass

    # Phase 3: Targeted refinement on the worst facet
    if best_points is not None and time.time() < deadline - 15.0:
        refined = targeted_refinement(best_points.copy(), max_iters=200)
        rad = compute_covering_radius_hull(refined)
        if rad < best_rad:
            best_rad = rad
            best_points = refined

    # Phase 4: Additional targeted refinement passes
    if best_points is not None and time.time() < deadline - 10.0:
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            refined = targeted_refinement(best_points.copy(), max_iters=100)
            rad = compute_covering_radius_hull(refined)
            if rad < best_rad:
                best_rad = rad
                best_points = refined

    # Phase 5: Final gradient descent polish
    if best_points is not None and time.time() < deadline - 3.0:
        alpha = 0.01
        for it in range(50):
            if time.time() > deadline - 1.0:
                break
            
            obj_val, grad = exact_covering_radius_gradient(best_points, n_samples=2000)
            
            if np.linalg.norm(grad) < 1e-10:
                break
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                grad = grad / grad_norm
            
            new_pts = best_points + alpha * grad.reshape(N, 3)
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            new_pts = new_pts / norms
            
            new_rad = compute_covering_radius_hull(new_pts)
            if new_rad < best_rad:
                best_rad = new_rad
                best_points = new_pts
                alpha = min(alpha * 1.1, 0.03)
            else:
                alpha *= 0.7

    # Final check
    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
