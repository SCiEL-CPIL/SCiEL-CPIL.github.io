"""Program for covering the sphere with 130 equal caps."""
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

    def compute_covering_radius(pts):
        """Compute covering radius as max circumradius over convex hull faces."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices  # shape (F, 3)
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
                # Circumradius of a spherical triangle is the angle from the 
                # circumcenter to the vertices. If the center is inside, r is 
                # arccos(cos_r). If outside, it might be pi - arccos(cos_r).
                # For small triangles (which we expect in a good covering), 
                # the center is "close" and r < pi/2 usually.
                # However, the circumradius of the *cap* covering the facet is 
                # the distance from the circumcenter to the vertices.
                # The formula r = arccos(cos_r) gives the angle if cos_r > 0.
                # If the triangle is obtuse, the circumcenter is outside.
                # The covering radius of the facet is the max distance from 
                # circumcenter to vertices.
                # For a spherical triangle, the circumradius R satisfies:
                # cos R = (cos a + cos b + cos c - 1 - 2 cos a cos b cos c) / (4 sin A sin B sin C) ... complex.
                # Simpler: The circumcenter C is the intersection of bisectors.
                # For small triangles, C is close to the centroid.
                # The distance from C to A is R.
                # cos R = C . A.
                # The code above calculates C direction.
                # r = arccos(C . A).
                # This is the correct geometric circumradius.
                radii[valid] = r
            
            if np.any(~valid):
                # Degenerate or flat triangles
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                # Fallback to max edge length / 2 or similar? 
                # Actually, if degenerate, the "radius" is likely half the longest edge?
                # Or just the max distance between points?
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca) / 2.0
            
            return float(np.max(radii))
        except Exception:
            # Fallback: grid-based estimate
            lat = np.linspace(-np.pi/2, np.pi/2, 50)
            lon = np.linspace(0, 2*np.pi, 100, endpoint=False)
            LA, LO = np.meshgrid(lat, lon, indexing='ij')
            z = np.sin(LA)
            x = np.cos(LA) * np.cos(LO)
            y = np.cos(LA) * np.sin(LO)
            grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            dot = np.dot(grid_pts, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    # Precompute a dense spherical grid for smooth covering radius
    # Using a slightly denser grid than before for better resolution
    n_lat = 60
    n_lon = 120
    lat = np.linspace(-np.pi/2, np.pi/2, n_lat)
    lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    LA, LO = np.meshgrid(lat, lon, indexing='ij')
    grid_z = np.sin(LA)
    grid_x = np.cos(LA) * np.cos(LO)
    grid_y = np.cos(LA) * np.sin(LO)
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    n_grid = len(grid_pts)

    def soft_max_objective_and_grad(pts_flat, beta=100.0):
        """
        Vectorized soft-max objective and gradient.
        """
        pts = pts_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms

        # Compute angular distances: (n_grid, N)
        dot = np.dot(grid_pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)  # (n_grid, N)

        # Min distance for each grid point
        min_dists = np.min(ang_dists, axis=1)  # (n_grid,)
        nearest_idx = np.argmin(ang_dists, axis=1)  # (n_grid,)

        # Soft-max
        m = np.max(min_dists)
        if m < 1e-15:
            return 0.0, np.zeros(3 * N)
        
        shifted = min_dists - m
        exp_vals = np.exp(beta * shifted)
        sum_exp = np.sum(exp_vals)
        obj = m + np.log(sum_exp) / beta
        
        weights = exp_vals / sum_exp  # (n_grid,)

        # Gradient calculation
        # d_obj / d_pts_j = sum_{i: nearest(i)=j} w_i * grad_{p_j} d_{ij}
        # grad_{p_j} d_{ij} = - (g_i - (g_i.p_j) p_j) / sqrt(1 - (g_i.p_j)^2)
        
        grad = np.zeros_like(pts)  # (N, 3)
        
        # Only compute for grid points with significant weight
        mask = weights > 1e-8
        if not np.any(mask):
            return obj, grad.ravel()
            
        idxs = nearest_idx[mask]
        ws = weights[mask]
        gs = grid_pts[mask]
        ps = pts[idxs]
        
        u = np.sum(gs * ps, axis=1)  # (num_valid,)
        u = np.clip(u, -1.0 + 1e-12, 1.0 - 1e-12)
        
        g_perp = gs - u[:, np.newaxis] * ps  # (num_valid, 3)
        denom = np.sqrt(1.0 - u * u)
        denom[denom < 1e-12] = 1e-12
        
        grads_i = -g_perp / denom[:, np.newaxis]  # (num_valid, 3)
        
        # Weighted sum
        weighted_grads = grads_i * ws[:, np.newaxis]
        
        # Scatter back to grad
        # We need to sum gradients for each point index
        for k in range(len(idxs)):
            grad[idxs[k]] += weighted_grads[k]

        # Project to tangent space
        grad_dot_p = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tan = grad - grad_dot_p * pts
        
        # Scale for raw coordinates (since we normalized pts inside)
        raw_norms = np.linalg.norm(pts_flat.reshape(N, 3), axis=1, keepdims=True)
        raw_norms[raw_norms < 1e-15] = 1.0
        grad_flat = (grad_tan / raw_norms).ravel()

        return obj, grad_flat

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

    best_points = None
    best_rad = np.inf

    # Phase 1: Initialization and Coarse Optimization
    # Use Fibonacci spiral as primary start, add random restarts
    starts = [fibonacci_spiral(N).astype(np.float64)]
    rng = np.random.RandomState(42)
    for _ in range(3):
        pts = rng.randn(N, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        starts.append(pts)
    
    # Also add perturbed fibonacci
    pts_pert = starts[0] + 0.2 * rng.randn(N, 3)
    pts_pert /= np.linalg.norm(pts_pert, axis=1, keepdims=True)
    starts.append(pts_pert)

    for start_pts in starts:
        if time.time() > deadline - 40.0:
            break
        # Coarse repulsion gradient descent
        pts = gradient_descent(start_pts, 0.05, 0.995, 200, 1.5)
        rad = compute_covering_radius(pts)
        if rad < best_rad:
            best_rad = rad
            best_points = pts.copy()

    if best_points is None:
        best_points = starts[0]
        best_rad = compute_covering_radius(best_points)

    # Phase 2: L-BFGS-B with Vectorized Soft-Max
    # Use a moderate beta to start
    x0 = best_points.ravel()
    
    def obj_grad_l1(x):
        if time.time() > deadline - 10.0:
            return 1e6, np.zeros_like(x)
        return soft_max_objective_and_grad(x, beta=80.0)

    try:
        result = minimize(
            obj_grad_l1,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
        )
        opt_pts = result.x.reshape(N, 3)
        opt_pts /= np.linalg.norm(opt_pts, axis=1, keepdims=True)
        rad_opt = compute_covering_radius(opt_pts)
        if rad_opt < best_rad:
            best_rad = rad_opt
            best_points = opt_pts.copy()
    except Exception:
        pass

    # Phase 3: High-Precision Polish with Higher Beta
    if time.time() < deadline - 15.0:
        x0 = best_points.ravel()
        
        def obj_grad_l2(x):
            if time.time() > deadline - 5.0:
                return 1e6, np.zeros_like(x)
            return soft_max_objective_and_grad(x, beta=150.0)

        try:
            result = minimize(
                obj_grad_l2,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
            )
            opt_pts = result.x.reshape(N, 3)
            opt_pts /= np.linalg.norm(opt_pts, axis=1, keepdims=True)
            rad_opt = compute_covering_radius(opt_pts)
            if rad_opt < best_rad:
                best_rad = rad_opt
                best_points = opt_pts.copy()
        except Exception:
            pass

    # Phase 4: Local Search on Worst Facets
    # Identify the worst facet and try to move its vertices to reduce the circumradius
    if time.time() < deadline - 5.0:
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            pts = best_points.copy()
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
                    radii[valid] = r
                if np.any(~valid):
                    radii[~valid] = np.pi / 4 # Large penalty
                
                worst_idx = np.argmax(radii)
                if radii[worst_idx] < best_rad - 1e-8:
                    continue # Consistency check failed?
                
                # Get the 3 points of the worst facet
                i, j, k = faces[worst_idx]
                
                # Try moving them slightly towards the circumcenter?
                # Or just random small perturbations and check
                improved = False
                for trial in range(20):
                    if time.time() > deadline - 2.0:
                        break
                    cand = pts.copy()
                    # Perturb the 3 points
                    pert = 0.01 * rng.randn(3, 3)
                    cand[i] += pert[0]
                    cand[j] += pert[1]
                    cand[k] += pert[2]
                    cand[i] /= np.linalg.norm(cand[i])
                    cand[j] /= np.linalg.norm(cand[j])
                    cand[k] /= np.linalg.norm(cand[k])
                    
                    rad_cand = compute_covering_radius(cand)
                    if rad_cand < best_rad - 1e-10:
                        best_rad = rad_cand
                        best_points = cand
                        improved = True
                        break
            except Exception:
                break

    return best_points

# EVOLVE-BLOCK-END
