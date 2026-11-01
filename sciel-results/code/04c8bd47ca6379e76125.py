"""Program for covering the sphere with 130 equal caps."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

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

    # Precompute a dense spherical grid
    n_lat = 120
    n_lon = 240
    lat = np.linspace(-np.pi/2, np.pi/2, n_lat)
    lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    LA, LO = np.meshgrid(lat, lon, indexing='ij')
    grid_z = np.sin(LA)
    grid_x = np.cos(LA) * np.cos(LO)
    grid_y = np.cos(LA) * np.sin(LO)
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    n_grid = len(grid_pts)
    
    # Also create a coarser grid for faster Lloyd iterations
    n_lat_c = 60
    n_lon_c = 120
    lat_c = np.linspace(-np.pi/2, np.pi/2, n_lat_c)
    lon_c = np.linspace(0, 2*np.pi, n_lon_c, endpoint=False)
    LA_c, LO_c = np.meshgrid(lat_c, lon_c, indexing='ij')
    grid_z_c = np.sin(LA_c)
    grid_x_c = np.cos(LA_c) * np.cos(LO_c)
    grid_y_c = np.cos(LA_c) * np.sin(LO_c)
    grid_pts_c = np.column_stack([grid_x_c.ravel(), grid_y_c.ravel(), grid_z_c.ravel()])

    def lloyd_iteration(points, n_iters=50, use_coarse=False):
        """Lloyd-like refinement: move each point to the centroid of its Voronoi cell on the sphere."""
        gpts = grid_pts_c if use_coarse else grid_pts
        for _ in range(n_iters):
            if time.time() > deadline - 10.0:
                break
            
            dot = np.dot(gpts, points.T)
            dot = np.clip(dot, -1.0, 1.0)
            nearest_idx = np.argmax(dot, axis=1)
            
            new_points = np.zeros_like(points)
            for i in range(N):
                mask = nearest_idx == i
                if np.any(mask):
                    cell_pts = gpts[mask]
                    d = np.clip(1.0 - dot[mask, i], 1e-10, None)
                    w = 1.0 / d
                    centroid = np.sum(cell_pts * w[:, np.newaxis], axis=0)
                    norm = np.linalg.norm(centroid)
                    if norm > 1e-10:
                        new_points[i] = centroid / norm
                    else:
                        new_points[i] = points[i]
                else:
                    new_points[i] = points[i]
            
            alpha = 0.5
            points = (1 - alpha) * points + alpha * new_points
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        return points

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
    
    def soft_max_objective_facets(pts_flat, beta=100.0):
        """
        Soft-max surrogate based on facet circumradii instead of grid.
        This directly targets the covering radius definition.
        """
        pts = pts_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms
        
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
        except Exception:
            # Fallback to grid-based
            val, grad = soft_max_objective_fast(pts_flat, beta=beta)
            return val, grad
        
        a = pts[faces[:, 0]]
        b = pts[faces[:, 1]]
        c = pts[faces[:, 2]]
        
        # Compute circumradius for each facet
        cos_ab = np.sum(a * b, axis=1)
        cos_bc = np.sum(b * c, axis=1)
        cos_ca = np.sum(c * a, axis=1)
        
        coeff_a = cos_bc - cos_ab * cos_ca
        coeff_b = cos_ca - cos_bc * cos_ab
        coeff_c = cos_ab - cos_ca * cos_bc
        
        cc_dir = coeff_a[:, np.newaxis] * a + coeff_b[:, np.newaxis] * b + coeff_c[:, np.newaxis] * c
        norms_cc = np.linalg.norm(cc_dir, axis=1, keepdims=True)
        norms_cc[norms_cc < 1e-10] = 1.0
        cc_dir = cc_dir / norms_cc
        
        cos_r = np.clip(np.sum(cc_dir * a, axis=1), -1.0, 1.0)
        radii = np.arccos(cos_r)
        
        m = np.max(radii)
        if m < 1e-15:
            return 0.0, np.zeros(3 * N)
        
        shifted = radii - m
        exp_vals = np.exp(beta * shifted)
        sum_exp = np.sum(exp_vals)
        obj = m + np.log(sum_exp) / beta
        
        weights = exp_vals / sum_exp
        significant = weights > 1e-8
        if not np.any(significant):
            return obj, np.zeros(3 * N)
        
        sig_weights = weights[significant]
        sig_faces = faces[significant]
        
        # For each significant facet, compute gradient w.r.t. its three vertices
        # The gradient of arccos(cos_r) w.r.t. vertex i is:
        # d(arccos(u))/du * du/d(vertex_i) where u = cc_dir · a_i
        # This is complex; use a simpler approach: move vertices toward circumcenter
        
        grad = np.zeros_like(pts)
        for idx, face in enumerate(sig_faces):
            w = sig_weights[idx]
            cc = cc_dir[significant][idx]
            # Gradient direction: moving vertex toward circumcenter reduces circumradius
            # Approximate gradient: -w * (cc - vertex) / |cc - vertex|^2 * sin(angle)
            for v in face:
                diff = cc - pts[v]
                norm_diff = np.linalg.norm(diff)
                if norm_diff > 1e-10:
                    grad[v] -= w * diff / (norm_diff * norm_diff)
        
        # Project gradient to tangent space
        grad_dot_p = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tan = grad - grad_dot_p * pts
        grad_flat = (grad_tan / norms).ravel()
        
        return obj, grad_flat

    def soft_max_objective(pts_flat, beta=100.0):
        """
        Smooth surrogate: soft-max of angular distances over the grid.
        Vectorized gradient computation.
        """
        pts = pts_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms

        dot = np.dot(grid_pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)

        min_dists = np.min(ang_dists, axis=1)
        nearest_idx = np.argmin(ang_dists, axis=1)

        m = np.max(min_dists)
        if m < 1e-15:
            return 0.0, np.zeros(3 * N)
        
        shifted = min_dists - m
        exp_vals = np.exp(beta * shifted)
        sum_exp = np.sum(exp_vals)
        obj = m + np.log(sum_exp) / beta

        weights = exp_vals / sum_exp
        significant = weights > 1e-8
        if not np.any(significant):
            return obj, np.zeros(3 * N)

        sig_weights = weights[significant]
        sig_idx = nearest_idx[significant]
        sig_grid = grid_pts[significant]

        u = np.sum(sig_grid * pts[sig_idx], axis=1)
        u = np.clip(u, -1.0 + 1e-12, 1.0 - 1e-12)
        g_perp = sig_grid - u[:, np.newaxis] * pts[sig_idx]
        denom = np.sqrt(1.0 - u * u)
        denom[denom < 1e-12] = 1e-12
        
        contrib = -sig_weights[:, np.newaxis] * g_perp / denom[:, np.newaxis]
        
        grad = np.zeros_like(pts)
        np.add.at(grad, sig_idx, contrib)
        
        grad_dot_p = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tan = grad - grad_dot_p * pts

        grad_flat = (grad_tan / norms).ravel()

        return obj, grad_flat

    def facet_circumcenter(pts, face):
        """Compute the circumcenter of a spherical triangle (face) and its circumradius."""
        a, b, c = pts[face]
        
        # Compute edge lengths
        d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
        d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
        d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
        
        # Spherical circumradius formula
        # Using the formula: cos(R) = (cos(a)cos(b)cos(c) + 2cos(a)cos(b)cos(c) - cos²(a) - cos²(b) - cos²(c)) / 
        #   (2(cos(a)cos(b) - cos(c)) ... )
        # Simpler: use the fact that the circumcenter is equidistant from all three vertices
        # The circumcenter direction is proportional to a cross b + b cross c + c cross a
        # (or the normal to the plane of the triangle, adjusted for sphere)
        
        # Actually, for a spherical triangle, the circumcenter is the point on the sphere
        # equidistant from a, b, c. This is the intersection of the perpendicular bisectors.
        # In 3D, it's the direction of the vector: a × b + b × c + c × a (for small triangles)
        # But for large triangles, we need the proper spherical circumcenter.
        
        # Use the formula: circumcenter = normalize(a * (cos(d_bc) - cos(d_ab)*cos(d_ca)) + 
        #   b * (cos(d_ca) - cos(d_bc)*cos(d_ab)) + c * (cos(d_ab) - cos(d_ca)*cos(d_bc)))
        # This is derived from the spherical law of cosines.
        
        cos_ab = np.dot(a, b)
        cos_bc = np.dot(b, c)
        cos_ca = np.dot(c, a)
        
        # Circumcenter direction
        coeff_a = cos_bc - cos_ab * cos_ca
        coeff_b = cos_ca - cos_bc * cos_ab
        coeff_c = cos_ab - cos_ca * cos_bc
        
        cc_dir = coeff_a * a + coeff_b * b + coeff_c * c
        norm = np.linalg.norm(cc_dir)
        
        if norm < 1e-10:
            # Degenerate case: use the centroid
            centroid = a + b + c
            norm_c = np.linalg.norm(centroid)
            if norm_c < 1e-10:
                return None, None
            cc_dir = centroid / norm_c
            norm = 1.0
        else:
            cc_dir = cc_dir / norm
        
        # Circumradius: angular distance from circumcenter to any vertex
        cos_r = np.clip(np.dot(cc_dir, a), -1.0, 1.0)
        r = np.arccos(cos_r)
        
        return cc_dir, r

    def facet_gradient_descent(points, n_iters=50):
        """
        Gradient descent that moves each vertex toward the circumcenter of its incident facets.
        This directly shrinks the circumradius of each facet.
        """
        for _ in range(n_iters):
            if time.time() > deadline - 10.0:
                break
            
            try:
                hull = ConvexHull(points)
                faces = hull.simplices
            except Exception:
                break
            
            # For each vertex, compute the average of circumcenters of its incident facets
            # weighted by the circumradius (larger facets get more weight)
            vertex_cc = np.zeros((N, 3))
            vertex_weights = np.zeros(N)
            
            for face in faces:
                cc_dir, r = facet_circumcenter(points, face)
                if cc_dir is None or r is None:
                    continue
                
                # Add contribution to each vertex of this face
                for v in face:
                    vertex_cc[v] += r * cc_dir
                    vertex_weights[v] += r
            
            # Move each vertex toward the average circumcenter of its incident facets
            for i in range(N):
                if vertex_weights[i] > 1e-10:
                    target = vertex_cc[i] / vertex_weights[i]
                    # Blend with current position
                    alpha = 0.3
                    points[i] = (1 - alpha) * points[i] + alpha * target
                    points[i] /= np.linalg.norm(points[i])
        
        return points
    
    def lloyd_facet_hybrid(points, n_iters=30):
        """
        Hybrid: alternate between Lloyd on coarse grid and facet-based refinement.
        """
        for _ in range(n_iters):
            if time.time() > deadline - 10.0:
                break
            
            # Lloyd step on coarse grid (fast)
            points = lloyd_iteration(points, n_iters=3, use_coarse=True)
            
            # Facet-based refinement
            try:
                hull = ConvexHull(points)
                faces = hull.simplices
            except Exception:
                continue
            
            vertex_cc = np.zeros((N, 3))
            vertex_weights = np.zeros(N)
            
            for face in faces:
                cc_dir, r = facet_circumcenter(points, face)
                if cc_dir is None or r is None:
                    continue
                for v in face:
                    vertex_cc[v] += r * cc_dir
                    vertex_weights[v] += r
            
            for i in range(N):
                if vertex_weights[i] > 1e-10:
                    target = vertex_cc[i] / vertex_weights[i]
                    alpha = 0.2
                    points[i] = (1 - alpha) * points[i] + alpha * target
                    points[i] /= np.linalg.norm(points[i])
        
        return points

    def soft_max_objective_fast(pts_flat, beta=100.0):
        """
        Faster variant using a coarser grid for gradient, exact for objective.
        """
        pts = pts_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms

        dot = np.dot(grid_pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)

        min_dists = np.min(ang_dists, axis=1)
        nearest_idx = np.argmin(ang_dists, axis=1)

        m = np.max(min_dists)
        if m < 1e-15:
            return 0.0, np.zeros(3 * N)
        
        shifted = min_dists - m
        exp_vals = np.exp(beta * shifted)
        sum_exp = np.sum(exp_vals)
        obj = m + np.log(sum_exp) / beta

        weights = exp_vals / sum_exp
        significant = weights > 1e-7
        if not np.any(significant):
            return obj, np.zeros(3 * N)

        sig_weights = weights[significant]
        sig_idx = nearest_idx[significant]
        sig_grid = grid_pts[significant]

        u = np.sum(sig_grid * pts[sig_idx], axis=1)
        u = np.clip(u, -1.0 + 1e-12, 1.0 - 1e-12)
        g_perp = sig_grid - u[:, np.newaxis] * pts[sig_idx]
        denom = np.sqrt(1.0 - u * u)
        denom[denom < 1e-12] = 1e-12
        
        contrib = -sig_weights[:, np.newaxis] * g_perp / denom[:, np.newaxis]
        
        grad = np.zeros_like(pts)
        np.add.at(grad, sig_idx, contrib)
        
        grad_dot_p = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tan = grad - grad_dot_p * pts

        grad_flat = (grad_tan / norms).ravel()

        return obj, grad_flat

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple random starts with Lloyd + repulsion + facet gradient
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 50.0:
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
            points = points + 0.2 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            rng = np.random.RandomState(200)
            centers = rng.randn(10, 3)
            centers /= np.linalg.norm(centers, axis=1, keepdims=True)
            points = np.zeros((N, 3))
            for i in range(N):
                c = centers[i % 10]
                points[i] = c + 0.3 * rng.randn(3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 4:
            if best_points is not None:
                rng = np.random.RandomState(300 + restart)
                points = best_points + 0.1 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 5:
            if best_points is not None:
                rng = np.random.RandomState(500 + restart)
                points = best_points + 0.15 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 6:
            rng = np.random.RandomState(600)
            # Fibonacci with different phase
            i = np.arange(N) + 0.3
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + 0.7
            r = np.sqrt(1.0 - z * z)
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        else:
            rng = np.random.RandomState(700)
            half = N // 2
            pts_half = rng.randn(half, 3)
            pts_half /= np.linalg.norm(pts_half, axis=1, keepdims=True)
            points = np.vstack([pts_half, -pts_half])
            if N % 2 != 0:
                points = np.vstack([points, [0, 0, 1]])
            points = points[:N]
        
        # Lloyd iteration on coarse grid first (fast)
        points = lloyd_iteration(points, n_iters=20, use_coarse=True)
        # Then refine on fine grid
        points = lloyd_iteration(points, n_iters=15, use_coarse=False)
        
        # Repulsion-based refinement
        alpha = 0.03
        decay = 0.995
        for _ in range(120):
            if time.time() > deadline - 35.0:
                break
            forces = repulsion_force(points, 1.5)
            points = points + alpha * forces
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            alpha *= decay
        
        # Hybrid Lloyd + facet refinement
        points = lloyd_facet_hybrid(points, n_iters=25)
        
        rad = compute_covering_radius(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Multi-scale L-BFGS-B optimization
    if best_points is not None and time.time() < deadline - 25.0:
        x0 = best_points.ravel()
        
        beta_schedule = [50.0, 100.0, 150.0, 200.0, 300.0]
        iters_per_beta = [120, 120, 100, 80, 60]
        
        for beta, maxiter in zip(beta_schedule, iters_per_beta):
            if time.time() > deadline - 12.0:
                break
            
            def obj_and_grad(x, b=beta):
                if time.time() > deadline - 5.0:
                    return 1e6, np.zeros_like(x)
                val, grad = soft_max_objective_fast(x, beta=b)
                return val, grad

            try:
                result = minimize(
                    obj_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8,
                             'maxfun': maxiter, 'disp': False}
                )
                opt_pts = result.x.reshape(N, 3)
                opt_norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
                opt_norms[opt_norms < 1e-15] = 1.0
                opt_pts = opt_pts / opt_norms

                rad_opt = compute_covering_radius(opt_pts)
                if rad_opt < best_rad:
                    best_rad = rad_opt
                    best_points = opt_pts.copy()
                    x0 = opt_pts.ravel()
            except Exception:
                pass

    # Phase 3: Final Lloyd + repulsion + facet gradient polish
    if best_points is not None and time.time() < deadline - 12.0:
        points = best_points.copy()
        points = lloyd_iteration(points, n_iters=20, use_coarse=True)
        points = lloyd_iteration(points, n_iters=15, use_coarse=False)
        
        alpha = 0.01
        decay = 0.998
        for _ in range(80):
            if time.time() > deadline - 6.0:
                break
            forces = repulsion_force(points, 1.5)
            points = points + alpha * forces
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            alpha *= decay
        
        points = lloyd_facet_hybrid(points, n_iters=20)
        
        rad = compute_covering_radius(points)
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 4: Final high-beta L-BFGS-B
    if best_points is not None and time.time() < deadline - 6.0:
        x0 = best_points.ravel()
        beta = 300.0

        def obj_and_grad_final(x):
            if time.time() > deadline - 2.0:
                return 1e6, np.zeros_like(x)
            val, grad = soft_max_objective(x, beta=beta)
            return val, grad

        try:
            result = minimize(
                obj_and_grad_final,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10,
                         'maxfun': 50, 'disp': False}
            )
            opt_pts = result.x.reshape(N, 3)
            opt_norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
            opt_norms[opt_norms < 1e-15] = 1.0
            opt_pts = opt_pts / opt_norms

            rad_opt = compute_covering_radius(opt_pts)
            if rad_opt < best_rad:
                best_rad = rad_opt
                best_points = opt_pts.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
