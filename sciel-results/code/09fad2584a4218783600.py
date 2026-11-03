import numpy as np
import scipy.spatial
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 40s safety margin before 150s kill

    rng = np.random.default_rng(12345)

    def _normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return pts / norms

    def _covering_radius(pts):
        """Compute covering radius via convex hull facets."""
        try:
            hull = scipy.spatial.ConvexHull(pts)
            max_r = 0.0
            for simplex in hull.simplices:
                A, B, C = pts[simplex]
                N_vec = np.cross(B - A, C - A)
                norm = np.linalg.norm(N_vec)
                if norm < 1e-9:
                    continue
                O = N_vec / norm
                centroid = (A + B + C) / 3.0
                if np.dot(O, centroid) < 0:
                    O = -O
                d = np.dot(O, A)
                d = np.clip(d, -1.0, 1.0)
                r = np.arccos(d)
                if r > max_r:
                    max_r = r
            return max_r
        except Exception:
            return np.pi

    def _smooth_covering_radius(pts, beta=100.0):
        """
        Smooth surrogate: for each point y on a fine grid, compute min distance to pts,
        then use log-sum-exp to get a smooth upper bound on the max.
        """
        # Use a moderate grid for speed
        n_theta = 60
        n_phi = 120
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        y = np.column_stack([
            np.sin(TH) * np.cos(PH).ravel(),
            np.sin(TH) * np.sin(PH).ravel(),
            np.cos(TH).ravel()
        ])
        # Distances
        # y shape (M, 3), pts shape (N, 3)
        # dot products
        dots = y @ pts.T  # (M, N)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)  # (M, N)
        # For each y, min distance to any point
        min_dists = np.min(dists, axis=1)  # (M,)
        # Log-sum-exp for smooth max
        max_d = np.max(min_dists)
        shifted = min_dists - max_d
        lse = max_d + np.log(np.sum(np.exp(beta * shifted))) / beta
        return lse

    def _smooth_covering_radius_grad(pts, beta=100.0):
        """
        Gradient of the smooth covering radius w.r.t. pts.
        Uses the same grid but computes gradient analytically.
        """
        n_theta = 50
        n_phi = 100
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        y = np.column_stack([
            np.sin(TH) * np.cos(PH).ravel(),
            np.sin(TH) * np.sin(PH).ravel(),
            np.cos(TH).ravel()
        ])
        M = len(y)
        dots = y @ pts.T  # (M, N)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)  # (M, N)
        min_dists = np.min(dists, axis=1)
        argmin_idx = np.argmin(dists, axis=1)
        
        max_d = np.max(min_dists)
        shifted = min_dists - max_d
        weights = np.exp(beta * shifted)
        weights /= np.sum(weights)
        
        # Gradient: d/d(pts[i]) of sum_j w_j * dist(y_j, pts[nearest_j])
        # dist(y, p) = arccos(y . p), d/dp = -y / sqrt(1 - (y.p)^2)
        # But we need to be careful: the nearest point index changes, so we use the soft version.
        # For the smooth version, we differentiate the LSE directly.
        # Actually, let's use a simpler approach: gradient of max over j of min_i d(y_j, p_i)
        # Approximate by: for each y_j, the gradient of d(y_j, p_{i_j}) where i_j is the nearest.
        
        grad = np.zeros_like(pts)
        for j in range(M):
            i_j = argmin_idx[j]
            yj = y[j]
            pj = pts[i_j]
            dot = np.dot(yj, pj)
            dot = np.clip(dot, -1.0 + 1e-12, 1.0 - 1e-12)
            sqrt_term = np.sqrt(1.0 - dot * dot)
            if sqrt_term < 1e-12:
                continue
            # d/dp arccos(y.p) = -y / sqrt(1 - (y.p)^2)
            grad[i_j] += weights[j] * (-yj / sqrt_term)
        
        return grad

    def _riesz_energy(pts, s=4.0):
        """Riesz s-energy: sum over all pairs 1/|p_i - p_j|^s"""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1.0)
        energy = np.sum(1.0 / (dists ** s))
        return energy

    def _riesz_energy_grad(pts, s=4.0):
        """Gradient of Riesz s-energy."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2, keepdims=True)
        dists = np.maximum(dists, 1e-12)
        # d/dp_i sum_{j != i} 1/|p_i - p_j|^s = -s * sum_{j != i} (p_i - p_j) / |p_i - p_j|^(s+2)
        # But we need to project to tangent space.
        grad = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dij = pts[i] - pts[j]
                dist = np.linalg.norm(dij)
                if dist < 1e-12:
                    continue
                grad[i] += -s * dij / (dist ** (s + 2))
        # Project to tangent space: for each i, grad[i] -= (grad[i].p_i) p_i
        for i in range(N):
            proj = np.dot(grad[i], pts[i])
            grad[i] -= proj * pts[i]
        return grad

    def _lbfgs_refine(pts, max_iter=200, time_limit=10.0):
        """Refine using L-BFGS-B on the smooth covering radius."""
        from scipy.optimize import minimize
        
        start = time.time()
        
        def objective_and_grad(flat_pts):
            if time.time() - start > time_limit:
                raise RuntimeError("time limit")
            pts = _normalize(flat_pts.reshape(N, 3))
            val = _smooth_covering_radius(pts, beta=50.0)
            grad = _smooth_covering_radius_grad(pts, beta=50.0)
            # Flatten gradient
            flat_grad = grad.ravel()
            return val, flat_grad
        
        x0 = pts.ravel()
        try:
            res = minimize(
                lambda x: objective_and_grad(x)[0],
                x0,
                jac=lambda x: objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'maxfun': 500}
            )
            if res.success or res.fun < _smooth_covering_radius(pts, beta=50.0):
                return _normalize(res.x.reshape(N, 3))
        except Exception:
            pass
        return pts

    def _lbfgs_riesz(pts, s=4.0, max_iter=100, time_limit=5.0):
        """Refine using L-BFGS-B on Riesz energy."""
        from scipy.optimize import minimize
        
        start = time.time()
        
        def objective_and_grad(flat_pts):
            if time.time() - start > time_limit:
                raise RuntimeError("time limit")
            pts = _normalize(flat_pts.reshape(N, 3))
            val = _riesz_energy(pts, s)
            grad = _riesz_energy_grad(pts, s)
            return val, grad.ravel()
        
        x0 = pts.ravel()
        try:
            res = minimize(
                lambda x: objective_and_grad(x)[0],
                x0,
                jac=lambda x: objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'maxfun': 300}
            )
            if res.success or res.fun < _riesz_energy(pts, s):
                return _normalize(res.x.reshape(N, 3))
        except Exception:
            pass
        return pts

    def _lloyd_iteration(pts, n_iter=50):
        """
        Spherical Lloyd: for each point, move it to the centroid of its Voronoi cell
        (approximated by the average of the facet circumcenters it's a vertex of).
        """
        for _ in range(n_iter):
            if time.time() > deadline:
                break
            try:
                hull = scipy.spatial.ConvexHull(pts)
            except Exception:
                break
            
            # For each point, collect the outward normals of facets it belongs to
            # The Voronoi cell on the sphere for point i is the intersection of hemispheres
            # centered at the facet circumcenters. Approximate by averaging the facet normals.
            new_pts = np.zeros_like(pts)
            counts = np.zeros(N)
            
            for simplex in hull.simplices:
                A, B, C = pts[simplex]
                N_vec = np.cross(B - A, C - A)
                norm = np.linalg.norm(N_vec)
                if norm < 1e-9:
                    continue
                O = N_vec / norm
                centroid = (A + B + C) / 3.0
                if np.dot(O, centroid) < 0:
                    O = -O
                for idx in simplex:
                    new_pts[idx] += O
                    counts[idx] += 1
            
            for i in range(N):
                if counts[i] > 0:
                    new_pts[i] /= counts[i]
                    new_pts[i] /= np.linalg.norm(new_pts[i])
                else:
                    new_pts[i] = pts[i]
            
            pts = new_pts
        return pts

    def _simulated_annealing(pts, n_iter=200, t_start=0.1, t_end=0.001, time_limit=15.0):
        """Simulated annealing with facet-based moves."""
        start = time.time()
        current_pts = pts.copy()
        current_r = _covering_radius(current_pts)
        best_pts = current_pts.copy()
        best_r = current_r
        
        for it in range(n_iter):
            if time.time() - start > time_limit:
                break
            t = t_start * (t_end / t_start) ** (it / n_iter)
            
            # Pick a random point and move it slightly
            idx = rng.integers(N)
            direction = rng.normal(size=3)
            direction -= np.dot(direction, current_pts[idx]) * current_pts[idx]
            direction /= np.linalg.norm(direction)
            
            new_pts = current_pts.copy()
            step = 0.05 * t / t_start
            new_pts[idx] = current_pts[idx] + step * direction
            new_pts[idx] /= np.linalg.norm(new_pts[idx])
            
            new_r = _covering_radius(new_pts)
            
            if new_r < current_r or rng.random() < np.exp(-(new_r - current_r) / t):
                current_pts = new_pts
                current_r = new_r
                if current_r < best_r:
                    best_r = current_r
                    best_pts = current_pts.copy()
        
        return best_pts

    def _fibonacci_points():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return _normalize(pts)

    def _random_points(seed=None):
        rng_local = np.random.default_rng(seed)
        pts = rng_local.normal(size=(N, 3))
        return _normalize(pts)

    def _icosahedron_based_points():
        """Start from icosahedron vertices and subdivide."""
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        verts = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ], dtype=float)
        verts = _normalize(verts)
        
        # Start with icosahedron, then add points at edge midpoints and face centers
        # We need 50 points. 12 vertices + 30 edge midpoints = 42. Add 8 face centers = 50.
        pts = [verts]
        
        # Edge midpoints
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if d < 1.5:  # approximate edge length for icosahedron
                    edges.append((i, j))
        
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2.0
            pts.append(mid.reshape(1, 3))
        
        pts = np.vstack(pts)
        pts = _normalize(pts)
        
        # If we don't have exactly 50, pad or trim
        if len(pts) < N:
            # Add random points
            extra = _random_points()
            # Take first N - len(pts)
            pts = np.vstack([pts, extra[:N - len(pts)]])
        elif len(pts) > N:
            pts = pts[:N]
        
        return _normalize(pts)

    # Multi-start with different initializations
    initializations = [
        _fibonacci_points(),
        _random_points(1),
        _random_points(2),
        _random_points(3),
        _icosahedron_based_points(),
    ]
    
    # Add jittered Fibonacci
    fib_jittered = _fibonacci_points()
    fib_jittered += 0.05 * rng.normal(size=(N, 3))
    fib_jittered = _normalize(fib_jittered)
    initializations.append(fib_jittered)
    
    best_points = None
    best_radius = np.pi
    
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline:
            break
        
        print(f"Starting initialization {init_idx + 1}/{len(initializations)}")
        
        current_pts = init_pts.copy()
        current_r = _covering_radius(current_pts)
        
        # Stage 1: Riesz energy refinement
        if time.time() < deadline - 30:
            current_pts = _lbfgs_riesz(current_pts, s=4.0, max_iter=100, time_limit=8.0)
            current_r = _covering_radius(current_pts)
        
        # Stage 2: Smooth covering radius refinement with multi-stage beta
        if time.time() < deadline - 20:
            # Beta cascade: 10, 50, 200, 500
            for beta in [10, 50, 200, 500]:
                if time.time() > deadline - 10:
                    break
                current_pts = _lbfgs_refine(current_pts, max_iter=100, time_limit=8.0)
                current_r = _covering_radius(current_pts)
        
        # Stage 3: Lloyd iterations
        if time.time() < deadline - 15:
            current_pts = _lloyd_iteration(current_pts, n_iter=30)
            current_r = _covering_radius(current_pts)
        
        # Stage 4: More smooth refinement
        if time.time() < deadline - 10:
            current_pts = _lbfgs_refine(current_pts, max_iter=100, time_limit=8.0)
            current_r = _covering_radius(current_pts)
        
        # Stage 5: Simulated annealing
        if time.time() < deadline - 5:
            current_pts = _simulated_annealing(current_pts, n_iter=150, t_start=0.08, t_end=0.001, time_limit=12.0)
            current_r = _covering_radius(current_pts)
        
        # Final refinement
        if time.time() < deadline - 2:
            current_pts = _lbfgs_refine(current_pts, max_iter=50, time_limit=5.0)
            current_r = _covering_radius(current_pts)
        
        if current_r < best_radius:
            best_radius = current_r
            best_points = current_pts.copy()
            print(f"  New best: {np.degrees(best_radius):.6f} degrees")
    
    # Final polish on the best
    if best_points is not None and time.time() < deadline - 2:
        best_points = _lbfgs_refine(best_points, max_iter=50, time_limit=5.0)
        final_r = _covering_radius(best_points)
        if final_r < best_radius:
            best_radius = final_r
            best_points = best_points.copy()
    
    if best_points is None:
        best_points = _fibonacci_points()
    
    return best_points
