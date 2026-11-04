import numpy as np
import scipy.spatial
import time
from scipy.optimize import minimize

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
                if len(simplex) != 3:
                    continue
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

    def _facet_data(pts):
        """Return (radii, centers, simplices) for triangular facets."""
        try:
            hull = scipy.spatial.ConvexHull(pts)
        except Exception:
            return np.array([]), [], []
        
        radii = []
        centers = []
        simplices = []
        for simplex in hull.simplices:
            if len(simplex) != 3:
                continue
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
            radii.append(r)
            centers.append(O)
            simplices.append(simplex)
            
        return np.array(radii), centers, simplices

    def _smooth_covering_radius(pts, beta=50.0):
        """
        Smooth surrogate: log-sum-exp of facet circumradii.
        """
        radii, centers, simplices = _facet_data(pts)
        if len(radii) == 0:
            return 0.0
        
        m = np.max(radii)
        if m < 1e-12:
            return 0.0
            
        exps = np.exp(beta * (radii - m))
        lse = m + np.log(np.sum(exps)) / beta
        return lse

    def _smooth_covering_radius_grad(pts, beta=50.0):
        """
        Gradient of the smooth LSE surrogate w.r.t. pts.
        Uses the derivative of arccos(d) where d = O.A, and O depends on A,B,C.
        However, for a fixed hull topology, the gradient of the circumradius r w.r.t. vertices
        can be computed.
        
        r = arccos(d), d = O.A
        O = (B-A) x (C-A) / ||(B-A) x (C-A)||
        
        This is complex. A simpler approximation often used in these contexts:
        Treat the circumcenter O as fixed for the local gradient step (first order),
        then d/dA arccos(O.A) = -O / sqrt(1 - (O.A)^2).
        Since A is on the sphere, we project the gradient to the tangent space.
        
        For each facet, we add the weighted gradient contribution to its 3 vertices.
        """
        radii, centers, simplices = _facet_data(pts)
        if len(radii) == 0:
            return np.zeros_like(pts)
            
        m = np.max(radii)
        exps = np.exp(beta * (radii - m))
        sum_exps = np.sum(exps)
        if sum_exps < 1e-15:
            return np.zeros_like(pts)
            
        weights = exps / sum_exps
        
        grad = np.zeros_like(pts)
        
        for i in range(len(radii)):
            w = weights[i]
            if w < 1e-10:
                continue
            center = centers[i]
            simplex = simplices[i]
            for idx in simplex:
                p = pts[idx]
                cos_r = np.dot(center, p)
                cos_r = np.clip(cos_r, -1.0 + 1e-12, 1.0 - 1e-12)
                sin_r = np.sqrt(1.0 - cos_r * cos_r)
                if sin_r < 1e-12:
                    continue
                # Gradient of arccos(center . p) w.r.t p is -center / sin_r
                # But we are on the sphere, so we project to tangent space
                g = -center / sin_r
                # Project g to tangent plane at p
                g_tan = g - np.dot(g, p) * p
                grad[idx] += w * g_tan
                
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
        for i in range(N):
            proj = np.dot(grad[i], pts[i])
            grad[i] -= proj * pts[i]
        return grad

    def _lbfgs_refine(pts, max_iter=200, time_limit=10.0, beta=50.0):
        """Refine using L-BFGS-B on the smooth covering radius."""
        start = time.time()
        
        def objective_and_grad(flat_pts):
            if time.time() - start > time_limit:
                raise RuntimeError("time limit")
            pts_local = _normalize(flat_pts.reshape(N, 3))
            val = _smooth_covering_radius(pts_local, beta=beta)
            grad = _smooth_covering_radius_grad(pts_local, beta=beta)
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
            if res.success or res.fun < _smooth_covering_radius(pts, beta=beta):
                return _normalize(res.x.reshape(N, 3))
        except Exception:
            pass
        return pts

    def _lbfgs_riesz(pts, s=4.0, max_iter=100, time_limit=5.0):
        """Refine using L-BFGS-B on Riesz energy."""
        start = time.time()
        
        def objective_and_grad(flat_pts):
            if time.time() - start > time_limit:
                raise RuntimeError("time limit")
            pts_local = _normalize(flat_pts.reshape(N, 3))
            val = _riesz_energy(pts_local, s)
            grad = _riesz_energy_grad(pts_local, s)
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
            
            new_pts = np.zeros_like(pts)
            counts = np.zeros(N)
            
            for simplex in hull.simplices:
                if len(simplex) != 3:
                    continue
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
        
        pts = [verts]
        
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if d < 1.5:
                    edges.append((i, j))
        
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2.0
            pts.append(mid.reshape(1, 3))
        
        pts = np.vstack(pts)
        pts = _normalize(pts)
        
        if len(pts) < N:
            extra = _random_points()
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
    
    fib_jittered = _fibonacci_points()
    fib_jittered += 0.05 * rng.normal(size=(N, 3))
    fib_jittered = _normalize(fib_jittered)
    initializations.append(fib_jittered)
    
    best_points = None
    best_radius = np.pi
    
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline:
            break
        
        current_pts = init_pts.copy()
        current_r = _covering_radius(current_pts)
        
        # Stage 1: Riesz energy refinement
        if time.time() < deadline - 30:
            current_pts = _lbfgs_riesz(current_pts, s=4.0, max_iter=100, time_limit=8.0)
            current_r = _covering_radius(current_pts)
        
        # Stage 2: Smooth covering radius refinement with multi-stage beta
        if time.time() < deadline - 20:
            for beta in [10, 50, 200, 500]:
                if time.time() > deadline - 10:
                    break
                current_pts = _lbfgs_refine(current_pts, max_iter=100, time_limit=8.0, beta=beta)
                current_r = _covering_radius(current_pts)
        
        # Stage 3: Lloyd iterations
        if time.time() < deadline - 15:
            current_pts = _lloyd_iteration(current_pts, n_iter=30)
            current_r = _covering_radius(current_pts)
        
        # Stage 4: More smooth refinement
        if time.time() < deadline - 10:
            current_pts = _lbfgs_refine(current_pts, max_iter=100, time_limit=8.0, beta=500)
            current_r = _covering_radius(current_pts)
        
        # Stage 5: Simulated annealing
        if time.time() < deadline - 5:
            current_pts = _simulated_annealing(current_pts, n_iter=150, t_start=0.08, t_end=0.001, time_limit=12.0)
            current_r = _covering_radius(current_pts)
        
        # Final refinement
        if time.time() < deadline - 2:
            current_pts = _lbfgs_refine(current_pts, max_iter=50, time_limit=5.0, beta=500)
            current_r = _covering_radius(current_pts)
        
        if current_r < best_radius:
            best_radius = current_r
            best_points = current_pts.copy()
    
    # Final polish on the best
    if best_points is not None and time.time() < deadline - 2:
        best_points = _lbfgs_refine(best_points, max_iter=50, time_limit=5.0, beta=500)
        final_r = _covering_radius(best_points)
        if final_r < best_radius:
            best_radius = final_r
            best_points = best_points.copy()
    
    if best_points is None:
        best_points = _fibonacci_points()
    
    return best_points
