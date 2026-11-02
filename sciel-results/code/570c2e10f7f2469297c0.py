"""Program for covering the sphere with 130 equal caps using repulsion and facet-local search."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# Pre-compute a dense grid for the smooth objective
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
            dot = np.dot(GRID_FINE, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    def riesz_energy_grad(pts, exponent=20.0):
        """Compute Riesz energy and gradient. High exponent approximates covering radius minimization."""
        dot = np.dot(pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-12] = 1e-12
        
        # E = sum_{i<j} d_{ij}^{-s}
        # Using dist_sq, d = sqrt(dist_sq). d^{-s} = dist_sq^{-s/2}
        s = exponent
        w = dist_sq ** (-s / 2.0)
        np.fill_diagonal(w, 0.0)
        
        energy = 0.5 * np.sum(w) # 0.5 to avoid double counting i<j vs j<i
        
        # Gradient
        # d/dx_i (d_{ij}^{-s}) = -s * d_{ij}^{-s-1} * d/dx_i (d_{ij})
        # d_{ij} = ||x_i - x_j||. d/dx_i d_{ij} = (x_i - x_j) / d_{ij}
        # So grad_i = sum_{j != i} -s * d_{ij}^{-s-1} * (x_i - x_j) / d_{ij}
        #           = sum_{j != i} -s * d_{ij}^{-s-2} * (x_i - x_j)
        # Note: dist_sq = d^2. d^{-s-2} = dist_sq^{-(s/2 + 1)}
        
        w_grad_factor = dist_sq ** (-(s / 2.0) - 1.0)
        np.fill_diagonal(w_grad_factor, 0.0)
        
        # grad_i = -s * sum_j w_grad_factor[i,j] * (x_i - x_j)
        #        = -s * ( sum_j w_grad_factor[i,j] * x_i - sum_j w_grad_factor[i,j] * x_j )
        #        = -s * ( x_i * sum_j w_grad_factor[i,j] - (w_grad_factor @ x)[i] )
        
        row_sums = np.sum(w_grad_factor, axis=1)
        w_grad_factor_x = w_grad_factor @ pts
        
        grad = -s * (pts * row_sums[:, np.newaxis] - w_grad_factor_x)
        
        return float(energy), grad

    def optimize_riesz(points, exponent, maxiter=200):
        """Optimize using L-BFGS-B on Riesz energy."""
        
        def objective(x):
            pts = x.reshape(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            e, _ = riesz_energy_grad(pts, exponent)
            return e
        
        def gradient(x):
            pts = x.reshape(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            _, g = riesz_energy_grad(pts, exponent)
            # Project gradient to tangent space
            for i in range(N):
                p = pts[i]
                g_tan = g[i] - np.dot(g[i], p) * p
                g[i] = g_tan
            
            return g.flatten()

        x0 = points.flatten()
        
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        x_opt = result.x.reshape(N, 3)
        x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
        
        return x_opt

    def facet_local_search(points, max_rounds=5):
        """
        Local search by perturbing points that belong to the largest facets.
        """
        best_points = points.copy()
        best_rad = compute_covering_radius_exact(best_points)
        
        for round_idx in range(max_rounds):
            if time.time() > deadline - 10.0:
                break
            
            try:
                hull = ConvexHull(best_points)
                faces = hull.simplices
                a = best_points[faces[:, 0]]
                b = best_points[faces[:, 1]]
                c = best_points[faces[:, 2]]
                
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
                
                sorted_indices = np.argsort(radii)[::-1]
                K = 5
                top_indices = sorted_indices[:K]
                
                improved = False
                for idx in top_indices:
                    if time.time() > deadline - 5.0:
                        break
                    
                    face = faces[idx]
                    unique_pts = np.unique(face)
                    
                    for p_idx in unique_pts:
                        old_p = best_points[p_idx].copy()
                        
                        if valid[idx]:
                            normal = c_dir[idx] / norms[idx]
                        else:
                            normal = np.cross(best_points[face[0]], best_points[face[1]])
                            normal /= np.linalg.norm(normal)
                        
                        rng = np.random.RandomState(p_idx + round_idx * 1000 + int(idx))
                        for trial in range(10):
                            direction = rng.randn(3)
                            direction -= np.dot(direction, old_p) * old_p
                            dir_norm = np.linalg.norm(direction)
                            if dir_norm < 1e-10:
                                continue
                            direction /= dir_norm
                            
                            for step in [0.01, 0.005, 0.001]:
                                new_p = old_p + step * direction
                                new_p /= np.linalg.norm(new_p)
                                best_points[p_idx] = new_p
                                
                                new_rad = compute_covering_radius_exact(best_points)
                                if new_rad < best_rad - 1e-12:
                                    best_rad = new_rad
                                    improved = True
                                    break
                                else:
                                    best_points[p_idx] = old_p
                            else:
                                continue
                            break
                        
                        if improved:
                            break
                    if improved:
                        break
                
                if not improved:
                    break
            except Exception:
                break
        
        return best_points, best_rad

    best_points = None
    best_rad = np.inf

    # Phase 1: Repulsion + Riesz L-BFGS
    n_restarts = 5
    
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
            points = points + 0.5 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            rng = np.random.RandomState(200 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is not None:
                rng = np.random.RandomState(300 + restart)
                points = best_points + 0.1 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        # Use Riesz Energy L-BFGS with high exponent
        if time.time() < deadline - 20.0:
            points = optimize_riesz(points, exponent=20.0, maxiter=300)
        
        rad = compute_covering_radius_exact(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Facet Local Search
    if best_points is not None and time.time() < deadline - 10.0:
        best_points, best_rad = facet_local_search(best_points, max_rounds=10)

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
