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

    def riesz_energy_grad(pts, s):
        """Compute Riesz s-energy and gradient. Higher s approximates hard sphere packing."""
        dot = np.dot(pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        # Avoid division by zero
        dist_sq[dist_sq < 1e-12] = 1e-12
        
        # Energy E = sum_{i!=j} (dist_sq)^(-s/2)
        # We use dist_sq to avoid sqrt
        E_term = dist_sq ** (-s/2.0)
        np.fill_diagonal(E_term, 0.0)
        
        # Gradient dE/dpts_i = sum_{j!=i} -s/2 * (dist_sq)^(-s/2 - 1) * 2 * (pts_i - pts_j)
        # = -s * sum_{j!=i} (dist_sq)^(-s/2 - 1) * (pts_i - pts_j)
        # Let W_ij = s * (dist_sq_ij)^(-s/2 - 1)
        # grad_i = sum_j W_ij * (pts_i - pts_j) = pts_i * sum_j W_ij - sum_j W_ij * pts_j
        
        W = s * (dist_sq ** (-s/2.0 - 1.0))
        np.fill_diagonal(W, 0.0)
        
        row_sums = np.sum(W, axis=1)
        grad = row_sums[:, np.newaxis] * pts - W @ pts
        
        energy = float(np.sum(E_term))
        return energy, grad

    def optimize_riesz(points, s, max_iter=200, alpha_init=0.01):
        """Optimize points by gradient descent on Riesz energy."""
        pts = points.copy()
        alpha = alpha_init
        best_energy = np.inf
        best_pts = pts.copy()
        
        for i in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            energy, grad = riesz_energy_grad(pts, s)
            
            # Project gradient to tangent space
            # grad_tan = grad - (grad . p) p
            dot_grad_p = np.sum(grad * pts, axis=1, keepdims=True)
            grad_tan = grad - dot_grad_p * pts
            
            # Normalize gradient to stabilize step size
            grad_norm = np.linalg.norm(grad_tan)
            if grad_norm < 1e-10:
                break
            grad_tan_norm = grad_tan / grad_norm
            
            # Update
            new_pts = pts + alpha * grad_tan_norm
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            
            # Simple backtracking line search if energy increases significantly
            # For simplicity, we just decay alpha if no improvement
            if energy < best_energy:
                best_energy = energy
                best_pts = new_pts.copy()
                # Slightly increase alpha if improving fast? No, keep it simple.
                # alpha *= 1.01 # Aggressive
            else:
                alpha *= 0.9
            
            pts = new_pts
            
            # Stop if gradient is small
            if np.linalg.norm(grad_tan) < 1e-6:
                break
        
        return best_pts

    def facet_local_search(points, max_rounds=5):
        """
        Local search by perturbing points that belong to the largest facets.
        This targets the covering radius directly.
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
                
                # Sort facets by radius, largest first
                sorted_indices = np.argsort(radii)[::-1]
                
                # Take top K facets
                K = 5
                top_indices = sorted_indices[:K]
                
                improved = False
                for idx in top_indices:
                    if time.time() > deadline - 5.0:
                        break
                    
                    face = faces[idx]
                    # Get unique points in this facet
                    unique_pts = np.unique(face)
                    
                    # For each point in the facet, try to move it to reduce the facet radius
                    for p_idx in unique_pts:
                        old_p = best_points[p_idx].copy()
                        
                        # Compute the normal of the facet (pointing outward)
                        # The circumcenter direction is c_dir / norm
                        if valid[idx]:
                            normal = c_dir[idx] / norms[idx]
                        else:
                            # Fallback normal
                            normal = np.cross(best_points[face[0]], best_points[face[1]])
                            normal /= np.linalg.norm(normal)
                        
                        # Try moving the point towards the normal (outward) or tangentially
                        # Moving outward reduces the cap size? No, moving outward on sphere means changing direction.
                        # We want to move the point such that the circumradius of the facet decreases.
                        # The circumcenter is at normal * cos(r). 
                        # If we move the point, the circumcenter moves.
                        # A heuristic: move the point slightly in the direction of the circumcenter?
                        # Or move it to balance the distances to the other two points?
                        
                        # Let's try a small random perturbation and check if radius decreases.
                        rng = np.random.RandomState(p_idx + round_idx * 1000 + int(idx))
                        for trial in range(10):
                            direction = rng.randn(3)
                            # Project to tangent plane
                            direction -= np.dot(direction, old_p) * old_p
                            dir_norm = np.linalg.norm(direction)
                            if dir_norm < 1e-10:
                                continue
                            direction /= dir_norm
                            
                            # Try small steps
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

    # Phase 1: Multiple starts with Riesz energy optimization
    n_restarts = 8
    
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
                rng = np.random.RandomState(300 + restart)
                points = best_points + 0.2 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        # Optimize with Riesz energy. Use high s to push points apart.
        # s=20 is a good balance between speed and "hard sphere" behavior
        points = optimize_riesz(points, s=20.0, max_iter=300, alpha_init=0.05)
        
        # Evaluate exact covering radius
        rad = compute_covering_radius_exact(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Facet Local Search
    if best_points is not None and time.time() < deadline - 30.0:
        best_points, best_rad = facet_local_search(best_points, max_rounds=10)

    # Phase 3: Local refinement with Riesz optimization from perturbed best
    if best_points is not None and time.time() < deadline - 20.0:
        for trial in range(5):
            if time.time() > deadline - 10.0:
                break
            rng = np.random.RandomState(500 + trial)
            candidate = best_points + 0.1 * rng.randn(N, 3)
            candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)
            
            # Use Riesz to push out of local minima
            candidate = optimize_riesz(candidate, s=20.0, max_iter=100, alpha_init=0.02)
            
            rad = compute_covering_radius_exact(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()
                
                # Run facet local search again on the new best
                if time.time() < deadline - 10.0:
                    best_points, best_rad = facet_local_search(best_points, max_rounds=5)

    # Phase 4: Final polish with Facet Local Search (most effective for covering radius)
    if best_points is not None and time.time() < deadline - 5.0:
        best_points, best_rad = facet_local_search(best_points, max_rounds=20)

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
