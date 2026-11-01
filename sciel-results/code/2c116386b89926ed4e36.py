"""Program for covering the sphere with 130 equal caps using smooth surrogate optimization."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

def _make_fib_grid(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

# Pre-compute grids
GRID_COARSE = _make_fib_grid(2000)
GRID_FINE = _make_fib_grid(8000)

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

    def compute_covering_radius_exact(pts):
        """Compute covering radius exactly as max circumradius over convex hull faces."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]
            
            # Compute circumradius of each face
            radii = np.zeros(len(faces))
            for i in range(len(faces)):
                ai, bi, ci = a[i], b[i], c[i]
                # Side lengths
                ab = np.linalg.norm(ai - bi)
                bc = np.linalg.norm(bi - ci)
                ca = np.linalg.norm(ci - ai)
                # Area using cross product
                area = 0.5 * np.linalg.norm(np.cross(ai - bi, ai - ci))
                if area < 1e-10:
                    radii[i] = 0.5 * max(ab, bc, ca)
                else:
                    radii[i] = (ab * bc * ca) / (4.0 * area)
            
            return float(np.max(radii))
        except Exception:
            # Fallback: dense grid-based estimate
            dot = np.dot(GRID_FINE, pts.T)
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

    def soft_covering_radius_and_grad(pts, grid, T_min=50.0, T_max=50.0):
        """
        Compute smooth approximation of covering radius and its gradient.
        Uses soft-min over points and soft-max over grid points.
        """
        M = len(grid)
        # Ensure points are normalized
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        pts_norm = pts / norms

        # Distances: M x N
        dot = np.dot(grid, pts_norm.T)
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)
        
        # Avoid division by zero in sin
        sin_dists = np.sin(ang_dists)
        sin_dists[sin_dists < 1e-10] = 1e-10

        # 1. Soft-min over points (axis 1) for each grid point
        # D_j = -1/T_min * log(sum_k exp(-T_min * d_jk))
        exp_neg_T_min_d = np.exp(-T_min * ang_dists)
        sum_exp_neg = np.sum(exp_neg_T_min_d, axis=1)
        D = -1.0 / T_min * np.log(sum_exp_neg)
        
        # Weights w_jk = exp(-T_min * d_jk) / sum_k exp(-T_min * d_jk)
        w_jk = exp_neg_T_min_d / sum_exp_neg[:, np.newaxis]

        # 2. Soft-max over grid points (axis 0)
        # R = 1/T_max * log(sum_j exp(T_max * D_j))
        exp_T_max_D = np.exp(T_max * D)
        sum_exp_pos = np.sum(exp_T_max_D)
        R = 1.0 / T_max * np.log(sum_exp_pos)
        
        # Weights W_j = exp(T_max * D_j) / sum_j exp(T_max * D_j)
        W_j = exp_T_max_D / sum_exp_pos

        # Gradient calculation
        # dR/dp_k = sum_j W_j * (dD_j/dp_k)
        # dD_j/dp_k = w_jk * (d d_jk / d p_k)
        # d d_jk / d p_k = proj_{p_k} (-grid_j / sin(d_jk))
        
        # Vectorized computation of gradient contributions
        # grad_contrib[j, k] = W_j * w_jk * (-grid_j / sin_dists[j, k])
        # Then project each row j,k to tangent space at p_k? 
        # No, the projection is specific to p_k.
        
        # Let's compute the raw gradient first:
        # raw_grad[j, k] = W_j * w_jk * (-grid_j / sin_dists[j, k])
        # This is an M x N x 3 tensor.
        
        # To save memory and time, we can sum over j first for each k.
        # G_k = sum_j W_j * w_jk * (-grid_j / sin_dists[j, k])
        
        # W_j is (M,), w_jk is (M, N), grid is (M, 3), sin_dists is (M, N).
        # Term for (j, k): W_j * w_jk * (-grid_j[:, np.newaxis] / sin_dists[:, np.newaxis])
        
        # We can compute this as:
        # weights_jk = W_j[:, np.newaxis] * w_jk  # M x N
        # inv_sin = 1.0 / sin_dists # M x N
        # grad_sum = np.sum(weights_jk[:, :, np.newaxis] * (-grid[:, np.newaxis, :] * inv_sin[:, :, np.newaxis]), axis=0)
        # This results in (N, 3)
        
        # Let's try a more efficient matmul approach if possible, or just use the loop if M is small enough.
        # M=8000, N=130. M*N = 1,040,000. This is manageable.
        
        weights_jk = W_j[:, np.newaxis] * w_jk
        # grad_raw[j, k] = weights_jk[j, k] * (-grid[j] / sin_dists[j, k])
        # We want G[k] = sum_j grad_raw[j, k]
        
        # grad_raw is (M, N, 3)
        # grad_raw = weights_jk[:, :, np.newaxis] * (-grid[:, np.newaxis, :] / sin_dists[:, :, np.newaxis])
        
        # This creates a large temporary array. Let's optimize.
        # G[k] = - sum_j (W_j * w_jk / sin_dists[j, k]) * grid[j]
        
        # Let A[j, k] = W_j * w_jk / sin_dists[j, k]
        # G[k] = - A[:, k] @ grid  (where A[:, k] is a vector of length M)
        
        A = weights_jk / sin_dists # M x N
        G = - (A.T @ grid) # N x 3
        
        # Now project G[k] to the tangent space at pts_norm[k]
        # proj_v(p) = v - (v . p) p
        dot_G_p = np.sum(G * pts_norm, axis=1, keepdims=True)
        G_tangent = G - dot_G_p * pts_norm
        
        return float(R), G_tangent.flatten()

    def optimize_with_lbfgsb(points, grid, maxiter=200, T_min=50.0, T_max=50.0):
        """Optimize using L-BFGS-B on the smooth covering radius objective."""
        
        def objective(x):
            val, _ = soft_covering_radius_and_grad(x.reshape(N, 3), grid, T_min, T_max)
            return val
        
        def gradient(x):
            _, g = soft_covering_radius_and_grad(x.reshape(N, 3), grid, T_min, T_max)
            return g
        
        x0 = points.flatten()
        
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        x_opt = result.x.reshape(N, 3)
        x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
        
        return x_opt, result.fun

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple starts with repulsion + L-BFGS-B optimization
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
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
        elif restart == 4:
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(300 + restart)
            points = points + 0.1 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 5:
            # Clustered start
            centers = fibonacci_spiral(5).astype(np.float64)
            points = np.concatenate([c + 0.1 * np.random.randn(26, 3) for c in centers])
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is not None:
                rng = np.random.RandomState(400 + restart)
                points = best_points + 0.05 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        # First, optimize with repulsion to get a good starting configuration
        points = gradient_descent_repulsion(points, 2.0, 0.05, 0.995, 100)
        
        # Then optimize with L-BFGS-B on the coarse grid
        if time.time() < deadline - 15.0:
            points, _ = optimize_with_lbfgsb(points, GRID_COARSE, maxiter=150)
        
        # Evaluate exact covering radius
        rad = compute_covering_radius_exact(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Local refinement with fine grid
    if best_points is not None and time.time() < deadline - 20.0:
        for trial in range(4):
            if time.time() > deadline - 10.0:
                break
            rng = np.random.RandomState(500 + trial)
            candidate = best_points + 0.03 * rng.randn(N, 3)
            candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)
            
            if time.time() < deadline - 8.0:
                candidate, _ = optimize_with_lbfgsb(candidate, GRID_FINE, maxiter=100)
            
            rad = compute_covering_radius_exact(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()

    # Phase 3: Facet-based local search
    # Identify the facets that define the maximum radius and perturb the points involved
    if best_points is not None and time.time() < deadline - 5.0:
        points = best_points.copy()
        try:
            hull = ConvexHull(points)
            faces = hull.simplices
            a = points[faces[:, 0]]
            b = points[faces[:, 1]]
            c = points[faces[:, 2]]
            
            radii = np.zeros(len(faces))
            for i in range(len(faces)):
                ai, bi, ci = a[i], b[i], c[i]
                ab = np.linalg.norm(ai - bi)
                bc = np.linalg.norm(bi - ci)
                ca = np.linalg.norm(ci - ai)
                area = 0.5 * np.linalg.norm(np.cross(ai - bi, ai - ci))
                if area < 1e-10:
                    radii[i] = 0.5 * max(ab, bc, ca)
                else:
                    radii[i] = (ab * bc * ca) / (4.0 * area)
            
            # Find the face with max radius
            max_idx = np.argmax(radii)
            face = faces[max_idx]
            
            # Perturb the points in this face
            for _ in range(20):
                if time.time() > deadline - 3.0:
                    break
                rng = np.random.RandomState(np.random.randint(10000))
                new_points = points.copy()
                for idx in face:
                    direction = rng.randn(3)
                    direction -= np.dot(direction, points[idx]) * points[idx]
                    direction /= np.linalg.norm(direction)
                    new_points[idx] = points[idx] + 0.01 * direction
                    new_points[idx] /= np.linalg.norm(new_points[idx])
                
                rad = compute_covering_radius_exact(new_points)
                if rad < best_rad:
                    best_rad = rad
                    best_points = new_points.copy()
                    points = new_points.copy()
                    # Recalculate hull for next iteration if improved
                    try:
                        hull = ConvexHull(points)
                        faces = hull.simplices
                    except:
                        break
        except:
            pass

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
