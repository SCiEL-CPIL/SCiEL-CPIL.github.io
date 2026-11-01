"""Program for covering the sphere with 130 equal caps."""
import numpy as np
import time
from scipy.spatial import ConvexHull

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

    def compute_covering_radius(pts):
        """Compute covering radius as max circumradius over convex hull faces (vectorized)."""
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

    def generate_grid_points(num_lat=60, num_lon=120):
        """Generate a grid of points on the sphere for soft-max objective."""
        lat = np.linspace(-np.pi/2, np.pi/2, num_lat)
        lon = np.linspace(0, 2*np.pi, num_lon, endpoint=False)
        LA, LO = np.meshgrid(lat, lon, indexing='ij')
        z = np.sin(LA)
        x = np.cos(LA) * np.cos(LO)
        y = np.cos(LA) * np.sin(LO)
        return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    # Pre-compute grid points for soft-max objective
    grid_pts = generate_grid_points(40, 80)  # 3200 points, good balance of speed/accuracy

    def soft_max_covering_radius_and_grad(pts, beta=50.0):
        """
        Compute soft-max covering radius and its gradient w.r.t. points.
        Objective: max_y min_i angle(y, p_i) approximated by soft-max of min distances.
        Actually, we want to minimize the max distance.
        Let d_i(y) = angle(y, p_i). We want to minimize max_y min_i d_i(y).
        This is equivalent to minimizing the soft-max over y of the soft-min over i of d_i(y).
        
        Simpler approach:
        Minimize F(pts) = sum_y softmax(-beta * min_i d_i(y))
        Where d_i(y) = arccos(y . p_i).
        Gradient of d_i(y) w.r.t p_i:
        d/dp_i arccos(y . p_i) = - y / sqrt(1 - (y.p_i)^2)
        Since we minimize d_i, we want to move p_i in direction -grad(d_i) = y / sqrt(...).
        But we have min_i d_i. The gradient of min is 0 except for the argmin.
        Using soft-min: m(y) = -1/beta * log(sum_i exp(-beta * d_i(y))).
        grad m(y) = sum_i (exp(-beta d_i) / sum_j exp(-beta d_j)) * grad d_i.
        
        Then we minimize sum_y m(y).
        Total gradient for p_k is sum_y [ weight_k(y) * grad d_k(y) ]
        where weight_k(y) = exp(-beta d_k(y)) / sum_j exp(-beta d_j(y)).
        
        Note: grad d_k(y) = - y / sqrt(1 - (y.p_k)^2).
        So we move p_k in direction sum_y weight_k(y) * y / sqrt(1 - (y.p_k)^2).
        This pushes p_k towards the points y that are currently "covered" by p_k.
        """
        # Compute distances d_i(y)
        dot = np.dot(grid_pts, pts.T)  # shape (G, N)
        dot = np.clip(dot, -1.0, 1.0)
        
        # Numerical stability for arccos
        eps = 1e-12
        dists = np.arccos(dot)  # shape (G, N)
        
        # Soft-min over i for each y
        # m(y) = -1/beta * log(sum_i exp(-beta * d_i(y)))
        neg_beta_dists = -beta * dists
        # Shift for stability
        max_neg_beta_dists = np.max(neg_beta_dists, axis=1, keepdims=True)
        exp_neg_beta_dists = np.exp(neg_beta_dists - max_neg_beta_dists)
        sum_exp = np.sum(exp_neg_beta_dists, axis=1, keepdims=True)
        
        # Weights w_k(y) = exp(-beta d_k) / sum_j exp(-beta d_j)
        weights = exp_neg_beta_dists / sum_exp  # shape (G, N)
        
        # Gradient of d_k(y) w.r.t p_k: - y / sqrt(1 - dot^2)
        # We want to minimize sum_y m(y).
        # grad m(y) w.r.t p_k = w_k(y) * grad d_k(y)
        # grad d_k(y) = - y / sqrt(1 - dot_k(y)^2)
        
        # Compute sqrt(1 - dot^2)
        sqrt_term = np.sqrt(np.maximum(1.0 - dot**2, eps))  # shape (G, N)
        
        # grad_d = - grid_pts[:, np.newaxis, :] / sqrt_term[:, :, np.newaxis]
        # We accumulate gradient for each p_k:
        # grad_p_k = sum_y w_k(y) * (- y / sqrt_term_k(y))
        # grad_p_k = - sum_y (w_k(y) / sqrt_term_k(y)) * y
        
        coeff = weights / sqrt_term  # shape (G, N)
        
        # grad_pts[k] = - sum_y coeff[y, k] * grid_pts[y]
        # grad_pts = - (coeff.T @ grid_pts)
        grad_pts = - (coeff.T @ grid_pts)  # shape (N, 3)
        
        # Normalize gradient to unit sphere? No, just project.
        # The gradient on the sphere is the projection of the Euclidean gradient.
        # grad_sphere = grad_pts - (grad_pts . p_k) p_k
        dot_grad_pts = np.sum(grad_pts * pts, axis=1, keepdims=True)
        grad_sphere = grad_pts - dot_grad_pts * pts
        
        # Objective value (soft-max approximation of max min-distance)
        # m(y) = -1/beta * log(sum_exp) - max_neg_beta_dists / beta
        m_y = -1.0/beta * (np.log(sum_exp) - max_neg_beta_dists)
        obj_val = np.sum(m_y)
        
        return obj_val, grad_sphere

    def gradient_descent_smooth(points, alpha_init, decay, max_iter, beta=50.0):
        """Optimize points via gradient descent on the soft-max covering radius."""
        alpha = alpha_init
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            obj, grad = soft_max_covering_radius_and_grad(points, beta)
            points = points - alpha * grad
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            alpha *= decay
        return points

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple random starts with varying beta
    n_restarts = 5
    betas = [20.0, 50.0, 100.0, 20.0, 50.0]
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            # Random start
            rng = np.random.RandomState(42 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            # Spherical cap initial: cluster around poles then spread
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(100 + restart)
            points = points + 0.3 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            # Different random seed
            rng = np.random.RandomState(123 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            # Use best found so far with perturbation
            if best_points is not None:
                rng = np.random.RandomState(200 + restart)
                points = best_points + 0.15 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        beta = betas[restart % len(betas)]
        points = gradient_descent_smooth(points, 0.05, 0.99, 200, beta)
        rad = compute_covering_radius(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Local refinement with finer steps and higher beta (sharper max)
    if best_points is not None and time.time() < deadline - 10.0:
        # Use higher beta to sharpen the objective towards the true max
        for trial_beta in [100.0, 200.0]:
            if time.time() > deadline - 5.0:
                break
            candidate = best_points.copy()
            candidate = gradient_descent_smooth(candidate, 0.01, 0.99, 100, trial_beta)
            rad = compute_covering_radius(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()

    # Final polish with very small steps
    if best_points is not None and time.time() < deadline - 3.0:
        final = gradient_descent_smooth(best_points.copy(), 0.001, 0.995, 50, 200.0)
        rad_final = compute_covering_radius(final)
        if rad_final < best_rad:
            best_rad = rad_final
            best_points = final

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
