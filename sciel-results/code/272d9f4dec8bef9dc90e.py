"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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

    def smooth_objective_and_grad(pts, grid_pts, T_min, T_max):
        """
        Compute the smooth approximation of the covering radius and its gradient.
        pts: (N, 3)
        grid_pts: (M, 3)
        Returns: (value, gradient (N, 3))
        """
        # Normalize points to ensure they are on the sphere
        norms_p = np.linalg.norm(pts, axis=1, keepdims=True)
        norms_p[norms_p < 1e-10] = 1e-10
        pts_norm = pts / norms_p
        
        M = len(grid_pts)
        
        # Compute angular distances: M x N
        dot = np.dot(grid_pts, pts_norm.T)  # M x N
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)  # M x N
        
        # Precompute denominator for derivative of arccos
        # sqrt(1 - u^2)
        one_minus_u_sq = 1.0 - dot * dot
        one_minus_u_sq = np.clip(one_minus_u_sq, 1e-12, None)
        sqrt_one_minus_u_sq = np.sqrt(one_minus_u_sq)
        
        # Gradient of arccos(u) w.r.t p_k:
        # d/dp arccos(g.p) = -(g - u*p) / sqrt(1-u^2)
        # We need this for each (j, k) pair.
        # g_j is (3,), p_k is (3,). u_jk is scalar.
        # term_jk = (g_j - u_jk * p_k) / sqrt(1-u_jk^2)
        # Shape: (M, N, 3)
        
        # To save memory, we might compute this in chunks or use broadcasting carefully.
        # grid_pts: (M, 1, 3)
        # pts_norm: (1, N, 3)
        # dot: (M, N, 1)
        
        g_exp = grid_pts[:, np.newaxis, :]  # (M, 1, 3)
        p_exp = pts_norm[np.newaxis, :, :]  # (1, N, 3)
        u_exp = dot[:, :, np.newaxis]       # (M, N, 1)
        denom_exp = sqrt_one_minus_u_sq[:, :, np.newaxis] # (M, N, 1)
        
        # grad_d_jk = - (g_j - u_jk * p_k) / denom_jk
        grad_d = -(g_exp - u_exp * p_exp) / denom_exp  # (M, N, 3)
        
        # Soft-min over points k for each grid point j
        # s_j = -1/T_min * log( sum_k exp(-T_min * d_jk) )
        
        exp_neg_T_min_d = np.exp(-T_min * ang_dists)  # (M, N)
        sum_exp_neg = np.sum(exp_neg_T_min_d, axis=1)  # (M,)
        
        # Avoid log(0)
        sum_exp_neg = np.clip(sum_exp_neg, 1e-15, None)
        
        soft_min_dists = -1.0 / T_min * np.log(sum_exp_neg)  # (M,)
        
        # Weights for soft-min: w_jk = exp(-T_min * d_jk) / sum_k exp(-T_min * d_jk)
        w_jk = exp_neg_T_min_d / sum_exp_neg[:, np.newaxis]  # (M, N)
        
        # Gradient of s_j w.r.t p_k:
        # ds_j/dp_k = w_jk * (d d_jk / d p_k)
        # Sum over j for the final gradient of L w.r.t p_k
        
        # Soft-max over grid points j
        # L = 1/T_max * log( sum_j exp(T_max * s_j) )
        
        exp_T_max_s = np.exp(T_max * soft_min_dists)  # (M,)
        sum_exp_T_max = np.sum(exp_T_max_s)
        
        # Avoid log(0)
        sum_exp_T_max = np.clip(sum_exp_T_max, 1e-15, None)
        
        value = 1.0 / T_max * np.log(sum_exp_T_max)
        
        # Weights for soft-max: v_j = exp(T_max * s_j) / sum_i exp(T_max * s_i)
        v_j = exp_T_max_s / sum_exp_T_max  # (M,)
        
        # Total gradient w.r.t p_k:
        # dL/dp_k = sum_j v_j * (ds_j/dp_k)
        # ds_j/dp_k = w_jk * grad_d_jk
        # dL/dp_k = sum_j v_j * w_jk * grad_d_jk
        
        # v_j: (M, 1)
        # w_jk: (M, N)
        # grad_d: (M, N, 3)
        
        # coeff_jk = v_j * w_jk  (M, N)
        coeff_jk = v_j[:, np.newaxis] * w_jk  # (M, N)
        
        # grad_L_k = sum_j coeff_jk * grad_d_jk
        # grad_d_jk is (M, N, 3). coeff_jk is (M, N, 1).
        grad_L = np.sum(coeff_jk[:, :, np.newaxis] * grad_d, axis=0)  # (N, 3)
        
        return float(value), grad_L

    def gradient_descent_smooth(pts, grid_pts, max_iter, time_limit):
        """
        Optimize points using gradient descent on the smooth objective.
        """
        T_min = 20.0
        T_max = 20.0
        alpha = 0.05
        decay = 0.995
        
        best_val = np.inf
        best_pts = pts.copy()
        
        # Adam optimizer state
        m = np.zeros_like(pts)
        v = np.zeros_like(pts)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        t = 0
        
        for it in range(max_iter):
            if time.time() > time_limit:
                break
            
            val, grad = smooth_objective_and_grad(pts, grid_pts, T_min, T_max)
            
            t += 1
            
            # Adam update
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad * grad)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            step = alpha * m_hat / (np.sqrt(v_hat) + eps)
            
            # Update points
            pts = pts - step
            
            # Project to unit sphere
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            
            if val < best_val:
                best_val = val
                best_pts = pts.copy()
            
            # Anneal temperatures slightly to sharpen the objective as we converge
            # This helps in the final stages to resolve the exact max/min
            if it % 10 == 0:
                T_min *= 1.05
                T_max *= 1.05
                alpha *= decay
                
            # Add small repulsion to prevent clustering if they get too close
            # Check min distance
            dot = np.dot(pts, pts.T)
            np.fill_diagonal(dot, 2.0)
            min_dist_sq = np.min(2.0 - dot)
            if min_dist_sq < 0.01: # ~0.1 rad
                # Apply repulsion
                rep_grad = np.zeros_like(pts)
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            d2 = 2.0 - 2.0 * np.dot(pts[i], pts[j])
                            if d2 < 0.01:
                                rep_grad[i] += (pts[i] - pts[j]) / max(d2, 1e-6)
                pts = pts + 0.01 * rep_grad
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                pts = pts / norms

        return best_pts, best_val

    # Precompute a grid for the smooth objective
    # 3000 points is a good balance between accuracy and speed
    grid_pts = fibonacci_spiral(3000)
    
    best_points = None
    best_rad = np.inf
    
    n_restarts = 4
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            points = np.random.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            # Perturb the best found so far
            points = best_points + 0.15 * np.random.randn(N, 3)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
        
        # Run gradient descent
        # Allocate most of the time to the first restart, less to subsequent ones
        time_alloc = 100.0 if restart == 0 else 30.0
        max_iter = 500
        
        points_opt, val_opt = gradient_descent_smooth(points, grid_pts, max_iter, start_time + time_alloc)
        
        # Evaluate exact covering radius
        rad = compute_covering_radius_exact(points_opt)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points_opt.copy()
            
        # Polish step: Run a few more iterations with very small alpha and higher temperatures
        # to fine-tune the result
        if time.time() < deadline - 10.0:
            # Re-initialize temperatures for polish
            T_min_polish = 50.0
            T_max_polish = 50.0
            alpha_polish = 0.005
            
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                val, grad = smooth_objective_and_grad(best_points, grid_pts, T_min_polish, T_max_polish)
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1e-10:
                    grad = grad / grad_norm
                best_points = best_points - alpha_polish * grad
                norms = np.linalg.norm(best_points, axis=1, keepdims=True)
                best_points = best_points / norms
                alpha_polish *= 0.95
                
            rad_polished = compute_covering_radius_exact(best_points)
            if rad_polished < best_rad:
                best_rad = rad_polished
                best_points = best_points.copy()

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)
    
    # Ensure all points are on the unit sphere
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points

# EVOLVE-BLOCK-END
