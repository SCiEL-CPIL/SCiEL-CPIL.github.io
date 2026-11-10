# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.special import softmax

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    timeout = 610  # Safety margin before 630s limit
    n = 14

    def calculate_score(points):
        dists = pdist(points)
        if len(dists) == 0 or np.any(~np.isfinite(dists)):
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def project_to_sphere(points):
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1e-12, norms)
        return points / norms

    def fibonacci_sphere(n_pts):
        golden = (1 + np.sqrt(5)) / 2
        i = np.arange(n_pts)
        z = 1 - (2 * i + 1) / n_pts
        r = np.sqrt(np.maximum(0, 1 - z**2))
        theta = 2 * np.pi * i / golden
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pts = np.column_stack([x, y, z])
        return pts

    def soft_min_max_score(points, alpha=100.0):
        """
        Differentiable surrogate for (d_min/d_max)^2 using soft min/max.
        """
        pts = project_to_sphere(points)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
        dist_matrix = np.sqrt(np.maximum(dist_sq, 1e-20))
        
        mask = ~np.eye(n, dtype=bool)
        dists = dist_matrix[mask]
        
        # Soft min: -logsumexp(-alpha * dists) / alpha
        log_min = -np.log(np.sum(np.exp(-alpha * dists))) / alpha
        # Soft max: logsumexp(alpha * dists) / alpha
        log_max = np.log(np.sum(np.exp(alpha * dists))) / alpha
        
        ratio = np.exp(log_min - log_max)
        return ratio ** 2

    def soft_min_max_grad(points, alpha=100.0):
        """
        Gradient of soft_min_max_score w.r.t. points on sphere.
        """
        pts = project_to_sphere(points)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
        dist_matrix = np.sqrt(np.maximum(dist_sq, 1e-20))
        
        mask = ~np.eye(n, dtype=bool)
        dists = dist_matrix[mask]
        
        # Soft min weights
        w_min = np.exp(-alpha * dists)
        w_min /= np.sum(w_min)
        
        # Soft max weights
        w_max = np.exp(alpha * dists)
        w_max /= np.sum(w_max)
        
        log_min = -np.log(np.sum(np.exp(-alpha * dists))) / alpha
        log_max = np.log(np.sum(np.exp(alpha * dists))) / alpha
        ratio = np.exp(log_min - log_max)
        score = ratio ** 2
        
        # Build gradient
        grad = np.zeros_like(pts)
        
        # For each pair, contribution to soft_min and soft_max
        # d/dp_i of dist_ij = (p_i - p_j) / dist_ij
        # d/dp_i of soft_min = -sum_j w_min_j * (p_i - p_j) / dist_ij * (1/alpha) ... 
        # Actually let's use the chain rule properly.
        # soft_min = -1/alpha * log(sum exp(-alpha * d_ij))
        # d(soft_min)/d(p_i) = -1/alpha * (1/sum(exp(-alpha*d))) * sum_j (-alpha * exp(-alpha*d_ij)) * d(d_ij)/d(p_i)
        # = sum_j w_min_j * d(d_ij)/d(p_i)
        # d(d_ij)/d(p_i) = (p_i - p_j) / d_ij
        
        # soft_max = 1/alpha * log(sum exp(alpha * d_ij))
        # d(soft_max)/d(p_i) = sum_j w_max_j * d(d_ij)/d(p_i)
        
        # score = (soft_min/soft_max)^2
        # d(score)/d(p_i) = 2 * (soft_min/soft_max) * ( (soft_max * d(soft_min)/d(p_i) - soft_min * d(soft_max)/d(p_i)) / soft_max^2 )
        # = 2 * ratio * ( d(soft_min)/d(p_i) / soft_max - soft_min * d(soft_max)/d(p_i) / soft_max^2 )
        # = 2 * ratio * ( d(soft_min)/d(p_i) / soft_max - ratio * d(soft_max)/d(p_i) / soft_max )
        # = 2 * ratio / soft_max * ( d(soft_min)/d(p_i) - ratio * d(soft_max)/d(p_i) )
        
        # We need to accumulate d(soft_min)/d(p_i) and d(soft_max)/d(p_i)
        d_soft_min = np.zeros_like(pts)
        d_soft_max = np.zeros_like(pts)
        
        # Vectorized accumulation
        # For each pair (i,j) with i<j:
        # d(d_ij)/d(p_i) = (p_i - p_j)/d_ij, d(d_ij)/d(p_j) = (p_j - p_i)/d_ij
        # d_soft_min[i] += w_min_ij * (p_i - p_j)/d_ij
        # d_soft_min[j] += w_min_ij * (p_j - p_i)/d_ij
        # Similarly for d_soft_max with w_max
        
        # This is O(n^2) which is fine for n=14
        for i in range(n):
            for j in range(i+1, n):
                vec = pts[i] - pts[j]
                d = np.linalg.norm(vec)
                if d < 1e-12:
                    continue
                unit = vec / d
                
                # Get the weight for this pair
                # Need to map (i,j) to index in dists array
                # dists is flattened upper triangle
                # Index for (i,j) in upper triangle: i*n - (i+1)*(i+2)/2 + j - i - 1 ... 
                # Let's just use the dist_matrix and mask approach differently
                # Actually, let's just recompute the weights per pair is inefficient.
                # Let's stick to the vectorized way if possible or just loop since n=14.
                
                # The weight w_min for pair (i,j) is w_min[idx] where idx is the flat index.
                # Let's create a weight matrix for clarity
                # This is getting complex. Let's use a simpler gradient ascent with finite differences
                # or just use scipy minimize on the soft score directly.
        
        # Let's use scipy minimize on the negative soft score
        return None

    def objective(points_flat):
        """
        Objective for scipy minimize: minimize -soft_min_max_score
        """
        pts = points_flat.reshape(n, 3)
        # We want to maximize soft_min_max_score, so minimize negative
        # But we need to handle the sphere constraint.
        # Better to parameterize differently or use penalty.
        # Let's use a penalty method: points are free in R3, but penalize if norm != 1
        norms = np.linalg.norm(pts, axis=1)
        penalty = np.sum((norms - 1)**2) * 1000.0
        score = soft_min_max_score(pts)
        return -score + penalty

    def objective_grad(points_flat):
        """
        Gradient of objective for scipy minimize.
        """
        pts = points_flat.reshape(n, 3)
        norms = np.linalg.norm(pts, axis=1)
        penalty_grad = 2 * 1000.0 * (norms - 1)[:, np.newaxis] * pts / np.maximum(norms[:, np.newaxis], 1e-12)
        
        # Numerical gradient for soft_min_max_score
        eps = 1e-7
        grad_score = np.zeros_like(pts)
        for i in range(n * 3):
            pts_plus = pts.copy()
            pts_flat_plus = pts_flat.copy()
            pts_flat_plus[i] += eps
            pts_plus = pts_flat_plus.reshape(n, 3)
            f_plus = soft_min_max_score(pts_plus)
            
            pts_minus = pts.copy()
            pts_flat_minus = pts_flat.copy()
            pts_flat_minus[i] -= eps
            pts_minus = pts_flat_minus.reshape(n, 3)
            f_minus = soft_min_max_score(pts_minus)
            
            grad_score.flat[i] = (f_plus - f_minus) / (2 * eps)
        
        # d(-score + penalty)/d(pts) = -grad_score + penalty_grad
        total_grad = -grad_score + penalty_grad
        return total_grad.flatten()

    def local_optimize(points, alpha=100.0, max_iter=2000):
        """
        Use scipy L-BFGS-B to optimize the soft surrogate.
        """
        pts_flat = points.flatten()
        result = minimize(
            objective,
            pts_flat,
            jac=objective_grad,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        best_pts = result.x.reshape(n, 3)
        best_pts = project_to_sphere(best_pts)
        return best_pts, calculate_score(best_pts)

    def simulated_annealing(init_points, n_iter=50000, t_start=1.0, t_end=0.01):
        points = project_to_sphere(init_points.copy())
        current_score = calculate_score(points)
        best_points = points.copy()
        best_score = current_score
        
        n_iter = min(n_iter, max(1000, int((timeout - (time.time() - start_time) - 20) * 100)))
        
        for it in range(n_iter):
            if time.time() - start_time > timeout - 15:
                break
            
            progress = it / max(n_iter - 1, 1)
            T = t_start * (t_end / t_start) ** progress
            
            idx = np.random.randint(n)
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            angle = np.random.randn() * 0.15
            
            p = points[idx]
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            cross = np.cross(axis, p)
            dot = np.dot(axis, p)
            p_new = p * cos_a + cross * sin_a + axis * dot * (1 - cos_a)
            
            new_points = points.copy()
            new_points[idx] = p_new
            new_points = project_to_sphere(new_points)
            
            new_score = calculate_score(new_points)
            delta = new_score - current_score
            
            if delta > 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
        
        return best_points, best_score

    best_score = 0.0
    best_points = None

    # Generate multiple initializations
    inits = []
    inits.append(fibonacci_sphere(n))
    
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]:
        np.random.seed(seed)
        inits.append(np.random.randn(n, 3))

    for seed in [777, 888, 999, 1111, 2222, 3333]:
        np.random.seed(seed)
        inits.append(fibonacci_sphere(n) + np.random.randn(n, 3) * 0.3)

    # Phase 1: Run SA on each initialization
    for init_idx, init in enumerate(inits):
        if time.time() - start_time > timeout - 30:
            break

        pts = project_to_sphere(init.copy())
        s_init = calculate_score(pts)
        if s_init > best_score:
            best_score = s_init
            best_points = pts.copy()

        sa_time_budget = min(20, (timeout - (time.time() - start_time) - 30) / max(len(inits) - init_idx, 1))
        n_sa_iter = int(sa_time_budget * 5000)
        
        pts_sa, s_sa = simulated_annealing(pts, n_iter=max(n_sa_iter, 1000))
        if s_sa > best_score:
            best_score = s_sa
            best_points = pts_sa.copy()

    # Phase 2: Local optimization using L-BFGS-B on soft surrogate
    if best_points is not None and time.time() - start_time < timeout - 10:
        # Try different alpha values
        for alpha in [50, 100, 200]:
            if time.time() - start_time > timeout - 20:
                break
            # Temporarily set alpha for objective
            global_alpha = alpha
            # We need to modify objective to use this alpha
            # Let's create a closure
            def obj_alpha(points_flat):
                pts = points_flat.reshape(n, 3)
                norms = np.linalg.norm(pts, axis=1)
                penalty = np.sum((norms - 1)**2) * 1000.0
                score = soft_min_max_score(pts, alpha=alpha)
                return -score + penalty
            
            def obj_grad_alpha(points_flat):
                pts = points_flat.reshape(n, 3)
                norms = np.linalg.norm(pts, axis=1)
                penalty_grad = 2 * 1000.0 * (norms - 1)[:, np.newaxis] * pts / np.maximum(norms[:, np.newaxis], 1e-12)
                
                eps = 1e-7
                grad_score = np.zeros_like(pts)
                for i in range(n * 3):
                    pts_flat_plus = points_flat.copy()
                    pts_flat_plus[i] += eps
                    pts_plus = pts_flat_plus.reshape(n, 3)
                    f_plus = soft_min_max_score(pts_plus, alpha=alpha)
                    
                    pts_flat_minus = points_flat.copy()
                    pts_flat_minus[i] -= eps
                    pts_minus = pts_flat_minus.reshape(n, 3)
                    f_minus = soft_min_max_score(pts_minus, alpha=alpha)
                    
                    grad_score.flat[i] = (f_plus - f_minus) / (2 * eps)
                
                total_grad = -grad_score + penalty_grad
                return total_grad.flatten()
            
            pts_flat = best_points.flatten()
            result = minimize(
                obj_alpha,
                pts_flat,
                jac=obj_grad_alpha,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            opt_pts = result.x.reshape(n, 3)
            opt_pts = project_to_sphere(opt_pts)
            opt_score = calculate_score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()

    # Phase 3: More SA rounds with the optimized result as seed
    if time.time() - start_time < timeout - 20:
        for round in range(3):
            if time.time() - start_time > timeout - 15:
                break
            n_sa_iter = int(min(30, (timeout - (time.time() - start_time) - 15)) * 3000)
            pts_sa, s_sa = simulated_annealing(best_points, n_iter=max(n_sa_iter, 2000))
            if s_sa > best_score:
                best_score = s_sa
                best_points = pts_sa.copy()
            
            # Polish again with L-BFGS-B
            if time.time() - start_time < timeout - 10:
                for alpha in [100, 200]:
                    if time.time() - start_time > timeout - 20:
                        break
                    def obj_alpha2(points_flat):
                        pts = points_flat.reshape(n, 3)
                        norms = np.linalg.norm(pts, axis=1)
                        penalty = np.sum((norms - 1)**2) * 1000.0
                        score = soft_min_max_score(pts, alpha=alpha)
                        return -score + penalty
                    
                    def obj_grad_alpha2(points_flat):
                        pts = points_flat.reshape(n, 3)
                        norms = np.linalg.norm(pts, axis=1)
                        penalty_grad = 2 * 1000.0 * (norms - 1)[:, np.newaxis] * pts / np.maximum(norms[:, np.newaxis], 1e-12)
                        
                        eps = 1e-7
                        grad_score = np.zeros_like(pts)
                        for i in range(n * 3):
                            pts_flat_plus = points_flat.copy()
                            pts_flat_plus[i] += eps
                            pts_plus = pts_flat_plus.reshape(n, 3)
                            f_plus = soft_min_max_score(pts_plus, alpha=alpha)
                            
                            pts_flat_minus = points_flat.copy()
                            pts_flat_minus[i] -= eps
                            pts_minus = pts_flat_minus.reshape(n, 3)
                            f_minus = soft_min_max_score(pts_minus, alpha=alpha)
                            
                            grad_score.flat[i] = (f_plus - f_minus) / (2 * eps)
                        
                        total_grad = -grad_score + penalty_grad
                        return total_grad.flatten()
                    
                    pts_flat = best_points.flatten()
                    result = minimize(
                        obj_alpha2,
                        pts_flat,
                        jac=obj_grad_alpha2,
                        method='L-BFGS-B',
                        options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
                    )
                    opt_pts = result.x.reshape(n, 3)
                    opt_pts = project_to_sphere(opt_pts)
                    opt_score = calculate_score(opt_pts)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_pts.copy()

    if best_points is None:
        best_points = fibonacci_sphere(n)

    best_points = project_to_sphere(best_points)
    return best_points

# EVOLVE-BLOCK-END
