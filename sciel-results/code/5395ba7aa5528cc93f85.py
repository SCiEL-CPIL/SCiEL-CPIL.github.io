# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    n = 14
    d = 3
    start_time = time.time()
    timeout = 580.0  # Safety margin before 630s limit

    def true_score(points):
        """Compute the actual score (d_min/d_max)^2"""
        points = points.reshape(n, d)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        points = points / norms
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-15:
            return 0.0
        return (d_min / d_max) ** 2

    def log_sum_dist_obj(x):
        """
        Smooth proxy: minimize -sum(log(dist_ij)) over all pairs.
        This is equivalent to maximizing the geometric mean of all pairwise distances.
        """
        if time.time() - start_time > timeout:
            raise TimeoutError()
        pts = x.reshape(n, d)
        # Normalize to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms
        
        # Compute all pairwise distances
        dists = pdist(pts)
        # Avoid log(0)
        dists = np.maximum(dists, 1e-12)
        return -np.sum(np.log(dists))

    def log_sum_dist_grad(x):
        """
        Gradient of -sum(log(dist_ij)) with respect to point positions.
        For each pair (i,j), d_ij = ||p_i - p_j||
        d(-log(d_ij))/dp_i = -(p_i - p_j) / d_ij^2
        d(-log(d_ij))/dp_j = (p_i - p_j) / d_ij^2
        """
        pts = x.reshape(n, d)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms
        
        grad = np.zeros_like(pts)
        for i in range(n):
            for j in range(i+1, n):
                diff = pts[i] - pts[j]
                d2 = np.dot(diff, diff)
                if d2 < 1e-20:
                    continue
                # d(-log(d))/dp_i = -diff/d2
                grad[i] -= diff / d2
                # d(-log(d))/dp_j = diff/d2
                grad[j] += diff / d2
        
        return grad.flatten()

    def sphere_retraction(x):
        """Project points onto unit sphere"""
        pts = x.reshape(n, d)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms
        return pts.flatten()

    def sphere_gradient_descent(x0, steps=5000, lr=0.05):
        """
        Gradient descent on the sphere for the log-sum-dist objective.
        Uses retraction to stay on the sphere.
        """
        x = x0.copy()
        best_x = x.copy()
        best_val = log_sum_dist_obj(x)
        
        for step in range(steps):
            if time.time() - start_time > timeout - 30:
                break
            if step % 500 == 0:
                val = log_sum_dist_obj(x)
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
            
            g = log_sum_dist_grad(x)
            # Adaptive learning rate: decrease if oscillating
            x_new = x - lr * g
            x_new = sphere_retraction(x_new)
            
            # Check for improvement
            val_new = log_sum_dist_obj(x_new)
            if val_new < log_sum_dist_obj(x):
                x = x_new
                # Slowly decay learning rate
                if step > 1000:
                    lr *= 0.999
            else:
                # Reduce learning rate and try smaller step
                lr *= 0.5
                x_new = x - lr * g
                x_new = sphere_retraction(x_new)
                x = x_new
        
        return best_x

    def fib_sphere(n_pts):
        """Fibonacci sphere distribution"""
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n_pts):
            y = 1 - (i / float(n_pts - 1)) * 2
            radius = np.sqrt(max(0, 1 - y * y))
            theta = phi * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    def random_sphere(n_pts, seed):
        """Random points on sphere"""
        rng = np.random.RandomState(seed)
        pts = rng.randn(n_pts, d)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        return pts

    best_points = None
    best_score = 0.0

    # Phase 1: Use smooth proxy with gradient descent from multiple starts
    starts = [fib_sphere(n)]
    for seed in range(20):
        starts.append(random_sphere(n, seed))
    
    for idx, init_pts in enumerate(starts):
        if time.time() - start_time > timeout - 60:
            break
        
        x0 = init_pts.flatten()
        
        # Run gradient descent on sphere
        try:
            x_gd = sphere_gradient_descent(x0, steps=3000, lr=0.03)
            candidate = x_gd.reshape(n, d)
            s = true_score(candidate)
            if s > best_score:
                best_score = s
                best_points = candidate
        except Exception:
            pass
        
        # Also try BFGS from the gradient descent result
        if time.time() - start_time > timeout - 40:
            break
        try:
            result = minimize(
                log_sum_dist_obj, x_gd,
                method='BFGS',
                jac=log_sum_dist_grad,
                options={'gtol': 1e-8, 'maxiter': 500}
            )
            if result.fun < 1e5:
                # Project back to sphere
                candidate = result.x.reshape(n, d)
                norms = np.linalg.norm(candidate, axis=1, keepdims=True)
                norms[norms < 1e-10] = 1.0
                candidate = candidate / norms
                s = true_score(candidate)
                if s > best_score:
                    best_score = s
                    best_points = candidate
        except Exception:
            pass

    # Phase 2: Fine-tune the best result using the true objective with Nelder-Mead
    if best_points is not None and time.time() - start_time < timeout - 20:
        x0 = best_points.flatten()
        
        def true_obj(x):
            if time.time() - start_time > timeout:
                raise TimeoutError()
            pts = x.reshape(n, d)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            pts = pts / norms
            dists = pdist(pts)
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max < 1e-15 or d_min < 1e-15:
                return 1e6
            return -(d_min / d_max) ** 2
        
        try:
            result = minimize(
                true_obj, x0, method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 5000, 'maxfev': 10000}
            )
            if result.fun < 1e5:
                candidate = result.x.reshape(n, d)
                norms = np.linalg.norm(candidate, axis=1, keepdims=True)
                norms[norms < 1e-10] = 1.0
                candidate = candidate / norms
                s = true_score(candidate)
                if s > best_score:
                    best_score = s
                    best_points = candidate
        except Exception:
            pass

    # Phase 3: Final fine-tuning with very tight tolerances
    if best_points is not None and time.time() - start_time < timeout - 10:
        x0 = best_points.flatten()
        try:
            result = minimize(
                true_obj, x0, method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 10000, 'maxfev': 20000}
            )
            if result.fun < 1e5:
                candidate = result.x.reshape(n, d)
                norms = np.linalg.norm(candidate, axis=1, keepdims=True)
                norms[norms < 1e-10] = 1.0
                candidate = candidate / norms
                s = true_score(candidate)
                if s > best_score:
                    best_score = s
                    best_points = candidate
        except Exception:
            pass

    # Final: if no good result, use Fibonacci sphere
    if best_points is None:
        best_points = fib_sphere(n)

    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        best_points = fib_sphere(n)

    return best_points

# EVOLVE-BLOCK-END
