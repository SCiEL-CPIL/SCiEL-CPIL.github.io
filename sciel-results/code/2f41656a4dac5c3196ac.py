# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3
    start_time = time.time()
    timeout = 590.0  # Safety margin before 630s limit

    def normalize(points):
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return points / norms

    def compute_score(points):
        """Compute the actual metric: (d_min / d_max)^2"""
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0 or d_min == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def fib_sphere(n):
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    def rotate_points(points, angles):
        """Rotate points by given angles around x, y, z axes."""
        pts = points.copy()
        for axis, angle in zip(range(3), angles):
            c, s = np.cos(angle), np.sin(angle)
            if axis == 0:  # x-axis
                rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            elif axis == 1:  # y-axis
                rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            else:  # z-axis
                rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            pts = pts @ rot.T
        return pts

    def gradient_descent_sphere(points, iterations=5000, lr=0.01, momentum=0.9, k_power=8.0):
        """
        Minimizes sum(1/d_ij^k) on the sphere using gradient descent with momentum.
        This is a smooth proxy for maximizing d_min.
        """
        pts = points.copy()
        v = np.zeros_like(pts)
        
        for _ in range(iterations):
            if time.time() - start_time > timeout - 10:
                break
                
            # Compute pairwise differences and distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists_sq = np.sum(diff ** 2, axis=2)
            
            # Avoid division by zero
            np.fill_diagonal(dists_sq, 1.0)
            dists = np.sqrt(dists_sq)
            
            # Gradient of E = sum_{i<j} d_ij^{-k}
            # dE/dp_i = -k * sum_{j != i} (p_i - p_j) * d_ij^{-k-2}
            
            dists_inv = 1.0 / (dists ** (k_power + 2))
            
            # Set diagonal to 0
            np.fill_diagonal(dists_inv, 0.0)
            
            # terms[i, j] = (p_i - p_j) * dists_inv[i, j]
            terms = diff * dists_inv[:, :, np.newaxis]
            
            # grad_i = -k * sum_j terms[i, j]
            grads = -k_power * np.sum(terms, axis=1)
            
            # Update velocity with momentum
            v = momentum * v - lr * grads
            
            # Update positions
            pts = pts + v
            
            # Project back to sphere
            pts = normalize(pts)
            
        return pts

    def refine_with_nm(points, maxfev=3000):
        """Refine using Nelder-Mead on the actual objective."""
        def objective(x):
            if time.time() - start_time > timeout - 5:
                raise TimeoutError()
            pts = x.reshape(n, d)
            pts = normalize(pts)
            dists = pdist(pts)
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max == 0 or d_min == 0:
                return 1e6
            return - (d_min / d_max) ** 2

        try:
            res = minimize(objective, points.flatten(), method='Nelder-Mead',
                           options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1500, 'maxfev': maxfev,
                                    'adaptive': True})
            res_pts = normalize(res.x.reshape(n, d))
            return res_pts
        except Exception:
            return points

    def scaled_nm_restart(points, scale=0.05):
        """Try NM restarts from perturbed versions of the best point."""
        best_pts = points.copy()
        best_score = compute_score(best_pts)
        
        np.random.seed(123)
        for _ in range(5):
            if time.time() - start_time > timeout - 15:
                break
            
            perturbed = normalize(best_pts + np.random.randn(n, d) * scale)
            refined = refine_with_nm(perturbed, maxfev=1000)
            s = compute_score(refined)
            
            if s > best_score:
                best_score = s
                best_pts = refined
        
        return best_pts

    best_points = None
    best_score = -1.0

    # Initializations - more diverse
    inits = []
    
    # 1. Fibonacci Sphere
    inits.append(fib_sphere(n))
    
    # 2. Random Sphere Points
    np.random.seed(42)
    for _ in range(12):
        pts = np.random.randn(n, d)
        inits.append(normalize(pts))
    
    # 3. Perturbed Fibonacci with varied scales
    for scale in [0.05, 0.1, 0.2, 0.3]:
        pts = fib_sphere(n) + np.random.randn(n, d) * scale
        inits.append(normalize(pts))
    
    # 4. Rotated Fibonacci
    for _ in range(8):
        angles = np.random.uniform(-np.pi, np.pi, 3)
        pts = rotate_points(fib_sphere(n), angles)
        inits.append(normalize(pts))
    
    # 5. Rotated + Perturbed Fibonacci
    for _ in range(8):
        angles = np.random.uniform(-np.pi, np.pi, 3)
        pts = rotate_points(fib_sphere(n), angles) + np.random.randn(n, d) * 0.1
        inits.append(normalize(pts))

    # Strategy:
    # 1. Run gradient descent on the smooth proxy for each init with multiple configs.
    # 2. Keep the best few results.
    # 3. Run Nelder-Mead on the exact objective for the best few.
    # 4. Try scaled NM restarts from the best.
    # 5. Return the overall best.

    candidates = []

    # Expanded parameter grid
    lr_values = [0.003, 0.005, 0.01, 0.015, 0.02, 0.03]
    k_values = [2.0, 4.0, 8.0, 12.0, 16.0]

    for init in inits:
        if time.time() - start_time > timeout - 90:
            break
            
        for lr in lr_values:
            if time.time() - start_time > timeout - 60:
                break
                
            for k in k_values:
                if time.time() - start_time > timeout - 30:
                    break
                    
                # Phase 1: Coarse GD
                pts = gradient_descent_sphere(init, iterations=1500, lr=lr, momentum=0.9, k_power=k)
                
                # Phase 2: Fine GD
                pts = gradient_descent_sphere(pts, iterations=1500, lr=lr*0.1, momentum=0.9, k_power=k)
                
                s = compute_score(pts)
                candidates.append((s, pts))
                
    # Sort candidates by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Keep top 8 unique-ish candidates for NM refinement
    top_candidates = []
    seen = []
    for s, pts in candidates:
        if time.time() - start_time > timeout - 40:
            break
            
        # Simple check to avoid duplicate starting points for NM
        is_dup = False
        for _, prev_pts in top_candidates:
            if np.max(np.abs(pts - prev_pts)) < 1e-4:
                is_dup = True
                break
        
        if not is_dup:
            top_candidates.append((s, pts))
            
        if len(top_candidates) >= 8:
            break

    # Refine top candidates with Nelder-Mead
    for s, pts in top_candidates:
        if time.time() - start_time > timeout - 20:
            break
            
        refined_pts = refine_with_nm(pts)
        refined_score = compute_score(refined_pts)
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts

    # Try scaled NM restarts from the best found so far
    if best_points is not None and time.time() - start_time < timeout - 20:
        restarted_pts = scaled_nm_restart(best_points)
        restarted_score = compute_score(restarted_pts)
        
        if restarted_score > best_score:
            best_score = restarted_score
            best_points = restarted_pts

    # Fallback
    if best_points is None:
        best_points = fib_sphere(n)
        best_points = normalize(best_points)

    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        best_points = fib_sphere(n)
        best_points = normalize(best_points)

    return best_points

# EVOLVE-BLOCK-END
