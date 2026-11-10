# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

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

    # Smooth proxy objective to maximize (minimize inverse sum)
    # Minimizing sum(1/d^k) approximates maximizing d_min
    k_power = 8.0

    def compute_energy(points):
        """Compute the sum of inverse distances raised to power k_power."""
        dists = pdist(points)
        # Avoid division by zero
        dists = np.maximum(dists, 1e-12)
        return np.sum(1.0 / (dists ** k_power))

    def compute_gradient(points):
        """
        Compute gradient of the energy function w.r.t. points.
        E = sum_{i<j} d_ij^{-k}
        dE/dp_i = -k * sum_{j != i} (p_i - p_j) * d_ij^{-k-2}
        """
        n_pts = points.shape[0]
        grads = np.zeros_like(points)
        # Vectorized calculation of pairwise differences and distances
        # diff[i, j] = p_i - p_j
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        # Set diagonal to 1 to avoid division by zero, will mask later
        np.fill_diagonal(dists_sq, 1.0)
        dists = np.sqrt(dists_sq)
        
        # We want term: (p_i - p_j) / d_ij^(k+2)
        # dists_inv = 1 / d_ij^(k+2)
        dists_inv = 1.0 / (dists ** (k_power + 2))
        
        # Multiply diff by dists_inv (broadcasting)
        # shape: (n, n, 3) * (n, n) -> (n, n, 3)
        terms = diff * dists_inv[:, :, np.newaxis]
        
        # Sum over j (axis=1) to get gradient for each i
        # Note: The formula is -k * sum_j ...
        grads = -k_power * np.sum(terms, axis=1)
        
        return grads

    def project_to_tangent(points, grads):
        """Project gradients onto the tangent plane of the unit sphere."""
        # grads = g - (g . p) p
        dots = np.sum(grads * points, axis=1, keepdims=True)
        tangent_grads = grads - dots * points
        return tangent_grads

    def normalize(points):
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return points / norms

    def score(points):
        """Calculate the actual metric (d_min / d_max)^2."""
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0 or d_min == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def fib_sphere(n):
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = phi * i  # golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    best_points = None
    best_score = -1.0

    # Initializations
    inits = []
    # 1. Fibonacci Sphere
    inits.append(fib_sphere(n))
    
    # 2. Random Sphere Points
    np.random.seed(42)
    for _ in range(10):
        pts = np.random.randn(n, d)
        inits.append(normalize(pts))

    # 3. Perturbed Fibonacci
    for _ in range(10):
        pts = fib_sphere(n) + np.random.randn(n, d) * 0.1
        inits.append(normalize(pts))

    for init in inits:
        if time.time() - start_time > timeout - 30:
            break
            
        points = init.copy()
        current_energy = compute_energy(points)
        
        # Simulated Annealing / Gradient Descent loop
        # Start with a larger step size, cool down
        lr = 0.01
        min_lr = 1e-8
        cooling_rate = 0.999
        stuck_count = 0
        max_stuck = 200
        
        for step in range(10000):
            if time.time() - start_time > timeout - 10:
                break
                
            # Compute gradient
            grads = compute_gradient(points)
            tangent_grads = project_to_tangent(points, grads)
            
            # Update
            points = points - lr * tangent_grads
            points = normalize(points)
            
            new_energy = compute_energy(points)
            
            # Accept if better, or with probability exp(-delta/E) (Metropolis)
            # Here we are minimizing energy.
            if new_energy < current_energy:
                current_energy = new_energy
                stuck_count = 0
            else:
                stuck_count += 1
                # If stuck, try a random perturbation to escape local minimum
                if stuck_count > max_stuck:
                    # Random kick
                    kick_strength = 0.1
                    points = points + np.random.randn(n, d) * kick_strength
                    points = normalize(points)
                    current_energy = compute_energy(points)
                    stuck_count = 0
                    lr = 0.01 # Reset learning rate
                    continue

            # Cool down learning rate
            lr *= cooling_rate
            if lr < min_lr:
                lr = min_lr
                
            # Check score periodically
            if step % 100 == 0:
                s = score(points)
                if s > best_score:
                    best_score = s
                    best_points = points.copy()

    # Final polish on the best points found using Nelder-Mead on the exact objective
    if best_points is not None:
        from scipy.optimize import minimize
        
        def exact_obj(x):
            pts = x.reshape(n, d)
            pts = normalize(pts)
            s = score(pts)
            return -s

        x0 = best_points.flatten()
        try:
            result = minimize(exact_obj, x0, method='Nelder-Mead', 
                              options={'xatol': 1e-10, 'fatol': 1e-10, 
                                       'maxiter': 5000, 'maxfev': 10000,
                                       'adaptive': True})
            res_points = normalize(result.x.reshape(n, d))
            res_score = score(res_points)
            if res_score > best_score:
                best_score = res_score
                best_points = res_points
        except Exception:
            pass

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
