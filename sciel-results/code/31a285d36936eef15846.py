# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or d_min == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _objective(points):
    """
    Returns -log(d_min) + log(d_max). Minimizing this maximizes log(d_min/d_max).
    This is a smooth, scale-invariant objective suitable for gradient ascent.
    """
    dists = pdist(points)
    if np.any(dists <= 0):
        return 1e10
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 1e10
    return -np.log(d_min) + np.log(d_max)

def _gradient(points):
    """
    Compute gradient of the objective with respect to points.
    Objective: f = -log(d_min) + log(d_max)
    df/dp_i = - (1/d_min) * dd_min/dp_i + (1/d_max) * dd_max/dp_i
    
    dd/dp_i = sum_{j != k, d_jk = d_target} (p_i - p_j)/d_target + sum_{k != j, d_jk = d_target} (p_i - p_k)/d_target
    Actually, for a specific pair (j,k) with distance d_jk:
    d(d_jk)/d(p_j) = (p_j - p_k)/d_jk
    d(d_jk)/d(p_k) = (p_k - p_j)/d_jk
    
    So, gradient contribution of pair (j,k) to point j is (p_j - p_k)/d_jk * (df/dd_jk)
    and to point k is (p_k - p_j)/d_jk * (df/dd_jk)
    
    df/dd_jk = -1/d_jk if d_jk == d_min, +1/d_jk if d_jk == d_max, 0 otherwise.
    """
    n = points.shape[0]
    grad = np.zeros_like(points)
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_matrix = np.linalg.norm(diff, axis=2)
    
    d_min = np.min(dists_matrix[np.triu_indices(n, k=1)])
    d_max = np.max(dists_matrix[np.triu_indices(n, k=1)])
    
    if d_min == 0 or d_max == 0:
        return grad
    
    # We need to sum contributions from pairs that are at min or max distance.
    # To be robust to floating point, we consider pairs within a small tolerance of min/max.
    # However, strictly speaking, the gradient is only non-zero for the pairs that achieve the min/max.
    # If there are multiple, we sum them.
    
    # Tolerance for identifying "min" and "max" pairs
    tol = 1e-8
    
    # Coefficients for each pair
    # coeff[i,j] is the factor multiplying (p_i - p_j) in the gradient for p_i
    # grad_p_i += coeff[i,j] * (p_i - p_j) / dist(i,j)
    
    # Let's just iterate over all pairs for n=14, it's fast enough.
    for i in range(n):
        for j in range(i + 1, n):
            d_ij = dists_matrix[i, j]
            if d_ij < 1e-10:
                continue
            
            coef = 0.0
            if abs(d_ij - d_min) < tol:
                coef += -1.0 / d_ij
            if abs(d_ij - d_max) < tol:
                coef += 1.0 / d_ij
            
            if coef != 0.0:
                # Gradient for p_i: coef * (p_i - p_j) / d_ij
                # Gradient for p_j: coef * (p_j - p_i) / d_ij
                vec = (points[i] - points[j]) / d_ij
                grad[i] += coef * vec
                grad[j] -= coef * vec
                
    return grad

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    start_time = time.time()
    deadline = start_time + 620  # Safety margin
    
    best_score = 0.0
    best_points = None
    
    n = 14
    
    def fibonacci_sphere(n):
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = 1 - 2 * (i + 0.5) / n
            r = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / golden_ratio
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)

    def project_to_sphere(points):
        # Center
        points -= np.mean(points, axis=0, keepdims=True)
        # Normalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        points /= norms
        return points

    # Multi-start strategy
    num_starts = 20
    max_iters = 2000
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 5:
            break
        
        # Different initializations
        if start_idx == 0:
            points = fibonacci_sphere(n)
        elif start_idx == 1:
            points = np.random.randn(n, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            # Perturbed fibonacci sphere
            points = fibonacci_sphere(n)
            points += np.random.randn(n, 3) * (0.1 + 0.1 * (start_idx % 5))
            points = project_to_sphere(points)
        
        # Simulated Annealing Gradient Ascent
        current_points = points.copy()
        T = 1.0 # Initial temperature
        lr = 0.05
        
        for i in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            # Compute gradient of objective (we want to minimize, so we descend)
            # Or compute gradient of -objective and ascend.
            # _gradient returns grad of f = -log(dmin) + log(dmax). We want to minimize f.
            grad = _gradient(current_points)
            
            # Normalize gradient to have unit norm to make lr scale-invariant
            g_norm = np.linalg.norm(grad)
            if g_norm > 1e-10:
                grad /= g_norm
            
            # Propose new point
            # Move in direction of negative gradient
            proposed = current_points - lr * grad
            
            # Project to sphere
            proposed = project_to_sphere(proposed)
            
            # Calculate objective change
            obj_curr = _objective(current_points)
            obj_prop = _objective(proposed)
            
            delta_obj = obj_prop - obj_curr
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta_obj < 0:
                current_points = proposed
            else:
                if T > 1e-6:
                    prob = np.exp(-delta_obj / T)
                    if np.random.rand() < prob:
                        current_points = proposed
                
            # Cooling
            T *= 0.999
            # Adaptive learning rate
            if i % 100 == 0:
                lr *= 0.95
                if lr < 1e-5:
                    lr = 1e-5

        # Evaluate final score
        s = _score(current_points)
        if s > best_score:
            best_score = s
            best_points = current_points.copy()
            
    # Final polish on the best found using pure gradient descent (no noise)
    if best_points is not None:
        T = 0.0
        lr = 0.01
        for i in range(2000):
            if time.time() > deadline - 1:
                break
            
            grad = _gradient(best_points)
            g_norm = np.linalg.norm(grad)
            if g_norm < 1e-10:
                break
            grad /= g_norm
            
            proposed = best_points - lr * grad
            proposed = project_to_sphere(proposed)
            
            obj_curr = _objective(best_points)
            obj_prop = _objective(proposed)
            
            if obj_prop < obj_curr:
                best_points = proposed
            else:
                # Try smaller step
                lr *= 0.5
                if lr < 1e-8:
                    break
            
            # Reset lr occasionally if stuck? No, just keep decreasing.
            # But we want to converge.
            
        # Update score
        s = _score(best_points)
        if s > best_score:
            best_score = s

    if best_points is None:
        # Fallback
        best_points = fibonacci_sphere(n)
        
    return best_points

# EVOLVE-BLOCK-END
