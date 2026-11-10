# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

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
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _objective_and_grad(x, time_limit):
    """
    Objective function and gradient for minimization.
    We minimize -log(d_min / d_max) = log(d_max) - log(d_min).
    This is scale and translation invariant.
    """
    points = x.reshape(14, 3)
    
    # Check time limit
    if time.time() > time_limit:
        return 1e6, np.zeros_like(x)
        
    if not np.all(np.isfinite(points)):
        return 1e6, np.zeros_like(x)
        
    # Compute pairwise distances
    dists = pdist(points)
    if len(dists) == 0:
        return 1e6, np.zeros_likex
        
    d_min = np.min(dists)
    d_max = np.max(dists)
    
    if d_min <= 1e-12 or d_max <= 1e-12:
        return 1e6, np.zeros_like(x)
        
    val = np.log(d_max) - np.log(d_min)
    
    # Compute gradient
    # We need the indices of the pairs corresponding to d_min and d_max
    i_min = np.argmin(dists)
    i_max = np.argmax(dists)
    
    # Map flat index to (i, j) pair
    def flat_to_pair(k, n):
        # pdist ordering: (0,1), (0,2), ..., (0,n-1), (1,2), ...
        # j = k + 1
        # i = (k - j) / (n - j - 1) ? No.
        # Let's use the standard inverse mapping
        # For n points, the index k corresponds to:
        # j = k + 1
        # i = (k - j) // (n - j - 1) ... this is tricky.
        
        # Simpler: generate all pairs
        # But for n=14, 91 pairs, it's fast enough to do it once or use a precomputed map.
        # Let's just compute it efficiently.
        # The number of pairs before the one starting with i is i*(2*n - i - 1)/2
        # Actually, let's just loop or use a lookup table generated once.
        pass

    # Precomputed lookup for 14 points
    # We can't easily precompute inside this function if it's called many times without global state.
    # But 91 pairs is small.
    # Let's derive i, j from k.
    # The sequence is:
    # 0: (0,1)
    # 1: (0,2)
    # ...
    # n-2: (0, n-1)
    # n-1: (1,2)
    
    n = 14
    # We can solve for i and j
    # k = i*(2*n - i - 1)/2 + (j - i - 1)
    
    def get_pair(k):
        # Binary search or direct formula
        # i*(2n - i - 1)/2 <= k
        # Let's just iterate i from 0 to n-2
        # Since n is small, this is fine.
        for i in range(n - 1):
            # Max k for this i is i*(2*n - i - 1)/2 + (n - i - 2)
            # Min k for this i is i*(2*n - i - 1)/2
            start_k = i * (2 * n - i - 1) // 2
            if k >= start_k:
                # j = i + 1 + (k - start_k)
                j = i + 1 + (k - start_k)
                return i, j
        return 0, 1 # Should not happen
        
    i1, j1 = get_pair(i_min)
    i2, j2 = get_pair(i_max)
    
    # Gradient of log(d) w.r.t points
    # d = ||p_i - p_j||
    # d(log d)/d(p_i) = (p_i - p_j) / d^2
    # d(log d)/d(p_j) = (p_j - p_i) / d^2
    
    grad = np.zeros_like(x)
    
    # Contribution from d_min (we are minimizing, so we subtract gradient of log(d_min))
    # Objective is log(d_max) - log(d_min)
    # grad_obj = grad_log_dmax - grad_log_dmin
    
    vec_min = points[i1] - points[j1]
    d_min_sq = d_min ** 2
    grad_log_dmin_i1 = vec_min / d_min_sq
    grad_log_dmin_j1 = -vec_min / d_min_sq
    
    vec_max = points[i2] - points[j2]
    d_max_sq = d_max ** 2
    grad_log_dmax_i2 = vec_max / d_max_sq
    grad_log_dmax_j2 = -vec_max / d_max_sq
    
    grad[i1] -= grad_log_dmin_i1
    grad[j1] -= grad_log_dmin_j1
    
    grad[i2] += grad_log_dmax_i2
    grad[j2] += grad_log_dmax_j2
    
    return val, grad.flatten()

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    start_time = time.time()
    deadline = start_time + 600  # Safety margin
    
    n = 14
    best_score = 0.0
    best_points = None
    
    # Initial guess: Fibonacci sphere
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
    
    fib_points = fibonacci_sphere(n)
    # Normalize to unit sphere and center
    fib_points = fib_points - np.mean(fib_points, axis=0)
    norms = np.linalg.norm(fib_points, axis=1)
    fib_points = fib_points / norms[:, np.newaxis]
    
    best_points = fib_points.copy()
    best_score = _score(best_points)
    
    # Multi-start optimization
    # We will run SLSQP from multiple random starting points on the unit sphere.
    # Constraints:
    # 1. Centroid is 0: sum(p_i) = 0
    # 2. Norm is 1: sum(||p_i||^2) = 1 (or each ||p_i||=1? No, we just want scale invariance.
    #    Fixing sum(||p_i||^2) = 1 fixes scale.
    #    Actually, fixing each point to unit sphere is a different manifold.
    #    Let's fix sum(||p_i||^2) = 1. This is a single equality constraint.
    #    And sum(p_i) = 0. 3 equality constraints.
    
    def constraints_centroid(x):
        p = x.reshape(14, 3)
        return np.sum(p, axis=0)
        
    def constraints_scale(x):
        p = x.reshape(14, 3)
        return np.sum(np.sum(p**2, axis=1)) - 1.0
        
    cons = [
        {'type': 'eq', 'fun': constraints_centroid},
        {'type': 'eq', 'fun': constraints_scale}
    ]
    
    # Generate random initial points
    rng = np.random.default_rng(42)
    
    num_restarts = 0
    while time.time() < deadline - 10:
        # Create a random point on the sphere (approx)
        # Random normal, then normalize to satisfy constraints roughly
        x0 = rng.normal(size=(n, 3))
        x0 = x0 - np.mean(x0, axis=0)
        x0_norm = np.sqrt(np.sum(np.sum(x0**2, axis=1)))
        x0 = x0 / x0_norm
        
        # Run SLSQP
        try:
            res = minimize(
                lambda x: _objective_and_grad(x, deadline)[0],
                x0.flatten(),
                jac=lambda x: _objective_and_grad(x, deadline)[1],
                method='SLSQP',
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.x is not None:
                candidate = res.x.reshape(n, 3)
                # Re-center and re-scale to ensure constraints are met precisely
                candidate = candidate - np.mean(candidate, axis=0)
                cand_norm = np.sqrt(np.sum(np.sum(candidate**2, axis=1)))
                if cand_norm > 0:
                    candidate = candidate / cand_norm
                    
                s = _score(candidate)
                if s > best_score:
                    best_score = s
                    best_points = candidate.copy()
                    
        except Exception:
            pass
        
        num_restarts += 1
        # If we are getting stuck with the same score, maybe try a different random seed or perturbation
        # But SLSQP is local, so random restarts are key.
        
    return best_points

# EVOLVE-BLOCK-END
