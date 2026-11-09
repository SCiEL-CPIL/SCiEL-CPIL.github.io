# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

def _score(points):
    """Compute (d_min/d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    d_min = dists.min()
    d_max = dists.max()
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    
    Uses Directed Simulated Annealing targeting the critical pairs.
    """
    n = 14
    d = 3
    
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 10.0
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > (time_limit - safety_margin)
    
    def objective(points):
        """Returns (d_min/d_max)^2, higher is better."""
        if len(points) < 2:
            return 0.0
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = dists.min()
        d_max = dists.max()
        if d_max == 0:
            return 0.0
        if not np.isfinite(d_min) or not np.isfinite(d_max):
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def get_min_max_pairs(points):
        """Returns indices of the min and max distance pairs."""
        dists = pdist(points)
        min_idx = np.argmin(dists)
        max_idx = np.argmax(dists)
        
        def tri_to_pair(k, n):
            i = int((np.sqrt(8 * k + 1) - 1) / 2)
            j = i + 1 + (k - i * (i + 1) // 2)
            return i, j
        
        i_min, j_min = tri_to_pair(min_idx, n)
        i_max, j_max = tri_to_pair(max_idx, n)
        
        return (i_min, j_min), (i_max, j_max)

    def directed_sa(points, max_iters, rng):
        """
        Simulated Annealing with directed moves on min/max pairs.
        Normalizes points such that d_max = 1 to maintain scale.
        """
        current_points = points.copy()
        # Initial normalization
        dists = pdist(current_points)
        d_max_init = dists.max()
        if d_max_init > 1e-6:
            current_points = current_points / d_max_init
            
        current_score = objective(current_points)
        best_points = current_points.copy()
        best_score = current_score
        
        temperature = 0.1
        cooling_rate = 0.9995
        
        for it in range(max_iters):
            if check_deadline():
                break
            
            (i_min, j_min), (i_max, j_max) = get_min_max_pairs(current_points)
            
            new_points = current_points.copy()
            
            # Directed moves
            # 1. Push min pair apart
            diff_min = new_points[j_min] - new_points[i_min]
            norm_min = np.linalg.norm(diff_min)
            if norm_min > 1e-6:
                step_min = (diff_min / norm_min) * (0.05 * temperature + 0.001)
                new_points[i_min] -= step_min
                new_points[j_min] += step_min

            # 2. Pull max pair together
            diff_max = new_points[j_max] - new_points[i_max]
            norm_max = np.linalg.norm(diff_max)
            if norm_max > 1e-6:
                step_max = (diff_max / norm_max) * (0.05 * temperature + 0.001)
                new_points[i_max] += step_max
                new_points[j_max] -= step_max

            # Global noise
            new_points += rng.standard_normal((n, d)) * temperature * 0.05
            
            # Normalize to d_max = 1
            dists_new = pdist(new_points)
            d_max_new = dists_new.max()
            if d_max_new > 1e-6:
                new_points = new_points / d_max_new
            else:
                # Degenerate case, add noise
                new_points = current_points + rng.standard_normal((n, d)) * 0.1
                dists_new = pdist(new_points)
                d_max_new = dists_new.max()
                if d_max_new > 1e-6:
                    new_points = new_points / d_max_new

            new_score = objective(new_points)
            
            # Metropolis
            delta = new_score - current_score
            if delta > 0 or rng.random() < np.exp(delta / max(temperature, 1e-10)):
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            
            temperature *= cooling_rate
        
        return best_points, best_score

    def fibonacci_sphere(n_points):
        points = []
        for i in range(n_points):
            phi = np.arccos(1 - 2 * (i + 0.5) / n_points)
            theta = np.pi * (1 + 5**0.5) * i
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points.append([x, y, z])
        return np.array(points)

    def random_sphere(n_points, rng):
        points = rng.standard_normal((n_points, 3))
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norms

    best_points = None
    best_score = -1.0
    
    rng = np.random.default_rng(42)
    
    # Start 1: Fibonacci
    init = fibonacci_sphere(n)
    pts, score = directed_sa(init, max_iters=20000, rng=rng)
    if score > best_score:
        best_score = score
        best_points = pts.copy()
        
    # Random Restarts
    num_restarts = 100
    for r in range(num_restarts):
        if check_deadline():
            break
        
        init = random_sphere(n, rng)
        # Perturb initial random sphere
        init += rng.standard_normal((n, 3)) * 0.2
        
        # Vary iterations based on remaining time
        remaining = time_limit - safety_margin - elapsed()
        iters = int(remaining / (num_restarts - r) * 0.8)
        iters = max(1000, min(iters, 10000))
        
        pts, score = directed_sa(init, max_iters=iters, rng=rng)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            
    if best_points is None:
        best_points = fibonacci_sphere(n)
        
    return best_points

# EVOLVE-BLOCK-END
