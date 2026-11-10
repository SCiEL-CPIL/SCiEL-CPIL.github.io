# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses gradient ascent on the log-ratio of min/max distances, constrained to the unit sphere.
    Multiple random restarts are used to avoid local minima.
    """
    n = 14
    d = 3
    best_score = 0.0
    best_points = None

    # Number of restarts to ensure we find a good global optimum
    n_restarts = 100
    n_iterations = 1000
    
    for r in range(n_restarts):
        # Initialize points on the unit sphere
        # Using a deterministic seed based on restart index for reproducibility, 
        # but varying the initial distribution
        np.random.seed(r)
        
        # Start with random points on sphere
        points = np.random.randn(n, d)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        current_score = 0.0
        alpha = 0.1  # Initial step size
        
        for i in range(n_iterations):
            # Compute pairwise distances
            dists = pdist(points)
            d_min = np.min(dists)
            d_max = np.max(dists)
            
            if d_max == 0:
                break
            
            score = d_min / d_max
            
            if score > current_score:
                current_score = score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
            
            # Decay step size
            alpha *= 0.995
            
            # Find indices of min and max distances
            # pdist returns condensed distance matrix
            # We need to map back to (i, j) indices
            min_idx = np.argmin(dists)
            max_idx = np.argmax(dists)
            
            # Helper to get i, j from condensed index
            def get_indices(k, n):
                # k is the index in the condensed matrix
                # Number of elements before row i is i*(n-1) - i*(i-1)/2
                # Simpler: just loop or use formula
                # Formula: i = floor((n - 0.5 - sqrt(0.25 + 2*k)) / 1) ? No.
                # Let's use a simple loop for n=14 (91 pairs), it's fast enough
                count = 0
                for i_idx in range(n - 1):
                    for j_idx in range(i_idx + 1, n):
                        if count == k:
                            return i_idx, j_idx
                        count += 1
                return 0, 1 # Fallback

            i_min, j_min = get_indices(min_idx, n)
            i_max, j_max = get_indices(max_idx, n)
            
            # Calculate gradient of log(d_min) - log(d_max)
            # d(log d_min)/dp_i = 2(p_i - p_j) / d_min^2
            # d(log d_max)/dp_i = 2(p_i - p_j) / d_max^2
            grad = np.zeros((n, d))
            
            # Contribution from min distance (we want to increase it)
            if d_min > 1e-10:
                grad[i_min] += 2.0 * (points[i_min] - points[j_min]) / (d_min * d_min)
                grad[j_min] += 2.0 * (points[j_min] - points[i_min]) / (d_min * d_min)
            
            # Contribution from max distance (we want to decrease it)
            if d_max > 1e-10:
                grad[i_max] -= 2.0 * (points[i_max] - points[j_max]) / (d_max * d_max)
                grad[j_max] -= 2.0 * (points[j_max] - points[i_max]) / (d_max * d_max)
            
            # Update points
            points += alpha * grad
            
            # Project back to unit sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1.0
            points /= norms
            
        # Final check after loop
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max > 0:
            final_score = d_min / d_max
            if final_score > best_score:
                best_score = final_score
                best_points = points.copy()

    if best_points is None:
        # Fallback to random if something went wrong
        np.random.seed(0)
        best_points = np.random.randn(n, d)
        best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)

    return best_points

# EVOLVE-BLOCK-END
