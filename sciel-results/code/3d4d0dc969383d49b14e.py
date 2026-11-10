import numpy as np
from scipy.optimize import minimize
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin for 120s limit

def construct():
    # 1. Initial Configuration: Hexagonal Lattice
    # We want to place N points. A hexagonal lattice is a good start.
    # We'll generate a grid and pick the N points closest to the origin 
    # that satisfy the constraint that they are inside the unit circle 
    # (with a buffer).
    
    def get_initial_points():
        # Try different spacings to find a lattice that yields at least N points
        # inside the unit circle.
        for spacing in np.linspace(0.1, 0.5, 50):
            # Generate points in a hexagonal grid
            # The grid extends roughly from -2 to 2 in both x and y
            # to ensure we have enough points to choose from.
            points = []
            # Hexagonal lattice vectors
            a = spacing
            b = spacing * np.sqrt(3)
            
            # Iterate over a range of lattice indices
            for i in range(-10, 11):
                for j in range(-10, 11):
                    x = i * a + (j % 2) * a / 2.0
                    y = j * b
                    r = np.hypot(x, y)
                    if r < 1.0:
                        points.append((x, y))
            
            points = np.array(points)
            if len(points) >= N:
                # Sort by distance to origin, pick the N closest
                # This ensures we pick the "densest" core part of the lattice
                dists = np.linalg.norm(points, axis=1)
                idx = np.argsort(dists)[:N]
                return points[idx]
        
        # Fallback: random points if lattice fails (unlikely)
        return np.random.rand(N, 2) * 2 - 1 # Uniform in square, will be projected/optimized

    x0 = get_initial_points()
    
    # Ensure x0 is valid (finite, correct shape)
    if not np.all(np.isfinite(x0)):
        return np.zeros((N, 2))
        
    best_x = x0.copy()
    best_score = -np.inf

    # Helper to compute the score (radius)
    def compute_score(x):
        # x: (N, 2)
        # Check boundaries
        r_bounds = 1.0 - np.linalg.norm(x, axis=1)
        if np.any(r_bounds < 0):
            return -np.inf # Invalid
            
        # Check pairwise distances
        # Efficient way to compute all pairwise distances
        # diff[i,j] = x[i] - x[j]
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Mask out self-distances
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        
        # The radius is half the min pair dist, limited by boundary distance
        r = min(min_pair_dist / 2.0, np.min(r_bounds))
        
        return r

    current_score = compute_score(best_x)
    if current_score > best_score:
        best_score = current_score

    # 2. Optimization: Maximize the minimum distance
    # We use a smooth surrogate: Softmin of the distances.
    # Objective: Minimize -Softmin(distances)
    # Softmin(d) = -1/beta * log(sum(exp(-beta * d)))
    # This is smooth and approximates min(d) as beta -> inf.
    
    def objective(x, beta):
        # Flatten x to (2N,) for scipy
        x = x.reshape(N, 2)
        
        # Boundary distances
        r_bounds = 1.0 - np.linalg.norm(x, axis=1)
        
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # We want to maximize r = min(min_pair_dist/2, min(r_bounds))
        # Let's define the "active" distances.
        # d_pairs = min_pair_dist / 2.0
        # d_bound = min(r_bounds)
        # We want to maximize min(d_pairs, d_bound).
        # Using softmin on the set of all "critical" distances:
        # All pairwise distances / 2 and all boundary distances.
        
        # To keep it efficient, we can just use the global min for the hard check
        # but for the gradient, we need the soft version.
        # Let's construct the vector of all relevant distances.
        # This is O(N^2). For N=50, 2500 elements. Acceptable.
        
        # Extract upper triangle for pairs
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu] / 2.0
        
        # Combine pair dists and boundary dists
        all_dists = np.concatenate([pair_dists, r_bounds])
        
        # Softmin
        # To avoid overflow, subtract max
        max_d = np.max(all_dists)
        exp_terms = np.exp(-beta * (all_dists - max_d))
        log_sum = np.log(np.sum(exp_terms))
        softmin = -1.0 / beta * log_sum + max_d
        
        # We want to maximize softmin, so minimize -softmin
        return -softmin

    # We will run optimization with increasing beta
    # Start with beta=10, go up to 100
    
    for beta in [10.0, 50.0, 100.0, 200.0]:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        # Use current best as start
        x_start = best_x.reshape(-1)
        
        # Bounds: Each coordinate in [-1, 1] (actually constrained by circle, but box is safe)
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        try:
            res = minimize(objective, x_start, args=(beta,), method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12})
            
            if res.success or res.fun < objective(x_start, beta):
                x_opt = res.x.reshape(N, 2)
                score_opt = compute_score(x_opt)
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_x = x_opt
        except Exception:
            pass

    # Final check and return
    # Ensure the returned points are valid
    final_score = compute_score(best_x)
    if final_score < 0 or not np.all(np.isfinite(best_x)):
        # Fallback to initial if something went wrong
        return x0
        
    return best_x

# Keep the original function signature if required by the harness
# The prompt says: "Keep the function names and signatures of the seed program."
# Seed program: def construct():
# So we just define construct().

# The code above defines construct().

if __name__ == "__main__":
    # Simple test
    pts = construct()
    print(pts.shape)
    # Calculate score
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    r_bound = np.min(1.0 - np.linalg.norm(pts, axis=1))
    r = min(min_pair / 2.0, r_bound)
    print(f"Score: {r}")
