import numpy as np
import time
import sys

N = 130
# Evaluator kills at 150s. We use 140s as a hard stop to ensure we return valid data.
_DEADLINE = time.time() + 140.0

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    # Handle zero vectors
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _time_ok(margin=2.0):
    """Check if we have enough time left."""
    return time.time() < _DEADLINE - margin

def _generate_grid(n_grid):
    """Generate a dense Fibonacci spiral grid for covering radius estimation."""
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(grid_pts)

def _covering_radius_fast(pts, grid_pts):
    """
    Estimate covering radius using a dense grid.
    Returns the max angular distance in radians.
    """
    # Compute dot products between grid points and config points
    # dots shape: (n_grid, N)
    dots = grid_pts @ pts.T
    # For each grid point, find the max dot product (nearest neighbor)
    max_dots = np.max(dots, axis=1)
    # Clip to avoid numerical issues with arccos
    max_dots = np.clip(max_dots, -1.0, 1.0)
    # Angular distance
    angles = np.arccos(max_dots)
    return np.max(angles)

def _fibonacci_spiral(n, seed=0):
    """Generate initial Fibonacci spiral configuration."""
    rng = np.random.RandomState(seed)
    # Add slight jitter to break symmetries
    i = np.arange(n) + 0.5
    # Random permutation to shuffle the spiral order, effectively randomizing the phase
    # This creates a different starting point while maintaining uniform distribution
    perm = rng.permutation(n)
    i = i[perm]
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def construct():
    """
    Main entry point.
    Strategy: Simulated Annealing (SA) to minimize the covering radius.
    1. Initialize with a Fibonacci spiral.
    2. Use a dense grid to estimate the covering radius.
    3. Perturb random points by random amounts.
    4. Accept moves if they improve the score, or with probability exp(-delta/T) if they worsen it.
    5. Cool down the temperature T over time.
    """
    start_time = time.time()
    
    # Use a moderately dense grid for fast evaluation during SA.
    # 10,000 points is a good balance. 10k * 130 = 1.3M dot products per eval.
    # NumPy handles this in ~1-2ms.
    n_grid = 10000
    grid_pts = _generate_grid(n_grid)
    
    # Initial configuration
    # Start with a standard Fibonacci spiral, which is already a good covering.
    current_pts = _fibonacci_spiral(N, seed=42)
    current_score = _covering_radius_fast(current_pts, grid_pts)
    
    best_pts = current_pts.copy()
    best_score = current_score
    
    # SA Parameters
    # Initial temperature should be large enough to accept bad moves to escape local minima.
    # Covering radius is in radians. 11 degrees ~ 0.2 rad.
    # A difference of 0.01 rad is significant.
    T = 0.05 
    T_min = 1e-6
    # Cooling rate per step
    alpha = 0.999
    
    # Move size: How much to perturb a point.
    # We move a point along a random tangent direction.
    base_step = 0.1
    
    # Track stagnation
    stagnation_counter = 0
    last_improve_score = current_score
    
    # Main Loop
    # We run until time runs out or T is too small.
    # Estimated steps: If eval takes 2ms, we can do ~1000 steps/sec.
    # 100 sec * 1000 = 100,000 steps.
    
    rng = np.random.RandomState(12345)
    
    while _time_ok(5.0):
        # Check for convergence
        if T < T_min:
            break
            
        # Generate a move
        # Pick a random point to move
        idx = rng.randint(N)
        
        # Generate a random direction on the sphere tangent to current point
        # Random vector
        rand_vec = rng.randn(3)
        # Project onto tangent plane: v - (v . p) p
        p = current_pts[idx]
        proj = np.dot(rand_vec, p)
        tangent = rand_vec - proj * p
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-12:
            continue
        tangent /= tangent_norm
        
        # Step size can be adapted. Keep it constant for simplicity, 
        # or scale with T.
        step_size = base_step * (T / T_min ** 0.5) # Scale down as T drops? 
        # Actually, fixed step size with T controlling acceptance is standard.
        # Let's use a fixed step size relative to the geometry.
        step_size = 0.05 
        
        new_p = p + step_size * tangent
        new_p = _normalize(new_p.reshape(1, 3))[0]
        
        # Apply move
        old_p = current_pts[idx].copy()
        current_pts[idx] = new_p
        
        # Evaluate new score
        new_score = _covering_radius_fast(current_pts, grid_pts)
        
        # Acceptance Criterion
        delta = new_score - current_score
        if delta < 0 or rng.rand() < np.exp(-delta / T):
            # Accept
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_pts = current_pts.copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
        else:
            # Reject
            current_pts[idx] = old_p
        
        # Cool down
        T *= alpha
        
        # If we are stuck for too long, we might want to restart or reheat?
        # For now, just keep going. The random nature of SA should handle it.
        
    # Final Polish: 
    # The grid approximation might be slightly off from the true convex hull covering radius.
    # The SA has found a very good local minimum on the grid.
    # We can try a few steps of gradient descent on the grid to refine.
    # But given time constraints, the SA result is likely robust.
    
    # Ensure we return the best found
    return _normalize(best_pts)

if __name__ == "__main__":
    t0 = time.time()
    pts = construct()
    print("Shape:", pts.shape)
    # Use a very dense grid to estimate the final score for local testing
    n_grid_test = 20000
    grid_test = _generate_grid(n_grid_test)
    score = _covering_radius_fast(pts, grid_test)
    print("Covering radius (deg, grid approx):", np.degrees(score))
    print("Time: %.1f s" % (time.time() - t0))
