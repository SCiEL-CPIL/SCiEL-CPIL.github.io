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
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _compute_dists(points):
    """Compute all pairwise distances."""
    return pdist(points)

def _repulsive_relaxation(points, iterations=1000, alpha=0.1):
    """
    Iteratively push points apart on the sphere.
    We fix the points on the unit sphere to handle scaling/translation.
    The goal is to maximize the minimum distance.
    """
    n = points.shape[0]
    # Normalize to sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    current_min_dist = 0.0
    for it in range(iterations):
        # Compute pairwise distances
        dists = _compute_dists(points)
        if len(dists) == 0:
            break
        d_min = np.min(dists)
        
        if d_min == 0:
            # If points coincide, add small random jitter
            points += np.random.randn(n, 3) * 0.01
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            continue
            
        if d_min > current_min_dist:
            current_min_dist = d_min
            # If we have a very good min dist, we can stop early? 
            # For 14 points in 3D, max min dist on unit sphere is approx 0.5-0.6?
            # Actually, let's just run all iterations or until convergence.
            
        # Calculate forces
        # Force is repulsive: F_ij = k * (1/d_ij^2) * unit_vector_ij
        # We want to move each point in the direction of the sum of forces from all others.
        
        # Vectorized force calculation
        # points shape (n, 3)
        # diff shape (n, n, 3)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_matrix = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        np.fill_diagonal(dists_matrix, 1.0)
        
        # Force magnitude: 1/d^2 is strong. Let's use a softer potential or just inverse.
        # To maximize min distance, we should focus force on the closest pairs.
        # A common approach: F_ij proportional to 1/d_ij^p.
        # Let's try p=2.
        
        # Normalize diff to get unit vectors
        unit_vectors = diff / dists_matrix[:, :, np.newaxis]
        
        # Apply force magnitude
        # We only want to push apart, so force is along the vector from j to i.
        # diff[i,j] = p_i - p_j. So unit_vector points from j to i.
        # We want to move p_i away from p_j, so add force in direction of unit_vector.
        
        force_mag = 1.0 / (dists_matrix ** 2)
        np.fill_diagonal(force_mag, 0.0)
        
        # Total force on each point
        forces = np.sum(unit_vectors * force_mag[:, :, np.newaxis], axis=1)
        
        # Update points
        # Step size decay
        step_size = alpha / (1.0 + it * 0.01)
        points = points + step_size * forces
        
        # Project back to sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        points = points / norms
        
    return points

def _local_hill_climb(points, max_iter=2000, step_size=0.05):
    """
    Local search by perturbing individual points.
    """
    n = points.shape[0]
    best_score = _score(points)
    best_points = points.copy()
    
    for it in range(max_iter):
        # Pick a random point to perturb
        idx = np.random.randint(n)
        original = best_points[idx].copy()
        
        # Generate a small random perturbation
        perturbation = np.random.randn(3) * step_size
        new_point = original + perturbation
        
        # Project back to sphere (maintaining the constraint that points are on the sphere)
        # Note: The optimal solution might not be strictly on the sphere, but for max min dist ratio,
        # pushing to the boundary is generally good.
        norm = np.linalg.norm(new_point)
        if norm > 0:
            new_point = new_point / norm
        else:
            new_point = original + np.random.randn(3) * 0.01
            new_point = new_point / np.linalg.norm(new_point)
            
        candidate = best_points.copy()
        candidate[idx] = new_point
        
        cand_score = _score(candidate)
        
        if cand_score > best_score:
            best_score = cand_score
            best_points = candidate
            # Keep step size or slightly increase?
            # step_size *= 1.05
        else:
            # Decay step size
            step_size *= 0.995
            if step_size < 0.001:
                break
                
    return best_points, best_score

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    start_time = time.time()
    deadline = start_time + 600  # Safety margin
    
    n_points = 14
    dim = 3
    
    best_score = 0.0
    best_points = None
    
    # Multiple restarts
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline - 60:
            break
            
        # 1. Initialize
        # Use a mix of random and structured initializations
        if r % 3 == 0:
            # Random on sphere
            points = np.random.randn(n_points, dim)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
        elif r % 3 == 1:
            # Fibonacci sphere
            i = np.arange(0, n_points) + 0.5
            phi = np.arccos(1 - 2 * i / n_points)
            theta = np.pi * (1 + 5 ** 0.5) * i
            x = np.cos(theta) * np.sin(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(phi)
            points = np.column_stack((x, y, z))
        else:
            # Simplex-like or grid perturbation
            points = np.random.uniform(-1, 1, (n_points, dim))
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
        
        # 2. Repulsive Relaxation
        # Run for a moderate number of iterations
        points_relaxed = _repulsive_relaxation(points, iterations=500, alpha=0.1)
        
        # 3. Local Hill Climbing
        points_opt, score_opt = _local_hill_climb(points_relaxed, max_iter=1000, step_size=0.05)
        
        if score_opt > best_score:
            best_score = score_opt
            best_points = points_opt.copy()
            
        # Optional: Second round of hill climbing with smaller steps if time permits
        if time.time() < deadline - 120 and score_opt > best_score:
            points_opt2, score_opt2 = _local_hill_climb(points_opt, max_iter=1000, step_size=0.01)
            if score_opt2 > best_score:
                best_score = score_opt2
                best_points = points_opt2.copy()

    # If no optimization succeeded or score is 0, return a valid fallback
    if best_points is None or best_score == 0:
        # Return points on a sphere (Fibonacci)
        i = np.arange(0, n_points) + 0.5
        phi = np.arccos(1 - 2 * i / n_points)
        theta = np.pi * (1 + 5 ** 0.5) * i
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        best_points = np.column_stack((x, y, z))
        
    # Final check and normalization
    # The metric is scale invariant, but we should return finite points.
    # The problem doesn't require unit sphere output, just maximizing the ratio.
    # However, keeping them on the sphere is a valid configuration.
    
    return best_points

# EVOLVE-BLOCK-END
