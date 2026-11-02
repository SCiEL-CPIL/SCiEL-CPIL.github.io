"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 300

# EVOLVE-BLOCK-START
import numpy as np


def _score(points):
    """Compute the packing score: min of pairwise half-distances and boundary distances."""
    if len(points) < 2:
        return 0.0
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Boundary distances
    radii = np.sqrt(np.sum(points ** 2, axis=1))
    min_boundary = 1.0 - np.max(radii)
    return float(min(min_pair, min_boundary))


def _soft_min(values, beta):
    """Smooth minimum approximation using log-sum-exp."""
    # soft min = -log(sum(exp(-beta * x))) / beta
    neg_beta_x = -beta * values
    max_val = np.max(neg_beta_x)
    if not np.isfinite(max_val):
        return np.min(values)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
    return -log_sum / beta


def _soft_score(points, beta):
    """Compute a soft version of the packing score for gradient-based optimization."""
    # Pairwise soft-min
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
    np.fill_diagonal(dists, np.inf)
    # Get all pairwise distances
    n = len(points)
    idx = np.triu_indices(n, k=1)
    pair_dists = dists[idx] / 2.0
    soft_pair = _soft_min(pair_dists, beta)
    
    # Boundary soft-min
    radii = np.sqrt(np.sum(points ** 2, axis=1))
    boundary_dists = 1.0 - radii
    soft_boundary = _soft_min(boundary_dists, beta)
    
    return float(min(soft_pair, soft_boundary))


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            points = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.98
    
    best_points = points.copy()
    best_score = _score(points)
    
    # Gradient-free optimization using Nelder-Mead style or simple iterative improvement
    # We'll use a simple iterative approach: for each point, try small moves in random directions
    
    rng = np.random.default_rng(42)
    current_points = points.copy()
    current_score = best_score
    
    # Parameters for optimization
    step_size = 0.01
    iterations = 0
    max_iterations = 50000
    
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        
        # Choose a random point to move
        idx = rng.integers(0, N)
        
        # Try random perturbations
        best_move = None
        best_move_score = current_score
        
        for _ in range(20):
            # Random direction
            direction = rng.standard_normal(2)
            direction /= np.linalg.norm(direction)
            
            # Try multiple step sizes
            for scale in [step_size, step_size * 0.5, step_size * 0.1]:
                new_point = current_points[idx].copy()
                new_point += direction * scale
                
                # Check if new point is inside the container
                if np.linalg.norm(new_point) > 0.999:
                    continue
                
                # Tentatively move the point
                old_point = current_points[idx].copy()
                current_points[idx] = new_point
                
                # Compute new score (approximate for speed)
                new_score = _score(current_points)
                
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move = new_point.copy()
                
                # Restore
                current_points[idx] = old_point
        
        # Apply the best move if it improved
        if best_move is not None and best_move_score > current_score:
            current_points[idx] = best_move
            current_score = best_move_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
            
            # Reduce step size on success
            step_size *= 0.95
        else:
            # No improvement, reduce step size
            step_size *= 0.98
        
        # Periodically check if we should stop
        if iterations % 1000 == 0 and time.time() > deadline - 15.0:
            break
    
    # Final polish: try to improve the score with smaller steps
    if time.time() < deadline - 5.0:
        step_size = 0.001
        polish_iterations = 0
        max_polish = 20000
        while time.time() < deadline - 5.0 and polish_iterations < max_polish:
            polish_iterations += 1
            idx = rng.integers(0, N)
            direction = rng.standard_normal(2)
            direction /= np.linalg.norm(direction)
            
            old_point = current_points[idx].copy()
            current_points[idx] = old_point + direction * step_size
            
            if np.linalg.norm(current_points[idx]) <= 0.999:
                new_score = _score(current_points)
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = current_points.copy()
                else:
                    current_points[idx] = old_point
            else:
                current_points[idx] = old_point
    
    return best_points
# EVOLVE-BLOCK-END
