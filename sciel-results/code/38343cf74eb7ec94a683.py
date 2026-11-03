"""Pack 50 equal circles in a unit square, maximising the common radius."""
import numpy as np
import time

N = 50

# Known best radius (Boll et al. 2000)
R_KNOWN = 0.071377103865

def _score(centers):
    """Compute the radius for a given set of centers."""
    # Distance to walls
    wall_dists = np.minimum(0.5 - np.abs(centers), 0.5)
    min_wall = np.min(wall_dists)
    # Pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(min_wall, min_pair / 2.0)


def _initial_grid():
    """Create an initial grid layout."""
    m = 7  # 7x7 grid, use 50 of the 49... wait, 7x7=49, need 50
    # Let's use a slightly different approach: 5 rows, 10 columns
    # Or better: start with a hexagonal-ish pattern
    # For now, use a 7x7 grid but that's only 49. Let's do 5x10 = 50
    # Actually, let me just use a perturbed grid of 50 points
    
    # Use a 7x8 grid and take the first 50
    xs = np.linspace(-0.5, 0.5, 8)
    ys = np.linspace(-0.5, 0.5, 7)
    grid = np.array([[x, y] for y in ys for x in xs])
    return grid[:N].copy()


def _hexagonal_init():
    """Create a hexagonal packing initial guess."""
    # Hexagonal packing in a square
    # For 50 circles, try to arrange in hexagonal pattern
    # This is complex; let's use a simpler approach with random initialization
    
    # Random initialization within the square
    np.random.seed(42)
    centers = np.random.uniform(-0.5, 0.5, size=(N, 2))
    return centers


def _simulated_annealing(centers, max_iter=50000, t_start=0.01, t_end=0.0001, 
                          alpha=0.9999, deadline=None):
    """Run simulated annealing to improve the packing."""
    current = centers.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    t = t_start
    n_iter = 0
    
    while n_iter < max_iter:
        if deadline is not None and time.time() > deadline:
            break
            
        # Propose a random move
        idx = np.random.randint(N)
        move_size = t * 0.5
        delta = np.random.normal(0, move_size, size=2)
        proposed = current.copy()
        proposed[idx] += delta
        
        # Keep within bounds
        proposed[idx, 0] = np.clip(proposed[idx, 0], -0.5, 0.5)
        proposed[idx, 1] = np.clip(proposed[idx, 1], -0.5, 0.5)
        
        proposed_score = _score(proposed)
        
        if proposed_score > current_score or np.random.random() < np.exp((proposed_score - current_score) / max(t, 1e-10)):
            current = proposed
            current_score = proposed_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        
        t *= alpha
        n_iter += 1
        
        # Check deadline periodically
        if n_iter % 1000 == 0 and deadline is not None and time.time() > deadline:
            break
    
    return best, best_score


def _local_search(centers, max_iter=10000, deadline=None):
    """Perform local search with gradient-like updates."""
    current = centers.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    for n_iter in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Find the most constrained pair and push them apart
        diffs = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Also consider wall constraints
        wall_dists = 0.5 - np.abs(current)
        min_wall_idx = np.argmin(np.min(wall_dists, axis=1))
        min_wall_val = np.min(wall_dists[min_wall_idx])
        
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_pair_val = dists[min_pair_idx] / 2.0
        
        if min_wall_val < min_pair_val:
            # Push the circle away from the wall
            idx = min_wall_idx
            for dim in range(2):
                if current[idx, dim] > 0 and wall_dists[idx, dim] == min_wall_val:
                    current[idx, dim] -= 0.001
                elif current[idx, dim] < 0 and wall_dists[idx, dim] == min_wall_val:
                    current[idx, dim] += 0.001
                elif wall_dists[idx, dim] == min_wall_val:
                    # At center, push randomly
                    current[idx, dim] += 0.001 * (1 if np.random.random() > 0.5 else -1)
        else:
            # Push the closest pair apart
            i, j = min_pair_idx
            if i == j:
                continue
            diff = current[i] - current[j]
            norm = np.linalg.norm(diff)
            if norm < 1e-10:
                diff = np.random.normal(0, 0.001, size=2)
                norm = np.linalg.norm(diff)
            step = 0.001 * diff / norm
            current[i] += step
            current[j] -= step
            # Keep within bounds
            current[i] = np.clip(current[i], -0.5, 0.5)
            current[j] = np.clip(current[j], -0.5, 0.5)
        
        new_score = _score(current)
        if new_score > current_score:
            current_score = new_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        else:
            # Revert
            current = best.copy()
    
    return best, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s safety margin
    
    # Start with a hexagonal-like initial guess
    # For 50 circles, try rows of varying length
    centers = _hexagonal_init()
    
    # Run simulated annealing
    best, best_score = _simulated_annealing(centers, max_iter=30000, 
                                              t_start=0.02, t_end=0.00001,
                                              alpha=0.99995, deadline=deadline)
    
    # If we have time, do local search refinement
    if time.time() < deadline - 10:
        best2, score2 = _local_search(best, max_iter=20000, deadline=deadline)
        if score2 > best_score:
            best = best2
            best_score = score2
    
    # Final check: ensure all centers are within bounds
    best = np.clip(best, -0.5, 0.5)
    
    return best
