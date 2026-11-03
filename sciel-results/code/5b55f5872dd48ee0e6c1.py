"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.spatial.distance import cdist

N = 26

# EVOLVE-BLOCK-START

def _max_radius_for_positions(pos):
    """
    Given positions (N,2), compute the maximum radius each circle can have
    such that all circles are disjoint and inside the unit square.
    Uses pairwise distances and boundary distances.
    """
    n = len(pos)
    # Pairwise distances
    dists = cdist(pos, pos)
    # Boundary distances: distance from each point to the 4 walls
    # For circle to be inside [0,1]^2, radius <= min(x, 1-x, y, 1-y)
    boundary = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                          np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    
    # For each circle i, the max radius is limited by:
    # - boundary[i]
    # - dists[i,j] / 2 for all j != i
    # So r_i = min(boundary[i], min_{j!=i} dists[i,j]/2)
    
    # Set diagonal to infinity so it doesn't constrain
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists, axis=1) / 2.0
    
    radii = np.minimum(boundary, min_pair_dist)
    # Ensure non-negative
    radii = np.maximum(radii, 0.0)
    return radii


def _objective(pos):
    """Compute sum of radii for given positions."""
    radii = _max_radius_for_positions(pos)
    return np.sum(radii)


def _is_valid(pos, eps=1e-9):
    """Check if positions yield valid non-overlapping circles in unit square."""
    radii = _max_radius_for_positions(pos)
    if np.any(radii < -eps):
        return False
    # Check pairwise non-overlap with a tiny margin
    dists = cdist(pos, pos)
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if dists[i, j] < radii[i] + radii[j] - eps:
                return False
        # Check boundary
        if (pos[i, 0] < radii[i] - eps or pos[i, 0] > 1 - radii[i] + eps or
            pos[i, 1] < radii[i] - eps or pos[i, 1] > 1 - radii[i] + eps):
            return False
    return True


def _initial_grid():
    """Create a grid-based initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    pos = []
    for i in range(side):
        for j in range(side):
            if len(pos) >= N:
                break
            pos.append([(i + 0.5) / side, (j + 0.5) / side])
    return np.array(pos[:N])


def _initial_random():
    """Create a random initial configuration with some spacing."""
    pos = []
    attempts = 0
    while len(pos) < N and attempts < 10000:
        attempts += 1
        x, y = random.random(), random.random()
        # Check minimum distance from existing points
        if all(np.hypot(x - px, y - py) > 0.05 for px, py in pos):
            pos.append([x, y])
    # If we couldn't place enough, fill with random
    while len(pos) < N:
        pos.append([random.random(), random.random()])
    return np.array(pos)


def _simulated_annealing(pos, t_start=1.0, t_end=0.001, n_iter=50000, seed=42):
    """
    Simulated annealing to maximize sum of radii.
    """
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_r = _max_radius_for_positions(current_pos)
    current_obj = np.sum(current_r)
    
    best_pos = current_pos.copy()
    best_r = current_r.copy()
    best_obj = current_obj
    
    n = len(current_pos)
    
    for it in range(n_iter):
        # Temperature schedule (geometric)
        t = t_start * (t_end / t_start) ** (it / n_iter)
        
        # Choose a random circle to move
        i = rng.randint(n)
        
        # Generate a perturbation
        step_size = 0.05 * t + 0.001
        dx = rng.uniform(-step_size, step_size)
        dy = rng.uniform(-step_size, step_size)
        
        new_pos = current_pos.copy()
        new_pos[i, 0] += dx
        new_pos[i, 1] += dy
        
        # Clamp to [0,1]
        new_pos[i, 0] = np.clip(new_pos[i, 0], 0.0, 1.0)
        new_pos[i, 1] = np.clip(new_pos[i, 1], 0.0, 1.0)
        
        new_r = _max_radius_for_positions(new_pos)
        new_obj = np.sum(new_r)
        
        delta = new_obj - current_obj
        
        if delta > 0 or rng.random() < np.exp(delta / t):
            current_pos = new_pos
            current_r = new_r
            current_obj = new_obj
            
            if current_obj > best_obj:
                best_pos = current_pos.copy()
                best_r = current_r.copy()
                best_obj = current_obj
    
    return best_pos, best_r, best_obj


def _local_search(pos, n_iter=5000, step_size=0.01, seed=42):
    """
    Local search: iteratively move each circle to improve the objective.
    """
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_r = _max_radius_for_positions(current_pos)
    current_obj = np.sum(current_r)
    
    n = len(current_pos)
    improved = True
    it = 0
    while improved and it < n_iter:
        improved = False
        it += 1
        # Random order
        order = rng.permutation(n)
        for i in order:
            # Try several small moves
            for _ in range(20):
                dx = rng.uniform(-step_size, step_size)
                dy = rng.uniform(-step_size, step_size)
                new_pos = current_pos.copy()
                new_pos[i, 0] = np.clip(current_pos[i, 0] + dx, 0.0, 1.0)
                new_pos[i, 1] = np.clip(current_pos[i, 1] + dy, 0.0, 1.0)
                
                new_r = _max_radius_for_positions(new_pos)
                new_obj = np.sum(new_r)
                
                if new_obj > current_obj + 1e-12:
                    current_pos = new_pos
                    current_r = new_r
                    current_obj = new_obj
                    improved = True
                    break
    
    return current_pos, current_r, current_obj


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Multiple diverse initializations
    initializations = []
    
    # Grid initialization
    initializations.append(_initial_grid())
    
    # Several random initializations
    for s in range(5):
        random.seed(s * 1000 + 42)
        np.random.seed(s * 1000 + 42)
        initializations.append(_initial_random())
    
    best_pos = None
    best_r = None
    best_obj = -np.inf
    
    for init_idx, init_pos in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        # Simulated annealing
        sa_pos, sa_r, sa_obj = _simulated_annealing(
            init_pos, t_start=0.5, t_end=0.001, n_iter=30000, seed=init_idx * 7 + 1
        )
        
        if sa_obj > best_obj:
            best_pos = sa_pos
            best_r = sa_r
            best_obj = sa_obj
        
        # Local search refinement
        if time.time() > deadline - 10:
            break
        
        ls_pos, ls_r, ls_obj = _local_search(
            sa_pos, n_iter=3000, step_size=0.005, seed=init_idx * 13 + 2
        )
        
        if ls_obj > best_obj:
            best_pos = ls_pos
            best_r = ls_r
            best_obj = ls_obj
    
    # Final local search on the best
    if time.time() < deadline - 5 and best_pos is not None:
        final_pos, final_r, final_obj = _local_search(
            best_pos, n_iter=5000, step_size=0.003, seed=999
        )
        if final_obj > best_obj:
            best_pos = final_pos
            best_r = final_r
            best_obj = final_obj
    
    if best_pos is None:
        # Fallback: grid
        pos = _initial_grid()
        r = _max_radius_for_positions(pos)
        best_pos = pos
        best_r = r
    
    # Build result array
    result = np.column_stack([best_pos, best_r])
    return result

# EVOLVE-BLOCK-END
