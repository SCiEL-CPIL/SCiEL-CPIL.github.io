"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(pos):
    """
    Computes the maximum radius for the given positions.
    Returns 0 if invalid.
    """
    if pos.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pos)):
        return 0.0
    
    # Check if inside [-0.5, 0.5]
    if np.any(pos < -0.5) or np.any(pos > 0.5):
        return 0.0

    # Use cKDTree for efficient nearest neighbor search
    tree = cKDTree(pos)
    # Find nearest neighbor for each point (excluding self)
    dists, _ = tree.query(pos, k=2)
    min_pair_dist = dists[:, 1].min()
    
    # Distance to walls
    min_wall_dist = (0.5 - np.abs(pos)).min()
    
    return min(0.5 * min_pair_dist, min_wall_dist)

def _hex_init():
    """Generate hexagonal lattice initialization."""
    # For N=50, use a hexagonal grid
    # Approximate rows needed
    m = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))
    pos = []
    for row in range(m):
        y = row * 0.866  # sqrt(3)/2
        offset = 0.5 if row % 2 else 0.0
        for col in range(m + 1):
            x = col * 1.0 + offset
            pos.append([x, y])
            if len(pos) >= N:
                break
        if len(pos) >= N:
            break
    
    pos = np.array(pos[:N])
    # Center and scale to fit in [-0.5, 0.5]
    pos -= pos.mean(axis=0)
    max_extent = np.abs(pos).max()
    if max_extent > 0:
        pos *= 0.45 / max_extent
    
    return pos

def _grid_init():
    """Generate square grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _random_init(seed=42):
    """Generate random initialization."""
    np.random.seed(seed)
    return np.random.uniform(-0.45, 0.45, size=(N, 2))

def _sa_optimize(pos_init, time_budget, seed=42):
    """
    Simulated annealing optimization.
    """
    np.random.seed(seed)
    t_start = time.time()
    
    cur_pos = pos_init.copy()
    cur_r = _compute_radius(cur_pos)
    best_pos = cur_pos.copy()
    best_r = cur_r
    
    # Temperature schedule
    T0 = 0.05
    T_min = 1e-6
    cooling = 0.995
    
    step_size = 0.01
    
    while time.time() - t_start < time_budget:
        T = max(T_min, T0 * (cooling ** (int((time.time() - t_start) * 100))))
        
        # Random move
        i = np.random.randint(N)
        old = cur_pos[i].copy()
        cur_pos[i] = old + np.random.normal(0, step_size, size=2)
        cur_pos[i] = np.clip(cur_pos[i], -0.499, 0.499)
        
        new_r = _compute_radius(cur_pos)
        
        if new_r >= cur_r:
            cur_r = new_r
            if cur_r > best_r:
                best_r = cur_r
                best_pos = cur_pos.copy()
        else:
            # Accept with probability exp((new_r - cur_r) / T)
            delta = new_r - cur_r
            if delta < 0 and np.random.rand() < np.exp(delta / T):
                cur_r = new_r
            else:
                cur_pos[i] = old
        
        # Adaptive step size
        if np.random.rand() < 0.05:
            step_size *= 0.98
        step_size = max(step_size, 1e-5)
        
        # Cool down
        T *= cooling
    
    return best_pos, best_r

def _polish(pos_init, time_budget):
    """
    Local polish using greedy hill climbing with cKDTree.
    """
    t_start = time.time()
    cur_pos = pos_init.copy()
    cur_r = _compute_radius(cur_pos)
    best_pos = cur_pos.copy()
    best_r = cur_r
    
    step = 0.002
    np.random.seed(123)
    
    while time.time() - t_start < time_budget:
        improved = False
        for i in range(N):
            old = cur_pos[i].copy()
            # Try several random perturbations
            for _ in range(5):
                cur_pos[i] = old + np.random.normal(0, step, size=2)
                cur_pos[i] = np.clip(cur_pos[i], -0.499, 0.499)
                
                new_r = _compute_radius(cur_pos)
                
                if new_r > cur_r:
                    cur_r = new_r
                    improved = True
                    if cur_r > best_r:
                        best_r = cur_r
                        best_pos = cur_pos.copy()
                    break
            else:
                # No improvement, revert
                cur_pos[i] = old
        
        # Reduce step size if no improvement
        if not improved:
            step *= 0.95
            if step < 1e-6:
                break
    
    return best_pos, best_r

def construct():
    t_start = time.time()
    deadline = 115.0  # Safety margin before 150s kill
    
    # Try multiple initializations
    initializations = [
        ("hex", _hex_init()),
        ("grid", _grid_init()),
        ("random1", _random_init(seed=42)),
        ("random2", _random_init(seed=123)),
    ]
    
    best_pos = None
    best_r = 0.0
    
    # Time budget per initialization
    n_inits = len(initializations)
    time_per_init = (deadline - 20) / n_inits  # Leave 20s for polish
    
    for name, pos_init in initializations:
        if time.time() - t_start > deadline - 15:
            break
        
        # Run SA for a portion of the budget
        sa_budget = min(time_per_init * 0.7, deadline - 25 - (time.time() - t_start))
        if sa_budget > 1:
            pos_sa, r_sa = _sa_optimize(pos_init, sa_budget, seed=hash(name) % 10000)
            if r_sa > best_r:
                best_r = r_sa
                best_pos = pos_sa
        
        # Quick polish
        polish_budget = min(time_per_init * 0.3, deadline - 15 - (time.time() - t_start))
        if polish_budget > 0.5 and best_pos is not None:
            pos_polished, r_polished = _polish(best_pos, polish_budget)
            if r_polished > best_r:
                best_r = r_polished
                best_pos = pos_polished
    
    # Final polish with remaining time
    remaining = deadline - 2 - (time.time() - t_start)
    if remaining > 2 and best_pos is not None:
        pos_final, r_final = _polish(best_pos, remaining)
        if r_final > best_r:
            best_r = r_final
            best_pos = pos_final
    
    # Fallback
    if best_pos is None or not np.all(np.isfinite(best_pos)):
        best_pos = _grid_init()
    
    return best_pos

# EVOLVE-BLOCK-END
