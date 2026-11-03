"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def _pairwise_dists(pos):
    """Compute pairwise distances between all circles. pos: (N, 2)"""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))

def _compute_radii(pos):
    """Compute max radii given positions. Each circle's radius is limited by
    distance to nearest other circle (half) and distance to walls."""
    n = len(pos)
    dists = _pairwise_dists(pos)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists, axis=1)  # distance to nearest other circle
    half_min_pair = 0.5 * min_pair_dist
    
    # Distance to walls
    wall_dist = np.minimum(
        np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
        np.minimum(pos[:, 1], 1.0 - pos[:, 1])
    )
    
    radii = np.minimum(half_min_pair, wall_dist)
    radii = np.clip(radii, 0, 0.5)
    return radii

def _objective(pos):
    """Negative sum of radii (for minimization)."""
    radii = _compute_radii(pos)
    return -np.sum(radii), radii

def _local_search(pos, max_iter=5000, step_init=0.05, cool=0.999, T_init=0.01, T_min=1e-8):
    """Simulated annealing local search."""
    n = len(pos)
    best_pos = pos.copy()
    best_val, best_radii = _objective(best_pos)
    cur_pos = pos.copy()
    cur_val, _ = _objective(cur_pos)
    T = T_init
    rng = np.random.default_rng()
    
    for it in range(max_iter):
        if T < T_min:
            break
        # Random perturbation
        idx = rng.integers(n)
        delta = rng.normal(0, step_init, size=2)
        new_pos = cur_pos.copy()
        new_pos[idx] += delta
        new_pos[idx] = np.clip(new_pos[idx], 0, 1)
        
        new_val, _ = _objective(new_pos)
        delta_val = new_val - cur_val
        
        if delta_val < 0 or rng.random() < np.exp(-delta_val / T):
            cur_pos = new_pos
            cur_val = new_val
            if cur_val < best_val:
                best_pos = cur_pos.copy()
                best_val = cur_val
                best_radii = _compute_radii(best_pos)
        
        T *= cool
        step_init *= 0.9995
    
    return best_pos, best_val, best_radii

def _gradient_local_search(pos, max_iter=3000, lr=0.005):
    """Gradient-based local search using subgradient of the radius function."""
    n = len(pos)
    best_pos = pos.copy()
    best_val, best_radii = _objective(best_pos)
    cur_pos = pos.copy()
    cur_val, _ = _objective(cur_pos)
    
    for it in range(max_iter):
        # Compute subgradient
        radii = _compute_radii(cur_pos)
        grad = np.zeros_like(cur_pos)
        
        for i in range(n):
            # Find which constraint is active
            dists = _pairwise_dists(cur_pos)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists)
            j = np.argmin(dists)
            
            wall_x = min(cur_pos[i, 0], 1 - cur_pos[i, 0])
            wall_y = min(cur_pos[i, 1], 1 - cur_pos[i, 1])
            wall = min(wall_x, wall_y)
            
            r_i = radii[i]
            half_pair = 0.5 * min_pair
            
            if half_pair <= wall + 1e-12:
                # Pair constraint active
                diff = cur_pos[i] - cur_pos[j]
                dist = min_pair if min_pair > 1e-10 else 1e-10
                grad[i] += diff / dist
                grad[j] -= diff / dist
            else:
                # Wall constraint active
                if wall_x <= wall_y:
                    if cur_pos[i, 0] < 0.5:
                        grad[i, 0] += 1
                    else:
                        grad[i, 0] -= 1
                else:
                    if cur_pos[i, 1] < 0.5:
                        grad[i, 1] += 1
                    else:
                        grad[i, 1] -= 1
        
        # Update
        cur_pos -= lr * grad
        cur_pos = np.clip(cur_pos, 0, 1)
        
        new_val, _ = _objective(cur_pos)
        if new_val < cur_val:
            cur_val = new_val
            if cur_val < best_val:
                best_pos = cur_pos.copy()
                best_val = cur_val
                best_radii = _compute_radii(best_pos)
        else:
            lr *= 0.95
        
        if lr < 1e-8:
            break
    
    return best_pos, best_val, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s out of 150s limit
    
    rng = np.random.default_rng(42)
    
    best_pos = None
    best_val = np.inf
    best_radii = None
    
    # Multiple diverse initializations
    n_restarts = 30
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial positions
        if restart == 0:
            # Grid-like initial
            side = int(np.ceil(np.sqrt(N)))
            pos = np.array([((i + 0.5) / side, (j + 0.5) / side) 
                          for i in range(side) for j in range(side)][:N])
        elif restart == 1:
            # Random uniform
            pos = rng.random((N, 2))
        elif restart == 2:
            # Hexagonal-like
            rows = int(np.ceil(N / 3))
            pos = []
            for r in range(rows):
                for c in range(3):
                    if len(pos) >= N:
                        break
                    x = (c + 0.5 + (r % 2) * 0.5) / 3
                    y = (r + 0.5) / rows
                    pos.append((x, y))
            pos = np.array(pos[:N])
        else:
            # Random with jitter from grid
            side = int(np.ceil(np.sqrt(N)))
            pos = np.array([((i + 0.5) / side, (j + 0.5) / side) 
                          for i in range(side) for j in range(side)][:N])
            pos += rng.normal(0, 0.05, size=pos.shape)
            pos = np.clip(pos, 0.01, 0.99)
        
        # SA local search
        pos_sa, val_sa, radii_sa = _local_search(
            pos, max_iter=2000, step_init=0.03, cool=0.9995, T_init=0.005, T_min=1e-9
        )
        
        if val_sa < best_val:
            best_val = val_sa
            best_pos = pos_sa
            best_radii = radii_sa
        
        # Gradient local search from SA result
        pos_grad, val_grad, radii_grad = _gradient_local_search(
            pos_sa, max_iter=2000, lr=0.003
        )
        
        if val_grad < best_val:
            best_val = val_grad
            best_pos = pos_grad
            best_radii = radii_grad
        
        # Additional SA from gradient result
        if time.time() < deadline - 5:
            pos_sa2, val_sa2, radii_sa2 = _local_search(
                pos_grad, max_iter=1000, step_init=0.01, cool=0.999, T_init=0.001, T_min=1e-10
            )
            if val_sa2 < best_val:
                best_val = val_sa2
                best_pos = pos_sa2
                best_radii = radii_sa2
    
    # Final result
    result = np.column_stack([best_pos, best_radii])
    return result
# EVOLVE-BLOCK-END
