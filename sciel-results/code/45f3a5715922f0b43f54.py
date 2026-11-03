"""Stochastic local search with simulated annealing for packing 100 equal spheres in a unit cube."""
import numpy as np
import random
import math

N = 100

def _compute_radius(c):
    """Compute the packing radius: min of half min pairwise distance and min boundary distance."""
    # Pairwise distances
    iu = np.triu_indices(N, k=1)
    diff = c[iu[0]] - c[iu[1]]
    dists = np.sqrt(np.sum(diff**2, axis=1))
    r_pair = np.min(dists) / 2.0
    
    # Boundary distances
    abs_c = np.abs(c)
    d_x = 0.5 - abs_c[:, 0]
    d_y = 0.5 - abs_c[:, 1]
    d_z = 0.5 - abs_c[:, 2]
    r_bound = np.min(np.minimum.reduce([d_x, d_y, d_z]))
    
    return min(r_pair, r_bound)

def _init_random():
    """Initialize with random positions in the cube."""
    return np.random.uniform(-0.45, 0.45, size=(N, 3))

def _init_grid():
    """Initialize with a grid pattern."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _sa_swap_perturb(c, max_iters=30000, T0=0.01, T1=1e-8, alpha=0.9997, pert_scale=0.01):
    """
    Simulated annealing with swap and perturbation moves.
    
    Moves:
    1. Swap two spheres that are closest together (with random jitter)
    2. Small random perturbation of one or two spheres
    3. Move the most boundary-constrained sphere inward
    """
    c = c.copy()
    current_r = _compute_radius(c)
    best_c = c.copy()
    best_r = current_r
    
    T = T0
    
    for it in range(max_iters):
        move_type = random.random()
        
        if move_type < 0.4:
            # Move 1: Find closest pair, swap with jitter or push apart
            iu = np.triu_indices(N, k=1)
            diff = c[iu[0]] - c[iu[1]]
            dists = np.sqrt(np.sum(diff**2, axis=1))
            min_idx = np.argmin(dists)
            i, j = iu[0][min_idx], iu[1][min_idx]
            
            c_new = c.copy()
            if random.random() < 0.5:
                # Swap positions with small jitter
                c_new[i] = c[j] + np.random.randn(3) * 0.005
                c_new[j] = c[i] + np.random.randn(3) * 0.005
            else:
                # Push apart
                d = c[i] - c[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    d = np.random.randn(3)
                    dist = np.linalg.norm(d)
                d = d / dist
                step = 0.003
                c_new[i] = c[i] + d * step
                c_new[j] = c[j] - d * step
            
            c_new = np.clip(c_new, -0.5, 0.5)
            
        elif move_type < 0.8:
            # Move 2: Small random perturbation of 1-2 spheres
            n_move = 1 if random.random() < 0.6 else 2
            idxs = random.sample(range(N), n_move)
            c_new = c.copy()
            for idx in idxs:
                c_new[idx] = c_new[idx] + np.random.randn(3) * pert_scale
            c_new = np.clip(c_new, -0.5, 0.5)
            
        else:
            # Move 3: Move most boundary-constrained sphere inward
            abs_c = np.abs(c)
            d_x = 0.5 - abs_c[:, 0]
            d_y = 0.5 - abs_c[:, 1]
            d_z = 0.5 - abs_c[:, 2]
            all_d = np.column_stack([d_x, d_y, d_z])
            flat_min = np.argmin(all_d)
            i = flat_min // 3
            axis = flat_min % 3
            
            c_new = c.copy()
            sign = 1 if c[i, axis] > 0 else -1
            c_new[i, axis] = c[i, axis] + sign * 0.005
            c_new = np.clip(c_new, -0.5, 0.5)
        
        new_r = _compute_radius(c_new)
        
        # Accept/reject
        if new_r >= current_r:
            c = c_new
            current_r = new_r
        else:
            delta = new_r - current_r
            if delta / T > -700:  # avoid underflow
                prob = math.exp(delta / T)
                if random.random() < prob:
                    c = c_new
                    current_r = new_r
        
        if current_r > best_r:
            best_r = current_r
            best_c = c.copy()
        
        T *= alpha
        if T < T1:
            break
    
    return best_c, best_r

def _local_separation(c, max_iters=5000):
    """
    Local refinement: iteratively separate the closest pair and fix boundary violations.
    """
    c = c.copy()
    current_r = _compute_radius(c)
    best_c = c.copy()
    best_r = current_r
    
    for it in range(max_iters):
        # Compute constraints
        iu = np.triu_indices(N, k=1)
        diff = c[iu[0]] - c[iu[1]]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        min_idx = np.argmin(dists)
        i, j = iu[0][min_idx], iu[1][min_idx]
        r_pair = dists[min_idx] / 2.0
        
        abs_c = np.abs(c)
        d_x = 0.5 - abs_c[:, 0]
        d_y = 0.5 - abs_c[:, 1]
        d_z = 0.5 - abs_c[:, 2]
        r_bound = np.min(np.minimum.reduce([d_x, d_y, d_z]))
        
        if r_pair <= r_bound:
            # Pairwise is limiting
            d = c[i] - c[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = np.random.randn(3)
                dist = np.linalg.norm(d)
            d = d / dist
            
            # Try several step sizes
            improved = False
            for step in [0.004, 0.002, 0.001, 0.0005]:
                c_new = c.copy()
                c_new[i] = np.clip(c_new[i] + d * step, -0.5, 0.5)
                c_new[j] = np.clip(c_new[j] - d * step, -0.5, 0.5)
                new_r = _compute_radius(c_new)
                if new_r > current_r:
                    c = c_new
                    current_r = new_r
                    improved = True
                    break
            
            if not improved:
                # Try moving just one sphere in a random direction
                for trial in range(3):
                    idx = i if random.random() < 0.5 else j
                    c_new = c.copy()
                    c_new[idx] = c_new[idx] + np.random.randn(3) * 0.003
                    c_new = np.clip(c_new, -0.5, 0.5)
                    new_r = _compute_radius(c_new)
                    if new_r > current_r:
                        c = c_new
                        current_r = new_r
                        improved = True
                        break
            
            if not improved:
                # Stuck, try a bigger random kick
                c_new = c.copy()
                idxs = random.sample(range(N), 3)
                for idx in idxs:
                    c_new[idx] = c_new[idx] + np.random.randn(3) * 0.01
                c_new = np.clip(c_new, -0.5, 0.5)
                new_r = _compute_radius(c_new)
                if new_r > current_r:
                    c = c_new
                    current_r = new_r
        else:
            # Boundary is limiting
            all_d = np.column_stack([d_x, d_y, d_z])
            flat_min = np.argmin(all_d)
            i = flat_min // 3
            axis = flat_min % 3
            
            sign = 1 if c[i, axis] > 0 else -1
            improved = False
            for step in [0.004, 0.002, 0.001]:
                c_new = c.copy()
                c_new[i, axis] = np.clip(c[i, axis] + sign * step, -0.5, 0.5)
                new_r = _compute_radius(c_new)
                if new_r > current_r:
                    c = c_new
                    current_r = new_r
                    improved = True
                    break
            
            if not improved:
                c_new = c.copy()
                idxs = random.sample(range(N), 2)
                for idx in idxs:
                    c_new[idx] = c_new[idx] + np.random.randn(3) * 0.008
                c_new = np.clip(c_new, -0.5, 0.5)
                new_r = _compute_radius(c_new)
                if new_r > current_r:
                    c = c_new
                    current_r = new_r
        
        if current_r > best_r:
            best_r = current_r
            best_c = c.copy()
    
    return best_c, best_r

def construct():
    """Construct the optimal arrangement of 100 equal spheres in a unit cube."""
    # Try multiple initializations and keep the best
    best_c = None
    best_r = -1.0
    
    for trial in range(5):
        if trial == 0:
            c = _init_grid()
        else:
            c = _init_random()
        
        # Phase 1: Coarse SA with larger steps
        c, r = _sa_swap_perturb(c, max_iters=10000, T0=0.02, T1=1e-5, alpha=0.9998, pert_scale=0.02)
        
        # Phase 2: Local separation
        c, r = _local_separation(c, max_iters=3000)
        
        # Phase 3: Fine SA with smaller steps
        c, r2 = _sa_swap_perturb(c, max_iters=10000, T0=0.005, T1=1e-7, alpha=0.9999, pert_scale=0.005)
        if r2 > r:
            r = r2
        
        # Phase 4: Final local separation
        c, r3 = _local_separation(c, max_iters=2000)
        if r3 > r:
            r = r3
        
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    return best_c
# EVOLVE-BLOCK-END
