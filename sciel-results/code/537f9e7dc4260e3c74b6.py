"""Improved program for packing 100 equal spheres in a unit cube using stochastic local search with swaps."""
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import random
import math
import time

def _pairwise_distances(c):
    """Compute all pairwise distances for an array of shape (N, 3)."""
    iu = np.triu_indices(N, k=1)
    diff = c[iu[0]] - c[iu[1]]
    dists = np.sqrt(np.sum(diff**2, axis=1))
    return iu, dists

def _min_pairwise_distance(c):
    """Return the minimum pairwise distance."""
    iu, dists = _pairwise_distances(c)
    return np.min(dists)

def _min_boundary_distance(c):
    """Return the minimum distance from any centre to the nearest face of the cube [-0.5, 0.5]^3."""
    abs_c = np.abs(c)
    dists_x = 0.5 - abs_c[:, 0]
    dists_y = 0.5 - abs_c[:, 1]
    dists_z = 0.5 - abs_c[:, 2]
    return np.min(np.minimum.reduce([dists_x, dists_y, dists_z]))

def _score(c):
    """Compute the radius: min(min_pairwise_dist/2, min_boundary_dist)."""
    r_pair = _min_pairwise_distance(c) / 2.0
    r_bound = _min_boundary_distance(c)
    return min(r_pair, r_bound)

def _initial_grid():
    """Create an initial grid placement with slight jitter."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c = grid[:N].copy()
    # Add small random jitter to break symmetry
    c += np.random.randn(N, 3) * 0.001
    c = np.clip(c, -0.5, 0.5)
    return c

def _stochastic_local_search(c, max_iters=50000, T0=0.01, T1=1e-8, alpha=0.9998, 
                               swap_prob=0.3, move_prob=0.5, perturb_prob=0.2,
                               step_scale=0.02):
    """
    Stochastic local search with swap-based moves and random perturbations.
    """
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    T = T0
    start_time = time.time()
    time_limit = 100  # seconds
    
    for it in range(max_iters):
        # Check time limit
        if time.time() - start_time > time_limit:
            break
        
        # Choose move type
        r = random.random()
        
        if r < swap_prob:
            # Swap two random points
            i, j = random.sample(range(N), 2)
            c_new = c.copy()
            c_new[i], c_new[j] = c[j].copy(), c[i].copy()
        elif r < swap_prob + move_prob:
            # Move one or two random points
            n_move = 1 if random.random() < 0.6 else 2
            idxs = random.sample(range(N), n_move)
            c_new = c.copy()
            for idx in idxs:
                delta = np.random.randn(3) * step_scale
                c_new[idx] = c_new[idx] + delta
                c_new[idx] = np.clip(c_new[idx], -0.5, 0.5)
        else:
            # Perturb: move a random subset of points with larger steps
            n_perturb = random.randint(3, 8)
            idxs = random.sample(range(N), n_perturb)
            c_new = c.copy()
            for idx in idxs:
                delta = np.random.randn(3) * step_scale * 2
                c_new[idx] = c_new[idx] + delta
                c_new[idx] = np.clip(c_new[idx], -0.5, 0.5)
        
        new_score = _score(c_new)
        
        # Acceptance criterion
        if new_score >= current_score:
            c = c_new
            current_score = new_score
        else:
            delta_score = new_score - current_score
            prob = math.exp(delta_score / T)
            if random.random() < prob:
                c = c_new
                current_score = new_score
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_c = c.copy()
        
        # Cool down
        T *= alpha
        
        if T < T1:
            break
    
    return best_c, best_score

def _refine_closest_pairs(c, max_iters=5000):
    """
    Refine by pushing apart the closest pairs of spheres.
    """
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    for it in range(max_iters):
        # Find the limiting constraint
        r_pair = _min_pairwise_distance(c) / 2.0
        r_bound = _min_boundary_distance(c)
        
        if r_pair <= r_bound:
            # Pairwise distance is limiting
            iu, dists = _pairwise_distances(c)
            min_idx = np.argmin(dists)
            i, j = iu[0][min_idx], iu[1][min_idx]
            
            # Push i and j apart
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            else:
                direction = diff / dist
            
            # Try multiple step sizes
            for move in [0.005, 0.002, 0.001, 0.0005]:
                c_new = c.copy()
                c_new[i] = np.clip(c_new[i] + direction * move, -0.5, 0.5)
                c_new[j] = np.clip(c_new[j] - direction * move, -0.5, 0.5)
                
                new_score = _score(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_c = c.copy()
                    break
        else:
            # Boundary is limiting
            abs_c = np.abs(c)
            dists_x = 0.5 - abs_c[:, 0]
            dists_y = 0.5 - abs_c[:, 1]
            dists_z = 0.5 - abs_c[:, 2]
            all_dists = np.column_stack([dists_x, dists_y, dists_z])
            min_idx = np.argmin(all_dists.flatten())
            i = min_idx // 3
            axis = min_idx % 3
            
            # Push point i away from the boundary
            sign = 1 if c[i, axis] > 0 else -1
            
            for move in [0.005, 0.002, 0.001, 0.0005]:
                c_new = c.copy()
                c_new[i, axis] = np.clip(c_new[i, axis] + sign * move, -0.5, 0.5)
                
                new_score = _score(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_c = c.copy()
                    break
        
        # Occasionally add random perturbation to escape local minima
        if it % 50 == 0:
            idx = random.randint(0, N - 1)
            c_new = c.copy()
            c_new[idx] = c_new[idx] + np.random.randn(3) * 0.015
            c_new[idx] = np.clip(c_new[idx], -0.5, 0.5)
            new_score = _score(c_new)
            if new_score > current_score:
                c = c_new
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_c = c.copy()
    
    return best_c, best_score

def construct():
    """Construct the optimal arrangement of 100 equal spheres in a unit cube."""
    # Start with a jittered grid
    c = _initial_grid()
    
    # Phase 1: Coarse stochastic local search with swaps
    c, score = _stochastic_local_search(c, max_iters=30000, T0=0.02, T1=1e-6, 
                                         alpha=0.9995, swap_prob=0.4, move_prob=0.4, 
                                         perturb_prob=0.2, step_scale=0.03)
    
    # Phase 2: Local refinement
    c, score = _refine_closest_pairs(c, max_iters=5000)
    
    # Phase 3: Fine stochastic local search
    c, score2 = _stochastic_local_search(c, max_iters=25000, T0=0.005, T1=1e-8, 
                                          alpha=0.9998, swap_prob=0.2, move_prob=0.6, 
                                          perturb_prob=0.2, step_scale=0.01)
    if score2 > score:
        score = score2
        c, score = _refine_closest_pairs(c, max_iters=3000)
    
    # Phase 4: Final fine-tuning
    c, score3 = _stochastic_local_search(c, max_iters=15000, T0=0.001, T1=1e-10, 
                                          alpha=0.9999, swap_prob=0.1, move_prob=0.7, 
                                          perturb_prob=0.2, step_scale=0.005)
    if score3 > score:
        score = score3
        c, score = _refine_closest_pairs(c, max_iters=2000)
    
    return c
# EVOLVE-BLOCK-END
