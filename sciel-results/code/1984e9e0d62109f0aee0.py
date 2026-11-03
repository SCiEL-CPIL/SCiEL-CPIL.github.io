"""Improved program for packing 100 equal spheres in a unit cube using simulated annealing."""
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import random
import math
import time

def _compute_score(c):
    """Compute the radius: min(min_pairwise_dist/2, min_boundary_dist)."""
    # Pairwise distances
    iu = np.triu_indices(N, k=1)
    diff = c[iu[0]] - c[iu[1]]
    dists = np.sqrt(np.sum(diff**2, axis=1))
    r_pair = np.min(dists) / 2.0
    
    # Boundary distances
    abs_c = np.abs(c)
    dists_x = 0.5 - abs_c[:, 0]
    dists_y = 0.5 - abs_c[:, 1]
    dists_z = 0.5 - abs_c[:, 2]
    r_bound = np.min(np.minimum.reduce([dists_x, dists_y, dists_z]))
    
    return min(r_pair, r_bound)

def _initial_grid():
    """Create an initial grid placement with 5x5x4 = 100 points."""
    # Use a 5x5x4 grid
    xs = (np.arange(5) + 0.5) / 5 - 0.5
    ys = (np.arange(5) + 0.5) / 5 - 0.5
    zs = (np.arange(4) + 0.5) / 4 - 0.5
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid[:N].copy()

def _repulsion_step(c, target_r, strength=1.0):
    """
    Apply repulsion forces to push spheres apart.
    Returns the updated configuration.
    """
    c = c.copy()
    
    # Compute pairwise distances
    iu = np.triu_indices(N, k=1)
    diff = c[iu[0]] - c[iu[1]]
    dists = np.sqrt(np.sum(diff**2, axis=1))
    
    # Find pairs that are too close (distance < 2 * target_r)
    threshold = 2.0 * target_r
    close_mask = dists < threshold
    close_idx = np.where(close_mask)[0]
    
    if len(close_idx) > 0:
        for idx in close_idx:
            i, j = iu[0][idx], iu[1][idx]
            d = c[i] - c[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            else:
                direction = d / dist
            
            # Repulsion force proportional to overlap
            overlap = threshold - dist
            force = strength * overlap * 0.5
            
            c[i] = np.clip(c[i] + direction * force, -0.5, 0.5)
            c[j] = np.clip(c[j] - direction * force, -0.5, 0.5)
    
    # Also handle boundary violations
    abs_c = np.abs(c)
    for i in range(N):
        for axis in range(3):
            dist_to_face = 0.5 - abs_c[i, axis]
            if dist_to_face < target_r:
                sign = 1 if c[i, axis] > 0 else -1
                force = strength * (target_r - dist_to_face) * 0.5
                c[i, axis] = np.clip(c[i, axis] + sign * force, -0.5, 0.5)
    
    return c

def _simulated_annealing_v2(c, max_iters=30000, T0=0.01, T1=1e-6, alpha=0.9995, step_scale=0.02):
    """
    Improved simulated annealing with better move generation.
    """
    c = c.copy()
    current_score = _compute_score(c)
    best_c = c.copy()
    best_score = current_score
    
    T = T0
    
    for it in range(max_iters):
        # Choose move type
        move_type = random.random()
        
        if move_type < 0.5:
            # Move a single random point
            idx = random.randint(0, N - 1)
            delta = np.random.randn(3) * step_scale
            c_new = c.copy()
            c_new[idx] = np.clip(c_new[idx] + delta, -0.5, 0.5)
        elif move_type < 0.8:
            # Move two random points
            idxs = random.sample(range(N), 2)
            c_new = c.copy()
            for idx in idxs:
                delta = np.random.randn(3) * step_scale
                c_new[idx] = np.clip(c_new[idx] + delta, -0.5, 0.5)
        else:
            # Swap two random points' positions (with small perturbation)
            i, j = random.sample(range(N), 2)
            c_new = c.copy()
            # Blend positions
            t = random.uniform(0.3, 0.7)
            new_i = t * c_new[j] + (1 - t) * c_new[i] + np.random.randn(3) * step_scale * 0.5
            new_j = t * c_new[i] + (1 - t) * c_new[j] + np.random.randn(3) * step_scale * 0.5
            c_new[i] = np.clip(new_i, -0.5, 0.5)
            c_new[j] = np.clip(new_j, -0.5, 0.5)
        
        new_score = _compute_score(c_new)
        
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

def _local_refine_v2(c, max_iters=10000):
    """
    Local refinement using repulsion forces and targeted moves.
    """
    c = c.copy()
    current_score = _compute_score(c)
    best_c = c.copy()
    best_score = current_score
    
    for it in range(max_iters):
        # Apply repulsion step
        c_rep = _repulsion_step(c, current_score, strength=0.5)
        rep_score = _compute_score(c_rep)
        
        if rep_score > current_score:
            c = c_rep
            current_score = rep_score
            if current_score > best_score:
                best_score = current_score
                best_c = c.copy()
        
        # Find the limiting constraint and try to fix it
        iu = np.triu_indices(N, k=1)
        diff = c[iu[0]] - c[iu[1]]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        r_pair = np.min(dists) / 2.0
        
        abs_c = np.abs(c)
        dists_x = 0.5 - abs_c[:, 0]
        dists_y = 0.5 - abs_c[:, 1]
        dists_z = 0.5 - abs_c[:, 2]
        r_bound = np.min(np.minimum.reduce([dists_x, dists_y, dists_z]))
        
        if r_pair <= r_bound:
            # Pairwise distance is limiting
            min_idx = np.argmin(dists)
            i, j = iu[0][min_idx], iu[1][min_idx]
            
            d = c[i] - c[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            else:
                direction = d / dist
            
            # Try several move sizes
            for move in [0.005, 0.002, 0.001]:
                c_new = c.copy()
                c_new[i] = np.clip(c_new[i] + direction * move, -0.5, 0.5)
                c_new[j] = np.clip(c_new[j] - direction * move, -0.5, 0.5)
                new_score = _compute_score(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_c = c.copy()
                    break
        else:
            # Boundary is limiting
            all_dists = np.column_stack([dists_x, dists_y, dists_z])
            min_idx = np.argmin(all_dists.flatten())
            i = min_idx // 3
            axis = min_idx % 3
            
            sign = 1 if c[i, axis] > 0 else -1
            for move in [0.005, 0.002, 0.001]:
                c_new = c.copy()
                c_new[i, axis] = np.clip(c_new[i, axis] + sign * move, -0.5, 0.5)
                new_score = _compute_score(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_c = c.copy()
                    break
        
        # Random perturbation to escape local minima
        if it % 50 == 0:
            idx = random.randint(0, N - 1)
            c_new = c.copy()
            c_new[idx] = c_new[idx] + np.random.randn(3) * 0.015
            c_new[idx] = np.clip(c_new[idx], -0.5, 0.5)
            new_score = _compute_score(c_new)
            if new_score > current_score:
                c = c_new
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_c = c.copy()
    
    return best_c, best_score

def construct():
    """Construct the optimal arrangement of 100 equal spheres in a unit cube."""
    # Start with a grid
    c = _initial_grid()
    
    # Phase 1: Simulated annealing with large steps
    c, score = _simulated_annealing_v2(c, max_iters=20000, T0=0.01, T1=1e-5, alpha=0.9997, step_scale=0.03)
    
    # Phase 2: Local refinement
    c, score = _local_refine_v2(c, max_iters=5000)
    
    # Phase 3: Simulated annealing with smaller steps
    c, score2 = _simulated_annealing_v2(c, max_iters=15000, T0=0.005, T1=1e-6, alpha=0.9998, step_scale=0.01)
    if score2 > score:
        score = score2
        c, score = _local_refine_v2(c, max_iters=3000)
    
    # Phase 4: Final fine-tuning
    c, score3 = _local_refine_v2(c, max_iters=3000)
    if score3 > score:
        score = score3
    
    return c
# EVOLVE-BLOCK-END
