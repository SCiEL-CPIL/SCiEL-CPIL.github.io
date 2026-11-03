"""Improved program for packing 100 equal spheres in a unit cube using simulated annealing."""
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import random
import math
import time

def _score_fast(c):
    """Fast vectorized score computation."""
    # Boundary distance
    abs_c = np.abs(c)
    boundary_dists = 0.5 - abs_c
    min_boundary = np.min(boundary_dists)
    
    # Pairwise distances - compute all pairs
    # Use broadcasting for memory efficiency with N=100
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Set diagonal to infinity
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_boundary)

def _score_with_details(c):
    """Score plus identifying the limiting constraint."""
    abs_c = np.abs(c)
    boundary_dists = 0.5 - abs_c
    min_boundary = np.min(boundary_dists)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_val = np.min(dists)
    min_pair = min_pair_val / 2.0
    
    score = min(min_pair, min_boundary)
    
    if min_pair <= min_boundary:
        # Find the pair
        idx = np.unravel_index(np.argmin(dists), dists.shape)
        return score, 'pair', idx
    else:
        # Find the boundary point
        flat_idx = np.argmin(boundary_dists)
        i = flat_idx // 3
        axis = flat_idx % 3
        return score, 'boundary', (i, axis)

def _initial_grid():
    """Create an initial grid placement with slight randomization."""
    m = 5  # 5x5x4 = 100
    ticks = (np.arange(5) + 0.5) / 5.0 - 0.5
    grid = []
    count = 0
    for z in ticks:
        for y in ticks:
            for x in ticks:
                if count < N:
                    grid.append([x, y, z])
                    count += 1
    c = np.array(grid, dtype=np.float64)
    # Add small random perturbation to break symmetry
    c += np.random.randn(N, 3) * 0.001
    c = np.clip(c, -0.5, 0.5)
    return c

def _simulated_annealing_v2(c, max_iters=30000, T0=0.008, T1=1e-7, alpha=0.9996, step_scale=0.02, time_limit=90):
    """
    Improved simulated annealing with adaptive step size and better acceptance.
    """
    c = c.copy()
    current_score = _score_fast(c)
    best_c = c.copy()
    best_score = current_score
    
    T = T0
    start_time = time.time()
    no_improve = 0
    step_scale_current = step_scale
    
    for it in range(max_iters):
        # Check time limit
        if time.time() - start_time > time_limit:
            break
        
        # Adaptive step size: increase if accepting too many, decrease if too few
        if it % 500 == 0 and it > 0:
            if no_improve > 400:
                step_scale_current *= 1.3  # Increase step to escape
            elif no_improve < 100:
                step_scale_current *= 0.9  # Decrease step to refine
        
        # Generate perturbation
        # Mix of single-point moves and multi-point moves
        r = random.random()
        if r < 0.5:
            # Single point move
            idx = random.randint(0, N - 1)
            c_new = c.copy()
            delta = np.random.randn(3) * step_scale_current
            c_new[idx] = np.clip(c_new[idx] + delta, -0.5, 0.5)
        elif r < 0.8:
            # Two point move
            idxs = random.sample(range(N), 2)
            c_new = c.copy()
            for idx in idxs:
                delta = np.random.randn(3) * step_scale_current
                c_new[idx] = np.clip(c_new[idx] + delta, -0.5, 0.5)
        else:
            # Swap-like move: exchange positions with perturbation
            i, j = random.sample(range(N), 2)
            c_new = c.copy()
            tmp = c_new[i].copy()
            c_new[i] = c_new[j] + np.random.randn(3) * step_scale_current * 0.5
            c_new[j] = tmp + np.random.randn(3) * step_scale_current * 0.5
            c_new[i] = np.clip(c_new[i], -0.5, 0.5)
            c_new[j] = np.clip(c_new[j], -0.5, 0.5)
        
        new_score = _score_fast(c_new)
        
        # Acceptance
        if new_score >= current_score:
            c = c_new
            current_score = new_score
            no_improve = 0
        else:
            delta_score = new_score - current_score
            if delta_score > -T * 50:  # Avoid underflow
                prob = math.exp(delta_score / T)
                if random.random() < prob:
                    c = c_new
                    current_score = new_score
            no_improve += 1
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_c = c.copy()
            no_improve = 0
        
        # Cool down
        T *= alpha
        if T < T1:
            T = T1
    
    return best_c, best_score

def _local_refine_v2(c, max_iters=5000, time_limit=15):
    """
    Local refinement focusing on the limiting constraint.
    """
    c = c.copy()
    current_score = _score_fast(c)
    best_c = c.copy()
    best_score = current_score
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        score, constraint_type, info = _score_with_details(c)
        
        if constraint_type == 'pair':
            i, j = info
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            else:
                direction = diff / dist
            
            # Try multiple move sizes
            accepted = False
            for move in [0.008, 0.004, 0.002, 0.001]:
                c_new = c.copy()
                c_new[i] = np.clip(c_new[i] + direction * move, -0.5, 0.5)
                c_new[j] = np.clip(c_new[j] - direction * move, -0.5, 0.5)
                new_score = _score_fast(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    accepted = True
                    break
            
            if not accepted:
                # Try moving only one point
                for idx in [i, j]:
                    for move in [0.004, 0.002]:
                        c_new = c.copy()
                        if idx == i:
                            c_new[idx] = np.clip(c_new[idx] + direction * move, -0.5, 0.5)
                        else:
                            c_new[idx] = np.clip(c_new[idx] - direction * move, -0.5, 0.5)
                        new_score = _score_fast(c_new)
                        if new_score > current_score:
                            c = c_new
                            current_score = new_score
                            accepted = True
                            break
                    if accepted:
                        break
        else:
            # Boundary constraint
            i, axis = info
            sign = 1 if c[i, axis] >= 0 else -1
            
            accepted = False
            for move in [0.008, 0.004, 0.002, 0.001]:
                c_new = c.copy()
                c_new[i, axis] = np.clip(c_new[i, axis] + sign * move, -0.5, 0.5)
                new_score = _score_fast(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    accepted = True
                    break
            
            if not accepted:
                # Try moving in a different direction
                for _ in range(3):
                    c_new = c.copy()
                    delta = np.random.randn(3) * 0.005
                    c_new[i] = np.clip(c_new[i] + delta, -0.5, 0.5)
                    new_score = _score_fast(c_new)
                    if new_score > current_score:
                        c = c_new
                        current_score = new_score
                        accepted = True
                        break
        
        if current_score > best_score:
            best_score = current_score
            best_c = c.copy()
        
        # Occasional random kick to escape local minima
        if it % 200 == 0:
            idx = random.randint(0, N - 1)
            c_new = c.copy()
            c_new[idx] = np.clip(c_new[idx] + np.random.randn(3) * 0.015, -0.5, 0.5)
            new_score = _score_fast(c_new)
            if new_score > current_score:
                c = c_new
                current_score = new_score
    
    return best_c, best_score

def construct():
    """Construct the optimal arrangement of 100 equal spheres in a unit cube."""
    start_time = time.time()
    time_budget = 110  # Leave 10s buffer
    
    # Multi-start: try several initial configurations
    best_c = None
    best_score = -1
    
    # Start 1: Grid
    c = _initial_grid()
    c, score = _simulated_annealing_v2(c, max_iters=20000, T0=0.006, T1=1e-7, alpha=0.9997, step_scale=0.018, time_limit=40)
    c, score = _local_refine_v2(c, max_iters=4000, time_limit=10)
    if score > best_score:
        best_score = score
        best_c = c.copy()
    
    # Start 2: Different random initialization
    if time.time() - start_time < time_budget - 30:
        c = np.random.randn(N, 3) * 0.4
        c = np.clip(c, -0.5, 0.5)
        c, score = _simulated_annealing_v2(c, max_iters=20000, T0=0.01, T1=1e-7, alpha=0.9996, step_scale=0.025, time_limit=30)
        c, score = _local_refine_v2(c, max_iters=3000, time_limit=8)
        if score > best_score:
            best_score = score
            best_c = c.copy()
    
    # Final refinement on best
    if time.time() - start_time < time_budget - 10:
        c, score = _simulated_annealing_v2(best_c, max_iters=10000, T0=0.003, T1=1e-8, alpha=0.9998, step_scale=0.008, time_limit=20)
        c, score = _local_refine_v2(c, max_iters=3000, time_limit=8)
        if score > best_score:
            best_score = score
            best_c = c.copy()
    
    return best_c
# EVOLVE-BLOCK-END
