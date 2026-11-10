"""Hypergraph Ramsey R(4,5;3) construction via simulated annealing on triple colorings."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115  # Safety margin from 150s kill
    
    target_n = 34
    M = target_n * (target_n - 1) * (target_n - 2) // 6  # 5984 edges
    
    # Precompute edge indices
    edge_index = {}
    idx = 0
    for a in range(target_n):
        for b in range(a + 1, target_n):
            for c in range(b + 1, target_n):
                edge_index[(a, b, c)] = idx
                idx += 1
                
    # Precompute edge indices for all K4s and K5s
    k4_edges = []
    for quad in combinations(range(target_n), 4):
        edges = []
        for triple in combinations(quad, 3):
            edges.append(edge_index[triple])
        k4_edges.append(edges)
        
    k5_edges = []
    for quint in combinations(range(target_n), 5):
        edges = []
        for triple in combinations(quint, 3):
            edges.append(edge_index[triple])
        k5_edges.append(edges)
        
    k4_np = np.array(k4_edges, dtype=np.int32)
    k5_np = np.array(k5_edges, dtype=np.int32)
    
    # Count of K4s and K5s
    num_k4 = len(k4_edges)
    num_k5 = len(k5_edges)
    
    # Helper to compute objective (number of violations)
    def compute_objective(colors):
        # Red K4: all 4 edges are 0 (red)
        k4_sums = np.sum(colors[k4_np], axis=1)
        red_k4_count = np.sum(k4_sums == 0)
        # Blue K5: all 10 edges are 1 (blue)
        k5_sums = np.sum(colors[k5_np], axis=1)
        blue_k5_count = np.sum(k5_sums == 10)
        return red_k4_count + blue_k5_count
    
    # Helper to check validity
    def is_valid(colors):
        k4_sums = np.sum(colors[k4_np], axis=1)
        if np.any(k4_sums == 0):
            return False
        k5_sums = np.sum(colors[k5_np], axis=1)
        if np.any(k5_sums == 10):
            return False
        return True
    
    # Simulated annealing approach
    # Objective: minimize total violations (red K4s + blue K5s)
    
    # Initialize with random coloring
    colors = np.random.randint(0, 2, M, dtype=np.int8)
    current_obj = compute_objective(colors)
    best_obj = current_obj
    best_colors = colors.copy()
    
    # Simulated annealing parameters
    T_start = 10.0
    T_end = 0.01
    cooling_rate = 0.999
    
    # Number of steps per temperature
    steps_per_temp = 500
    
    # Track progress
    no_improve_steps = 0
    max_no_improve = 5000
    
    T = T_start
    step = 0
    
    while T > T_end and time.time() - start_time < deadline - 5:
        # Check time more frequently
        if step % 100 == 0 and time.time() - start_time > deadline - 10:
            break
            
        # Pick a random edge to flip
        edge_idx = np.random.randint(0, M)
        old_val = colors[edge_idx]
        colors[edge_idx] = 1 - old_val
        
        # Compute new objective efficiently
        # We only need to recompute K4s and K5s that contain this edge
        # For simplicity in this version, we recompute fully but could optimize
        new_obj = compute_objective(colors)
        delta_obj = new_obj - current_obj
        
        # Accept if improvement or with probability exp(-delta/T) for worsening
        if delta_obj < 0 or random.random() < np.exp(-delta_obj / T):
            current_obj = new_obj
            no_improve_steps = 0
            if new_obj < best_obj:
                best_obj = new_obj
                best_colors = colors.copy()
                if best_obj == 0:
                    # Found valid coloring!
                    return {'n': target_n, 'colors': best_colors.tolist()}
        else:
            # Reject: flip back
            colors[edge_idx] = old_val
            no_improve_steps += 1
        
        # Cool down
        T *= cooling_rate
        step += 1
        
        # If stuck too long, restart with new random initialization
        if no_improve_steps > max_no_improve:
            if time.time() - start_time > deadline - 15:
                break
            colors = np.random.randint(0, 2, M, dtype=np.int8)
            current_obj = compute_objective(colors)
            T = T_start
            no_improve_steps = 0
    
    # Check if best solution is valid
    if best_obj == 0:
        return {'n': target_n, 'colors': best_colors.tolist()}
    
    # If SA didn't find valid, try local search from best state
    colors = best_colors.copy()
    current_obj = best_obj
    
    # Local search: flip edges in violating structures
    local_steps = 0
    max_local_steps = 50000
    
    while current_obj > 0 and local_steps < max_local_steps and time.time() - start_time < deadline - 3:
        # Find violating structures
        k4_sums = np.sum(colors[k4_np], axis=1)
        k5_sums = np.sum(colors[k5_np], axis=1)
        
        bad_k4_indices = np.where(k4_sums == 0)[0]
        bad_k5_indices = np.where(k5_sums == 10)[0]
        
        if len(bad_k4_indices) == 0 and len(bad_k5_indices) == 0:
            current_obj = 0
            break
        
        # Collect edges from bad structures (sample to limit size)
        bad_edges = set()
        if len(bad_k4_indices) > 0:
            sample_size = min(20, len(bad_k4_indices))
            sample_k4 = np.random.choice(bad_k4_indices, sample_size, replace=False)
            for idx in sample_k4:
                bad_edges.update(k4_np[idx].tolist())
        if len(bad_k5_indices) > 0:
            sample_size = min(20, len(bad_k5_indices))
            sample_k5 = np.random.choice(bad_k5_indices, sample_size, replace=False)
            for idx in sample_k5:
                bad_edges.update(k5_np[idx].tolist())
                
        if not bad_edges:
            break
            
        # Flip a random edge from bad set
        target_edge = random.choice(list(bad_edges))
        old_val = colors[target_edge]
        colors[target_edge] = 1 - old_val
        
        new_obj = compute_objective(colors)
        if new_obj < current_obj:
            current_obj = new_obj
            if current_obj == 0:
                break
        else:
            # Revert
            colors[target_edge] = old_val
            # If we can't improve, try a different strategy: flip random edge
            if local_steps % 1000 == 0:
                random_edge = np.random.randint(0, M)
                colors[random_edge] ^= 1
                new_obj = compute_objective(colors)
                if new_obj < current_obj:
                    current_obj = new_obj
                else:
                    colors[random_edge] ^= 1  # revert
            
        local_steps += 1
    
    if current_obj == 0:
        return {'n': target_n, 'colors': colors.tolist()}
    
    # Fallback: return known valid small solution
    n_fallback = 9
    return {'n': n_fallback, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_fallback), 3)]}
# EVOLVE-BLOCK-END
