import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit

    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
    
    # Compute conflict graph: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to sparse-like structure for faster neighbor lookups
    # We'll use the dense matrix but optimize the greedy MIS
    
    best_selected = None
    best_count = 0
    
    # Multiple strategies for finding large independent sets
    
    # Strategy 1: Minimum degree greedy with randomization
    for restart in range(30):
        if time.time() > deadline - 15:
            break
        
        rng = np.random.default_rng(1000 + restart * 7)
        alive = np.ones(N4, dtype=bool)
        selected = np.zeros(N4, dtype=bool)
        
        # Use a priority queue approach: repeatedly pick min-degree alive vertex
        # For efficiency, we'll use a simpler approach with periodic re-sorting
        
        alive_indices = np.arange(N4)
        
        while len(alive_indices) > 0:
            if time.time() > deadline - 10:
                break
            
            # Compute degrees among alive vertices
            # This is expensive, so we use a heuristic: pick randomly from low-degree candidates
            # First, compute approximate degrees
            if len(alive_indices) > 50:
                # Sample-based degree estimation
                sample_size = min(100, len(alive_indices))
                sample = rng.choice(len(alive_indices), size=sample_size, replace=False)
                sampled_alive = alive_indices[sample]
                
                # Compute degrees for sampled vertices
                sample_conflicts = conflicts[sampled_alive][:, alive_indices]
                sample_degrees = sample_conflicts.sum(axis=1)
                
                # Find min degree in sample
                min_deg = np.min(sample_degrees)
                # Get all alive vertices with degree close to min
                threshold = min_deg + max(1, int(0.1 * N4))
                
                # Compute degrees for all alive (expensive but necessary for accuracy)
                alive_conflicts = conflicts[alive_indices][:, alive_indices]
                alive_degrees = alive_conflicts.sum(axis=1)
                
                cands = alive_indices[alive_degrees <= threshold]
            else:
                alive_conflicts = conflicts[alive_indices][:, alive_indices]
                alive_degrees = alive_conflicts.sum(axis=1)
                min_deg = np.min(alive_degrees)
                cands = alive_indices[alive_degrees == min_deg]
            
            if len(cands) == 0:
                cands = alive_indices
            
            chosen_idx = cands[rng.integers(len(cands))]
            chosen_pos = np.where(alive_indices == chosen_idx)[0][0]
            
            selected[chosen_idx] = True
            
            # Remove chosen and its neighbors from alive
            to_remove = conflicts[chosen_idx]
            to_remove[chosen_idx] = True
            alive_indices = alive_indices[~to_remove[alive_indices]]
        
        count = selected.sum()
        if count > best_count:
            best_count = count
            best_selected = selected.copy()
    
    # Strategy 2: Randomized greedy with different ordering
    for restart in range(20):
        if time.time() > deadline - 15:
            break
        
        rng = np.random.default_rng(5000 + restart * 13)
        alive = np.ones(N4, dtype=bool)
        selected = np.zeros(N4, dtype=bool)
        
        order = rng.permutation(N4)
        
        for i in order:
            if time.time() > deadline - 10:
                break
            if not alive[i]:
                continue
            selected[i] = True
            to_remove = conflicts[i].copy()
            to_remove[i] = True
            alive[to_remove] = False
        
        count = selected.sum()
        if count > best_count:
            best_count = count
            best_selected = selected.copy()
    
    # Strategy 3: Local search improvement on best found so far
    if best_selected is not None and time.time() < deadline - 20:
        improved = best_selected.copy()
        improved_count = improved.sum()
        
        # Try to add vertices that don't conflict with any selected vertex
        # And try swaps: remove one selected, add two non-selected
        
        # First pass: add any compatible non-selected vertices
        for i in range(N4):
            if time.time() > deadline - 15:
                break
            if improved[i]:
                continue
            # Check if i conflicts with any selected vertex
            if not np.any(conflicts[i] & improved):
                improved[i] = True
                improved_count += 1
        
        if improved_count > best_count:
            best_count = improved_count
            best_selected = improved.copy()
        
        # Second pass: try 1-out 2-in swaps
        selected_set = np.where(best_selected)[0]
        non_selected_set = np.where(~best_selected)[0]
        
        for trial in range(50):
            if time.time() > deadline - 15:
                break
            
            # Pick a random selected vertex to remove
            if len(selected_set) == 0:
                break
            rem_idx = np.random.randint(len(selected_set))
            rem_v = selected_set[rem_idx]
            
            # Find non-selected vertices that don't conflict with (selected - rem_v)
            # i.e., they only conflict with rem_v among selected vertices
            temp_selected = best_selected.copy()
            temp_selected[rem_v] = False
            
            compatible = non_selected_set[~np.any(conflicts[non_selected_set][:, temp_selected], axis=1)]
            
            if len(compatible) < 2:
                continue
            
            # Pick two compatible vertices
            if len(compatible) == 2:
                add_v1, add_v2 = compatible[0], compatible[1]
            else:
                picks = np.random.choice(len(compatible), size=2, replace=False)
                add_v1, add_v2 = compatible[picks[0]], compatible[picks[1]]
            
            # Check that add_v1 and add_v2 don't conflict with each other
            if conflicts[add_v1, add_v2]:
                continue
            
            # Apply the swap
            new_count = best_count - 1 + 2
            if new_count > best_count:
                best_selected[rem_v] = False
                best_selected[add_v1] = True
                best_selected[add_v2] = True
                best_count = new_count
                selected_set = np.where(best_selected)[0]
                non_selected_set = np.where(~best_selected)[0]
    
    result_v4 = vectors_v4[best_selected]
    
    # Add all V1 vectors (they're all compatible with V4 and each other)
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
