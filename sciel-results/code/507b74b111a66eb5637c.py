import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin for 150s limit

    # 1. Precomputation
    # All t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    num_t = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}
    
    # All k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Coverage Matrix M: (num_blocks, num_t)
    # M[i, j] = 1 if block i covers tsub j
    # Using uint8 for memory efficiency and speed
    M = np.zeros((num_blocks, num_t), dtype=np.uint8)
    
    # Fill M
    # This is the most expensive part of precomputation, but it's a one-time cost.
    # 50388 blocks * 35 tsubs/block = ~1.7M operations. Very fast.
    for i, block in enumerate(all_blocks):
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            M[i, idx] = 1
    
    best_blocks = None
    best_count = float('inf')
    
    # Helper to get block list from indices
    def get_blocks(indices):
        return [list(all_blocks[i]) for i in indices]

    def greedy_with_noise(seed):
        """
        Greedy set cover with random noise to explore different solutions.
        """
        rng = random.Random(seed)
        uncovered = np.ones(num_t, dtype=np.uint8)
        selected_indices = []
        selected_mask = np.zeros(num_blocks, dtype=bool)
        
        # Optional: Precompute a "diversity" score or just rely on noise
        # We add small random noise to gains to break ties and avoid deterministic local minima
        
        while np.any(uncovered):
            if time.time() > deadline - 10.0:
                break
                
            # Calculate gains: number of uncovered tsubs covered by each block
            gains = M @ uncovered
            
            # Exclude already selected blocks
            gains[selected_mask] = -1
            
            # Add small random noise to break ties
            # Noise magnitude should be small, e.g., 0 to 0.5
            noise = np.random.RandomState(seed).rand(num_blocks) * 0.1
            gains = gains + noise
            
            best_idx = np.argmax(gains)
            
            if gains[best_idx] <= 0:
                break
                
            selected_indices.append(best_idx)
            selected_mask[best_idx] = True
            
            # Update uncovered
            # M[best_idx] is a vector of 0s and 1s
            # We want to set uncovered[j] = 0 if M[best_idx, j] == 1
            # Using boolean indexing
            row = M[best_idx]
            covered_indices = np.where(row == 1)[0]
            uncovered[covered_indices] = 0
            
        return selected_indices

    def local_optimize(indices, seed):
        """
        Local search: Remove redundant blocks and try swaps.
        """
        if not indices:
            return []
            
        rng = random.Random(seed)
        indices_set = set(indices)
        
        # 1. Redundancy Removal
        # Calculate coverage counts
        coverage_counts = np.zeros(num_t, dtype=np.int32)
        for idx in indices:
            coverage_counts[M[idx]] += 1
            
        changed = True
        while changed:
            changed = False
            if time.time() > deadline - 5.0:
                break
                
            # Find redundant blocks: blocks where all covered tsubs have count > 1
            redundant = []
            for idx in indices:
                row = M[idx]
                covered_tsubs = np.where(row == 1)[0]
                if len(covered_tsubs) == 0:
                    continue
                if np.all(coverage_counts[covered_tsubs] > 1):
                    redundant.append(idx)
                    
            if redundant:
                # Remove one random redundant block
                idx_to_remove = rng.choice(redundant)
                indices.remove(idx_to_remove)
                indices_set.remove(idx_to_remove)
                coverage_counts[M[idx_to_remove]] -= 1
                changed = True
                
        # 2. Local Swap Search
        # Try to replace a selected block with a non-selected block
        # Goal: Maintain coverage, potentially allow further reductions
        # Simplified: If we can swap block A (selected) with block B (unselected)
        # such that B covers all tsubs that A uniquely covers, and B might cover others,
        # we might enable further redundancy removal.
        
        # This is complex to do efficiently. 
        # Alternative: Just try to remove more blocks after a perturbation.
        
        return indices

    # 3. Main Loop: Multiple Runs
    # We run several greedy constructions with different noise seeds
    # Then we polish them with local optimization
    
    num_runs = 50 # Max runs, limited by time
    
    for run_id in range(num_runs):
        if time.time() > deadline - 15.0:
            break
            
        seed = run_id * 12345 + 67890
        indices = greedy_with_noise(seed)
        
        if len(indices) < best_count:
            best_count = len(indices)
            # Store a copy
            best_indices_copy = indices[:]
            
        # Local Optimization
        # Only optimize if we have time
        if time.time() < deadline - 10.0:
            optimized_indices = local_optimize(indices, seed + 999)
            if len(optimized_indices) < best_count:
                best_count = len(optimized_indices)
                best_indices_copy = optimized_indices[:]
                
        # Early stopping if we hit a very good score
        # Lower bound is 131. If we get close to 135, we might stop if time is tight
        if best_count <= 140 and time.time() > deadline - 30.0:
             # Continue to try for better, but be cautious
             pass

    # Fallback if no blocks found (should not happen)
    if best_indices_copy is None:
        # Simple greedy without noise
        uncovered = np.ones(num_t, dtype=np.uint8)
        selected_indices = []
        selected_mask = np.zeros(num_blocks, dtype=bool)
        while np.any(uncovered):
            gains = M @ uncovered
            gains[selected_mask] = -1
            best_idx = np.argmax(gains)
            if gains[best_idx] <= 0:
                break
            selected_indices.append(best_idx)
            selected_mask[best_idx] = True
            row = M[best_idx]
            covered_indices = np.where(row == 1)[0]
            uncovered[covered_indices] = 0
        best_indices_copy = selected_indices
        best_count = len(best_indices_copy)

    return get_blocks(best_indices_copy)
