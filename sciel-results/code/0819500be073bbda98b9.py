import numpy as np
import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    points = list(range(v))
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Generate all possible k-subsets (blocks)
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Map each block to the list of subset indices it covers
    block_to_subsets = [[] for _ in range(num_blocks)]
    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            block_to_subsets[b_idx].append(s_idx)
            
    # Precompute subset_to_blocks for efficient gain updates
    subset_to_blocks = [[] for _ in range(num_subsets)]
    for b_idx in range(num_blocks):
        for s_idx in block_to_subsets[b_idx]:
            subset_to_blocks[s_idx].append(b_idx)

    def fast_greedy(seed):
        rng = random.Random(seed)
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        selected = []
        remaining = num_subsets
        gains = np.array([len(s) for s in block_to_subsets], dtype=np.int32)
        
        while remaining > 0:
            if time.time() > deadline - 2.0:
                break
            
            # Add noise for diversity
            noise = rng.uniform(0, 0.5, size=num_blocks)
            scores = gains.astype(np.float64) + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected.append(best_idx)
            
            b_subs = block_to_subsets[best_idx]
            newly_covered = []
            for s_idx in b_subs:
                if cover_count[s_idx] == 0:
                    newly_covered.append(s_idx)
                    remaining -= 1
                cover_count[s_idx] += 1
            
            if newly_covered:
                # Update gains for all blocks containing newly covered subsets
                for s_idx in newly_covered:
                    for b_idx in subset_to_blocks[s_idx]:
                        gains[b_idx] -= 1
                        
        return selected

    def local_search(blocks, seed):
        rng = random.Random(seed)
        # Calculate cover counts
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in blocks:
            for s_idx in block_to_subsets[b_idx]:
                cover_count[s_idx] += 1
                
        current_blocks = list(blocks)
        improved = True
        while improved:
            improved = False
            # Randomize order for stability
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 2.0:
                    break
                if i >= len(current_blocks):
                    continue
                    
                b_idx = current_blocks[i]
                b_subs = block_to_subsets[b_idx]
                
                # Check if redundant
                is_redundant = True
                for s_idx in b_subs:
                    if cover_count[s_idx] == 1:
                        is_redundant = False
                        break
                        
                if is_redundant:
                    # Remove it
                    for s_idx in b_subs:
                        cover_count[s_idx] -= 1
                    current_blocks.pop(i)
                    improved = True
                    # Restart loop to check for new redundancies
                    break
        
        return current_blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy + local search
    num_trials = 15
    for trial in range(num_trials):
        if time.time() > deadline - 5.0:
            break
            
        greedy_res = fast_greedy(trial * 100 + 42)
        min_res = local_search(greedy_res, trial * 200 + 84)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
            
    if best_blocks is None:
        # Fallback: just return a valid cover (maybe not optimal)
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        selected = []
        remaining = num_subsets
        gains = np.array([len(s) for s in block_to_subsets], dtype=np.int32)
        
        while remaining > 0 and time.time() < deadline - 1.0:
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0: break
            selected.append(best_idx)
            b_subs = block_to_subsets[best_idx]
            newly_covered = []
            for s_idx in b_subs:
                if cover_count[s_idx] == 0:
                    newly_covered.append(s_idx)
                    remaining -= 1
                cover_count[s_idx] += 1
            for s_idx in newly_covered:
                for b_idx in subset_to_blocks[s_idx]:
                    gains[b_idx] -= 1
                    
        best_blocks = selected

    # Convert indices to lists of points
    return [list(all_blocks[i]) for i in best_blocks]
