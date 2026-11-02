import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    points = list(range(v))
    
    # Generate all t-subsets
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    # Map t-subset to index
    ts_to_idx = {comb: i for i, comb in enumerate(all_ts)}
    
    # Generate all k-blocks
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # Precompute subsets for each block
    # Each block covers C(k, t) subsets
    c_k_t = 1
    for i in range(t):
        c_k_t = c_k_t * (k - i) // (i + 1)
        
    all_block_subs = np.zeros((n_ks, c_k_t), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = [ts_to_idx[comb] for comb in itertools.combinations(block, t)]
        all_block_subs[i] = subs
        
    def run_greedy():
        covered = np.zeros(n_ts, dtype=bool)
        selected_blocks = []
        remaining = n_ts
        
        while remaining > 0:
            if time.time() - start_time > time_limit - 10.0:
                return selected_blocks
            
            # Calculate gains: number of currently uncovered subsets covered by each block
            # ~covered[all_block_subs] is a boolean array (n_ks, c_k_t)
            # Summing along axis 1 gives the gain for each block
            gains = np.sum(~covered[all_block_subs], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break # Should not happen if covering is possible
            
            # Find all blocks with max gain
            max_gain_blocks = np.where(gains == max_gain)[0]
            
            # Pick one randomly to diversify
            best_idx = max_gain_blocks[random.randint(0, len(max_gain_blocks) - 1)]
            
            selected_blocks.append(best_idx)
            
            # Update covered
            newly_covered = ~covered[all_block_subs[best_idx]]
            new_count = np.sum(newly_covered)
            remaining -= new_count
            
            # Update covered array
            covered[all_block_subs[best_idx]] = True
            
        return selected_blocks

    def greedy_deletion(selected_blocks):
        if not selected_blocks:
            return []
            
        # Build coverage count
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b in selected_blocks:
            coverage_count[all_block_subs[b]] += 1
            
        current_blocks = list(selected_blocks)
        
        # Iterate until no more deletions
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 5.0:
                break
            improved = False
            new_blocks = []
            # Shuffle to make deletion order random
            random.shuffle(current_blocks)
            
            for b in current_blocks:
                # Check if all subsets in b are covered by other blocks
                # i.e., coverage_count[subs] > 1 for all subs in b
                if np.all(coverage_count[all_block_subs[b]] > 1):
                    # Remove b
                    coverage_count[all_block_subs[b]] -= 1
                    improved = True
                else:
                    new_blocks.append(b)
            current_blocks = new_blocks
            
        return current_blocks

    best_solution = None
    best_score = float('inf')
    
    # Run multiple restarts
    # Estimate time per greedy run. 
    # If it's too slow, we reduce restarts.
    # For now, try 50 restarts.
    
    for seed in range(50):
        if time.time() - start_time > time_limit - 10.0:
            break
            
        random.seed(seed)
        
        # Run greedy
        blocks = run_greedy()
        
        if not blocks:
            continue
            
        # Refine with deletion
        blocks = greedy_deletion(blocks)
        
        # Try a few more deletion passes
        for _ in range(2):
            new_blocks = greedy_deletion(blocks)
            if len(new_blocks) < len(blocks):
                blocks = new_blocks
            else:
                break
        
        score = len(blocks)
        if score < best_score:
            best_score = score
            best_solution = blocks
            
    if best_solution is None:
        # Fallback
        random.seed(0)
        best_solution = run_greedy()
        best_solution = greedy_deletion(best_solution)
        
    # Convert indices back to lists of points
    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
