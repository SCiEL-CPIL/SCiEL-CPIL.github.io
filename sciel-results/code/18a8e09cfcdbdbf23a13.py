import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Precomputation
    points = range(v)
    t_subsets = list(itertools.combinations(points, t))
    k_subsets = list(itertools.combinations(points, k))
    
    num_t = len(t_subsets)
    num_k = len(k_subsets)
    
    # Map t-subset to index
    t_index = {s: i for i, s in enumerate(t_subsets)}
    
    # For each k-subset, find the indices of t-subsets it covers
    # block_to_t_indices[i] = list of t-subset indices covered by k-subset i
    block_to_t_indices = [[] for _ in range(num_k)]
    # For each t-subset, find the indices of k-subsets that cover it
    # t_to_block_indices[j] = list of k-subset indices covering t-subset j
    t_to_block_indices = [[] for _ in range(num_t)]
    
    for i, ks in enumerate(k_subsets):
        for ts in itertools.combinations(ks, t):
            j = t_index[ts]
            block_to_t_indices[i].append(j)
            t_to_block_indices[j].append(i)
            
    # Initialize counts
    # count_t[j] = number of blocks in current cover that cover t-subset j
    # count_k[i] = number of t-subsets in current cover that are covered ONLY by block i? No.
    # For greedy, we need gain = number of UNCOVERED t-subsets covered by block i.
    
    # Data structures for Dynamic Greedy
    # uncovered_count[j]: 1 if uncovered, 0 if covered
    # block_uncovered_gain[i]: number of uncovered subsets covered by block i
    
    best_blocks = None
    best_count = float('inf')
    
    num_restarts = 1000 # Will be limited by time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize state
        uncovered_count = [1] * num_t
        # gain[i] = sum(uncovered_count[j] for j in block_to_t_indices[i])
        gain = [0] * num_k
        
        # Initial gain calculation
        for i in range(num_k):
            g = 0
            for j in block_to_t_indices[i]:
                g += uncovered_count[j]
            gain[i] = g
            
        current_blocks = []
        used_blocks = [False] * num_k
        
        # Active blocks: those with gain > 0
        # We can just iterate all, but skip if gain==0
        # To speed up, we could maintain a heap or list, but full scan is O(num_k) = 75k.
        # 75k * 300 blocks = 22.5M ops per restart. 1000 restarts = 22.5B. Too slow.
        
        # Optimization: Only consider blocks that cover at least one uncovered subset.
        # We can maintain a list of active block indices.
        
        active_blocks = list(range(num_k))
        
        while True:
            if time.time() > deadline:
                break
                
            # Find max gain in active_blocks
            max_gain = 0
            best_idx = -1
            # Random tie-breaking
            candidates = []
            
            # To speed up, we can sample if active_blocks is large?
            # Or just scan.
            # Let's scan.
            for i in active_blocks:
                if gain[i] > max_gain:
                    max_gain = gain[i]
                    best_idx = i
                    candidates = [i]
                elif gain[i] == max_gain and gain[i] > 0:
                    candidates.append(i)
                    
            if max_gain == 0:
                break # All covered or no blocks available
                
            chosen = random.choice(candidates)
            current_blocks.append(chosen)
            used_blocks[chosen] = True
            
            # Update gains and uncovered counts
            # For each t-subset j covered by chosen:
            #   if uncovered_count[j] == 1:
            #       uncovered_count[j] = 0
            #       for each block i covering j:
            #           gain[i] -= 1
            #           if gain[i] == 0:
            #               remove i from active_blocks
            #   else:
            #       uncovered_count[j] -= 1
            #       for each block i covering j:
            #           gain[i] -= 1
            
            # This inner loop is expensive.
            # Total work per block addition: sum over its t-subsets of (1 + num_blocks_covering_subset)
            # num_blocks_covering_subset is ~364.
            # t-subsets per block is C(8,5)=56.
            # 56 * 365 ~ 20k ops per block.
            # 300 blocks * 20k = 6M ops per restart.
            # 1000 restarts = 6B ops. Still might be too slow in Python.
            
            # Alternative: Don't update all gains.
            # Just recalculate gain for active blocks? No.
            
            # Let's try to optimize the update.
            # We can just mark the block as used.
            # And in the selection loop, we can calculate gain on the fly?
            # No, that's O(num_k * C(8,5)) = 75k * 56 = 4.2M per step. 300 steps = 1.2B. Too slow.
            
            # The dynamic update is the way.
            # But we need to remove from active_blocks efficiently.
            # Using a list and `remove` is O(N).
            # We can use a boolean array `is_active` and just skip inactive in the scan.
            
            is_active = [True] * num_k
            is_active[chosen] = False
            
            for j in block_to_t_indices[chosen]:
                if uncovered_count[j] == 1:
                    uncovered_count[j] = 0
                    # This subset is now covered.
                    # Decrease gain for all blocks covering this subset.
                    for i in t_to_block_indices[j]:
                        if is_active[i]:
                            gain[i] -= 1
                            if gain[i] == 0:
                                is_active[i] = False
                else:
                    uncovered_count[j] -= 1
                    for i in t_to_block_indices[j]:
                        if is_active[i]:
                            gain[i] -= 1
                            
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            # Check validity (should be valid by construction)
            best_blocks = [list(k_subsets[i]) for i in current_blocks]
            
        # Local Search: Redundancy Removal
        # We need to re-verify if we can remove any block.
        # We'll use the same gain logic but in reverse?
        # Simpler: Just try removing each block and checking if cover is still valid.
        # To check validity, we can recompute coverage counts.
        # This is O(N * C(8,5)) per removal check.
        # N=300, C(8,5)=56. 16.8k ops.
        # 300 checks = 5M ops. Fast.
        
        if best_blocks is not None and len(current_blocks) == best_count:
            # Try to polish current_blocks
            cover_indices = set(current_blocks)
            # Compute coverage count for each t-subset
            cov_count = [0] * num_t
            for i in current_blocks:
                for j in block_to_t_indices[i]:
                    cov_count[j] += 1
                    
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for idx, b_idx in enumerate(current_blocks):
                    # Try removing b_idx
                    # Check if any subset in b_idx has cov_count == 1
                    redundant = False
                    for j in block_to_t_indices[b_idx]:
                        if cov_count[j] == 1:
                            redundant = True
                            break
                            
                    if not redundant:
                        # Safe to remove
                        current_blocks.pop(idx)
                        cover_indices.remove(b_idx)
                        for j in block_to_t_indices[b_idx]:
                            cov_count[j] -= 1
                        improved = True
                        break # Restart loop as indices changed
                        # Actually, just continue to next block?
                        # If we remove, indices shift. Break and restart is safer.
                        
            if len(current_blocks) < best_count:
                best_count = len(current_blocks)
                best_blocks = [list(k_subsets[i]) for i in current_blocks]

    return best_blocks
