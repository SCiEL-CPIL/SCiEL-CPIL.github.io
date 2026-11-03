import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 148.0  # 148s limit (2s safety margin)
    
    def remaining_time():
        return deadline - time.time()
    
    # Precompute all t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)
    
    # Precompute all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_subsets)
    
    # Precompute for each block, the list of t-subset indices it covers
    # Use a flat array for speed
    t_per_block = list(itertools.combinations(range(k), t))
    num_t_per_block = len(t_per_block)
    
    block_indices = np.empty((num_blocks, num_t_per_block), dtype=np.int32)
    for i, block in enumerate(all_k_subsets):
        for j, ts in enumerate(itertools.combinations(block, t)):
            block_indices[i, j] = t_subset_to_idx[ts]
    
    # For local search, we need to know which blocks cover each t-subset
    # tsub_block_sets[i] = set of block indices that cover t-subset i
    # But this is huge, so we'll build it on demand or use a different approach
    
    # Actually, for local search we need: for each t-subset, the count of blocks covering it
    # We can maintain this incrementally
    
    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Multiple randomized greedy attempts with heap
    # Use a min-heap where each entry is (negative_gain, block_idx)
    # But gain changes as we add blocks, so we need a lazy deletion approach
    
    num_attempts = 8  # Try a few attempts
    
    for attempt in range(num_attempts):
        if remaining_time() < 10.0:
            break
        
        random.seed(42 + attempt)
        
        uncovered = np.ones(num_t, dtype=bool)
        blocks = []
        # count[i] = number of blocks covering t-subset i
        cover_count = np.zeros(num_t, dtype=np.int32)
        
        while True:
            if remaining_time() < 3.0:
                break
            
            # Find the block with maximum gain
            # gain = number of currently uncovered t-subsets covered by this block
            # We can compute this efficiently
            # For each block, gain = sum of uncovered[block_indices[i]]
            
            # This is a matrix-vector product, but we want max row sum
            # Let's compute gains for all blocks
            gains = np.zeros(num_blocks, dtype=np.int32)
            for i in range(num_blocks):
                gains[i] = uncovered[block_indices[i]].sum()
            
            max_gain = gains.max()
            if max_gain == 0:
                break
            
            # Randomized selection from top blocks
            top_k = 20
            if num_blocks > top_k:
                top_idx = np.argpartition(gains, -top_k)[-top_k:]
                # Weight by gain or just random
                # Use softmax-like: pick with probability proportional to gain
                # But simpler: just random from top
                best_idx = int(np.random.choice(top_idx))
            else:
                best_idx = int(np.argmax(gains))
            
            block = list(all_k_subsets[best_idx])
            blocks.append(block)
            
            # Update uncovered and cover_count
            covered = block_indices[best_idx]
            uncovered[covered] = False
            cover_count[covered] += 1
            
            if not uncovered.any():
                break
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks.copy()
    
    # Phase 2: Local search / refinement
    # Try to remove blocks and add better ones
    
    if best_blocks is not None and remaining_time() > 15.0:
        # Convert to set of frozensets for fast lookup
        block_set = set(frozenset(b) for b in best_blocks)
        
        # Build cover count
        cover_count = np.zeros(num_t, dtype=np.int32)
        for b in best_blocks:
            for ts in itertools.combinations(b, t):
                cover_count[t_subset_to_idx[ts]] += 1
        
        improved = True
        while improved and remaining_time() > 5.0:
            improved = False
            
            # Try to remove blocks that don't cause any t-subset to become uncovered
            # i.e., blocks where all covered t-subsets have cover_count > 1
            n = len(best_blocks)
            for i in range(n):
                if remaining_time() < 5.0:
                    break
                
                b = best_blocks[i]
                can_remove = True
                for ts in itertools.combinations(b, t):
                    idx = t_subset_to_idx[ts]
                    if cover_count[idx] <= 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove this block
                    block_set.discard(frozenset(b))
                    for ts in itertools.combinations(b, t):
                        idx = t_subset_to_idx[ts]
                        cover_count[idx] -= 1
                    best_blocks.pop(i)
                    improved = True
                    break  # Restart the loop since indices changed
            
            # Try 1-for-1 swaps: remove block i, add block j
            if remaining_time() < 8.0:
                break
            
            n = len(best_blocks)
            # For each block, try to replace it with a better one
            # To keep this tractable, only try a limited number of candidates
            for i in range(min(n, 30)):  # Limit to first 30 blocks
                if remaining_time() < 5.0:
                    break
                
                b_i = best_blocks[i]
                # Find t-subsets that would become uncovered if we remove b_i
                critical_tsubs = []
                for ts in itertools.combinations(b_i, t):
                    idx = t_subset_to_idx[ts]
                    if cover_count[idx] == 1:
                        critical_tsubs.append(ts)
                
                if not critical_tsubs:
                    # Can remove without adding (already handled above, but just in case)
                    continue
                
                # Try to find a block that covers all critical_tsubs and gives net gain
                # This is expensive, so we'll just try a few random blocks
                for _ in range(50):
                    if remaining_time() < 5.0:
                        break
                    
                    j = random.randint(0, num_blocks - 1)
                    if j == i or frozenset(all_k_subsets[j]) in block_set:
                        continue
                    
                    b_j = list(all_k_subsets[j])
                    # Check if b_j covers all critical_tsubs
                    b_j_set = set(itertools.combinations(b_j, t))
                    if all(ts in b_j_set for ts in critical_tsubs):
                        # Compute gain: how many NEW t-subsets does b_j cover?
                        new_coverage = 0
                        for ts in itertools.combinations(b_j, t):
                            idx = t_subset_to_idx[ts]
                            if cover_count[idx] == 0:
                                new_coverage += 1
                        
                        # Remove b_i, add b_j
                        # Net change in size: 0 (swap)
                        # Net change in coverage: new_coverage
                        # Actually we need to check if this helps
                        # Since we're swapping, size stays same, but coverage might improve
                        # Let's just do the swap if it doesn't hurt
                        # For simplicity, only swap if new_coverage > 0
                        
                        if new_coverage > 0:
                            # Perform swap
                            block_set.discard(frozenset(b_i))
                            for ts in itertools.combinations(b_i, t):
                                cover_count[t_subset_to_idx[ts]] -= 1
                            
                            block_set.add(frozenset(b_j))
                            for ts in itertools.combinations(b_j, t):
                                cover_count[t_subset_to_idx[ts]] += 1
                            
                            best_blocks[i] = b_j
                            improved = True
                            break
        
        # Final cleanup: try removals again
        if remaining_time() > 3.0:
            n = len(best_blocks)
            for i in range(n):
                if remaining_time() < 3.0:
                    break
                b = best_blocks[i]
                can_remove = True
                for ts in itertools.combinations(b, t):
                    idx = t_subset_to_idx[ts]
                    if cover_count[idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for ts in itertools.combinations(b, t):
                        cover_count[t_subset_to_idx[ts]] -= 1
                    best_blocks.pop(i)
    
    # Fallback: if something went wrong, return a greedy solution
    if best_blocks is None:
        uncovered = np.ones(num_t, dtype=bool)
        blocks = []
        while True:
            if remaining_time() < 1.0:
                break
            gains = np.zeros(num_blocks, dtype=np.int32)
            for i in range(num_blocks):
                gains[i] = uncovered[block_indices[i]].sum()
            max_gain = gains.max()
            if max_gain == 0:
                break
            best_idx = int(np.argmax(gains))
            blocks.append(list(all_k_subsets[best_idx]))
            covered = block_indices[best_idx]
            uncovered[covered] = False
            if not uncovered.any():
                break
        best_blocks = blocks
    
    return best_blocks
