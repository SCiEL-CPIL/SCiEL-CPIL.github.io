import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    all_ksets = list(itertools.combinations(range(v), k))
    num_ksets = len(all_ksets)
    
    # Precompute bitsets for each k-set
    # bitset[i] is an integer where bit j is 1 if tset j is in kset i
    block_bitsets = [0] * num_ksets
    for i, kset in enumerate(all_ksets):
        mask = 0
        for ts in itertools.combinations(kset, t):
            idx = tset_to_idx[ts]
            mask |= (1 << idx)
        block_bitsets[i] = mask

    # Helper to count set bits
    def popcount(x):
        return bin(x).count('1')

    # Greedy Initialization
    def greedy_init(rng):
        uncovered_mask = (1 << num_tsets) - 1
        blocks = []
        coverage_count = np.zeros(num_tsets, dtype=np.int32)
        
        while uncovered_mask != 0 and time.time() < deadline - 20:
            best_block_idx = -1
            best_gain = 0
            
            # Iterate through all blocks to find the one with max gain
            # Optimization: If num_ksets is large, this is O(num_ksets * num_tsets/word_size)
            # 75k blocks * 11k bits is fast in Python with bit operations? 
            # Actually, 75k iterations is fine.
            for i in range(num_ksets):
                gain = popcount(block_bitsets[i] & uncovered_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                elif gain == best_gain and best_block_idx != -1:
                    # Tie breaking: random or prefer higher index?
                    if rng.random() < 0.5:
                        best_block_idx = i
                        best_gain = gain
            
            if best_block_idx == -1:
                break
                
            blocks.append(best_block_idx)
            # Update coverage count
            mask = block_bitsets[best_block_idx]
            # Extract set bits
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                coverage_count[idx] += 1
                m ^= lsb
            
            uncovered_mask &= ~block_bitsets[best_block_idx]
            
        return blocks, coverage_count

    # Local Search
    def local_search(blocks, coverage_count, rng):
        blocks = list(blocks)
        coverage_count = coverage_count.copy()
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Pass 1: Delete redundant blocks
            i = 0
            while i < len(blocks):
                b = blocks[i]
                mask = block_bitsets[b]
                redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if coverage_count[idx] <= 1:
                        redundant = False
                        break
                    m ^= lsb
                if redundant:
                    # Remove block
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        coverage_count[idx] -= 1
                        m ^= lsb
                    blocks.pop(i)
                    improved = True
                    # Do not increment i, check new block at position i
                else:
                    i += 1
            
            # Pass 2: Replace blocks
            # We try to replace a block with another that covers its unique t-sets
            # and maximizes redundancy gain
            for i in range(len(blocks)):
                if time.time() >= deadline - 5:
                    break
                    
                b = blocks[i]
                mask_b = block_bitsets[b]
                
                # Find unique t-sets for this block
                unique_mask = 0
                m = mask_b
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if coverage_count[idx] == 1:
                        unique_mask |= lsb
                    m ^= lsb
                
                if unique_mask == 0:
                    # Should have been deleted in Pass 1, but just in case
                    continue
                
                # Find candidate blocks that cover unique_mask
                best_candidate = -1
                best_score = 0
                
                # To speed up, we can pre-filter? No, 75k is small enough.
                for j in range(num_ksets):
                    if j == b:
                        continue
                    mask_j = block_bitsets[j]
                    # Check if mask_j covers unique_mask
                    if (unique_mask & ~mask_j) != 0:
                        continue
                    
                    # Calculate score: number of t-sets in mask_j that have coverage_count == 1
                    # These are the ones that would become 2, increasing redundancy
                    # We also want to minimize the number of t-sets in mask_j that have coverage_count == 0? 
                    # No, they are already covered by b, so they will be covered by j.
                    # The gain is in making singly-covered sets doubly-covered.
                    score = 0
                    m = mask_j
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if coverage_count[idx] == 1:
                            score += 1
                        m ^= lsb
                    
                    if score > best_score:
                        best_score = score
                        best_candidate = j
                
                if best_candidate != -1 and best_score > 0:
                    # Perform replacement
                    # Decrement counts for b
                    m = mask_b
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        coverage_count[idx] -= 1
                        m ^= lsb
                    
                    # Increment counts for best_candidate
                    m = block_bitsets[best_candidate]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        coverage_count[idx] += 1
                        m ^= lsb
                    
                    blocks[i] = best_candidate
                    improved = True
                    # Break to restart passes with new state? Or continue?
                    # Restarting is safer to catch new deletions
                    break
                
            # Perturbation: Random valid replacement
            if time.time() < deadline - 5 and len(blocks) > 1:
                if rng.random() < 0.3: # 30% chance to perturb
                    i = rng.randrange(len(blocks))
                    b = blocks[i]
                    mask_b = block_bitsets[b]
                    unique_mask = 0
                    m = mask_b
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if coverage_count[idx] == 1:
                            unique_mask |= lsb
                        m ^= lsb
                    
                    if unique_mask != 0:
                        candidates = []
                        for j in range(num_ksets):
                            if j == b: continue
                            if (unique_mask & ~block_bitsets[j]) != 0: continue
                            candidates.append(j)
                        
                        if candidates:
                            best_candidate = rng.choice(candidates)
                            # Swap
                            m = mask_b
                            while m:
                                lsb = m & -m
                                idx = lsb.bit_length() - 1
                                coverage_count[idx] -= 1
                                m ^= lsb
                            
                            m = block_bitsets[best_candidate]
                            while m:
                                lsb = m & -m
                                idx = lsb.bit_length() - 1
                                coverage_count[idx] += 1
                                m ^= lsb
                            
                            blocks[i] = best_candidate
                            improved = True

        return blocks

    rng = random.Random(42)
    
    # Initial Greedy
    blocks, coverage_count = greedy_init(rng)
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    # Iterated Local Search
    while time.time() < deadline - 10:
        # Local Search on current best
        current_blocks, current_cc = local_search(best_blocks, coverage_count, rng)
        
        if len(current_blocks) < best_count:
            best_blocks = current_blocks
            best_count = len(current_blocks)
            coverage_count = current_cc
            
        # Perturb the best solution to escape local minima
        # Pick a random block and replace it with a random valid alternative
        if len(best_blocks) > 0:
            i = rng.randrange(len(best_blocks))
            b = best_blocks[i]
            mask_b = block_bitsets[b]
            unique_mask = 0
            m = mask_b
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if coverage_count[idx] == 1:
                    unique_mask |= lsb
                m ^= lsb
            
            if unique_mask != 0:
                candidates = []
                for j in range(num_ksets):
                    if j == b: continue
                    if (unique_mask & ~block_bitsets[j]) != 0: continue
                    candidates.append(j)
                
                if candidates:
                    new_b = rng.choice(candidates)
                    # Update coverage count for this single change
                    m = mask_b
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        coverage_count[idx] -= 1
                        m ^= lsb
                    
                    m = block_bitsets[new_b]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        coverage_count[idx] += 1
                        m ^= lsb
                    
                    best_blocks[i] = new_b
            else:
                # If no unique t-sets, we can just pick a random block that covers the same t-sets?
                # Or just skip perturbation for this block
                pass

    # Convert indices back to lists of points
    final_blocks = [list(all_ksets[b]) for b in best_blocks]
    
    return final_blocks
