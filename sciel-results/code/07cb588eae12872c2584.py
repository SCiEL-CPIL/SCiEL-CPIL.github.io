"""
Strategy: Vectorized Greedy Set Cover with Bitmask Acceleration
+ Iterative Local Search (remove redundant blocks, swap for better coverage)

Key improvements over the previous approach:
1. Precompute all triples as a bitset (2024 bits) using numpy uint64 array.
2. Each candidate block maps to a bitset of triples it covers.
3. Greedy set cover uses vectorized operations to find the block covering the most uncovered triples.
4. Local search: try removing any block (if its triples are all covered by others),
   and try swapping a block for a different one that covers its critical triples plus more.
"""

import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Precompute all t-subsets and assign each an index
    all_triples = list(itertools.combinations(range(v), t))
    triple_count = len(all_triples)
    triple_index = {tr: i for i, tr in enumerate(all_triples)}
    
    # Number of 64-bit words needed to represent triple_count bits
    num_words = (triple_count + 63) // 64
    triple_mask_size = num_words
    
    def block_to_triple_mask(block):
        """Convert a block (list of k points) to a bitmask of triples it covers."""
        mask = np.zeros(num_words, dtype=np.uint64)
        for tr in itertools.combinations(block, t):
            idx = triple_index[tr]
            word_idx = idx // 64
            bit_idx = idx % 64
            mask[word_idx] |= (1 << bit_idx)
        return mask
    
    # --- Phase 1: Generate initial valid covering using greedy ---
    # We'll build a pool of candidate blocks first.
    
    # Deterministic greedy to get a baseline valid covering
    uncovered_set = set(all_triples)
    baseline_blocks = []
    
    while uncovered_set:
        first_triple = next(iter(uncovered_set))
        block = list(first_triple)
        
        while len(block) < k:
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for subset in itertools.combinations(block, t - 1):
                    new_triple = tuple(sorted(subset + (x,)))
                    if new_triple in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point is None:
                break
            block.append(best_point)
        
        block.sort()
        baseline_blocks.append(tuple(block))
        for tr in itertools.combinations(block, t):
            uncovered_set.discard(tr)
    
    # --- Phase 2: Expand pool with random candidates ---
    rng = random.Random(42)
    pool = list(set(baseline_blocks))
    
    # Target pool size: enough for good search, not too large for time
    target_pool_size = min(500, max(len(pool) + 100, 300))
    
    while len(pool) < target_pool_size:
        # Random block construction: start with a random triple, fill greedily-ish
        r_triple = tuple(sorted(rng.sample(range(v), t)))
        block = list(r_triple)
        while len(block) < k:
            remaining = [x for x in range(v) if x not in block]
            if not remaining:
                break
            block.append(rng.choice(remaining))
        block.sort()
        b = tuple(block)
        if b not in pool:
            pool.append(b)
    
    # Precompute triple masks for all pool blocks
    pool_masks = [block_to_triple_mask(b) for b in pool]
    
    # --- Phase 3: Greedy Set Cover ---
    full_universe_mask = np.ones(num_words, dtype=np.uint64)
    
    def popcount_mask(mask):
        """Count bits in a numpy uint64 array."""
        total = 0
        for word in mask:
            total += bin(word).count('1')
        return total
    
    def mask_sub(a, b):
        """Bitwise AND NOT: a & ~b"""
        return a & (~b) & np.full_like(a, (1 << 64) - 1)
    
    def mask_and(a, b):
        return a & b
    
    def mask_or(a, b):
        return a | b
    
    def masks_equal(a, b):
        return np.array_equal(a, b)
    
    # Greedy set cover
    covered_mask = np.zeros(num_words, dtype=np.uint64)
    selected_indices = []
    available = list(range(len(pool)))
    
    while not masks_equal(covered_mask, full_universe_mask):
        if time.time() > deadline - 20:
            break
        
        best_idx = -1
        best_gain = -1
        
        for idx in available:
            gain = popcount_mask(mask_and(pool_masks[idx], mask_sub(full_universe_mask, covered_mask)))
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        
        if best_idx == -1 or best_gain == 0:
            break
        
        selected_indices.append(best_idx)
        covered_mask = mask_or(covered_mask, pool_masks[best_idx])
        available.remove(best_idx)
    
    # Verify coverage
    if not masks_equal(covered_mask, full_universe_mask):
        # Fallback: return baseline blocks directly
        return [list(b) for b in baseline_blocks]
    
    best_size = len(selected_indices)
    best_selected = selected_indices[:]
    
    # --- Phase 4: Local Search ---
    # Build coverage counts per triple
    # coverage_count[triple_idx] = number of selected blocks covering that triple
    coverage_count = np.zeros(triple_count, dtype=np.int32)
    
    for idx in selected_indices:
        mask = pool_masks[idx]
        for word_i in range(num_words):
            word = mask[word_i]
            if word == 0:
                continue
            base = word_i * 64
            while word:
                bit = (word & -word).bit_length() - 1
                triple_idx = base + bit
                if triple_idx < triple_count:
                    coverage_count[triple_idx] += 1
                word &= word - 1
    
    def recompute_coverage(selected_list):
        """Recompute coverage counts from scratch."""
        cc = np.zeros(triple_count, dtype=np.int32)
        for idx in selected_list:
            mask = pool_masks[idx]
            for word_i in range(num_words):
                word = mask[word_i]
                if word == 0:
                    continue
                base = word_i * 64
                while word:
                    bit = (word & -word).bit_length() - 1
                    triple_idx = base + bit
                    if triple_idx < triple_count:
                        cc[triple_idx] += 1
                    word &= word - 1
        return cc
    
    def is_valid_cover(selected_list):
        """Check if selected_list covers all triples."""
        mask = np.zeros(num_words, dtype=np.uint64)
        for idx in selected_list:
            mask = mask_or(mask, pool_masks[idx])
        return masks_equal(mask, full_universe_mask)
    
    # Local search: try to remove redundant blocks
    for _iteration in range(20):  # Multiple passes
        if time.time() > deadline - 10:
            break
        
        progress = False
        new_coverage = coverage_count.copy()
        
        for i in range(len(selected_indices)):
            idx = selected_indices[i]
            mask = pool_masks[idx]
            
            # Check if any triple in this block is only covered by this block
            has_critical = False
            for word_i in range(num_words):
                word = mask[word_i]
                if word == 0:
                    continue
                base = word_i * 64
                w = word
                while w:
                    bit = (w & -w).bit_length() - 1
                    triple_idx = base + bit
                    if triple_idx < triple_count and coverage_count[triple_idx] <= 1:
                        has_critical = True
                        break
                    w &= w - 1
                if has_critical:
                    break
            
            if not has_critical:
                # Remove this block
                selected_indices.pop(i)
                for word_i in range(num_words):
                    word = mask[word_i]
                    if word == 0:
                        continue
                    base = word_i * 64
                    w = word
                    while w:
                        bit = (w & -w).bit_length() - 1
                        triple_idx = base + bit
                        if triple_idx < triple_count:
                            coverage_count[triple_idx] -= 1
                        w &= w - 1
                progress = True
                if len(selected_indices) < best_size:
                    best_size = len(selected_indices)
                    best_selected = selected_indices[:]
                break  # Restart the pass
        
        if not progress:
            break
    
    # --- Phase 5: Try swaps to reduce size ---
    # Try replacing a block with another that covers all its critical triples
    for _iteration in range(10):
        if time.time() > deadline - 5:
            break
        
        improved = False
        new_coverage = coverage_count.copy()
        
        for i in range(len(selected_indices)):
            idx = selected_indices[i]
            mask_i = pool_masks[idx]
            
            # Find critical triples in block i
            critical_triples = []
            for word_i in range(num_words):
                word = mask_i[word_i]
                if word == 0:
                    continue
                base = word_i * 64
                w = word
                while w:
                    bit = (w & -w).bit_length() - 1
                    triple_idx = base + bit
                    if triple_idx < triple_count and coverage_count[triple_idx] <= 1:
                        critical_triples.append(triple_idx)
                    w &= w - 1
            
            if not critical_triples:
                # Block is redundant, remove it
                selected_indices.pop(i)
                for ti in critical_triples:  # empty
                    pass
                # Decrement coverage for all triples in mask_i
                for word_i in range(num_words):
                    word = mask_i[word_i]
                    if word == 0:
                        continue
                    base = word_i * 64
                    w = word
                    while w:
                        bit = (w & -w).bit_length() - 1
                        triple_idx = base + bit
                        if triple_idx < triple_count:
                            coverage_count[triple_idx] -= 1
                        w &= w - 1
                if len(selected_indices) < best_size:
                    best_size = len(selected_indices)
                    best_selected = selected_indices[:]
                improved = True
                break
            
            # Try to find a replacement block that covers all critical triples
            found = False
            for j in range(len(pool)):
                if j == idx:
                    continue
                if j in selected_indices:
                    continue
                
                # Check if pool[j] covers all critical triples
                mask_j = pool_masks[j]
                covers_all = True
                for ti in critical_triples:
                    word_idx = ti // 64
                    bit_idx = ti % 64
                    if not (mask_j[word_idx] & (1 << bit_idx)):
                        covers_all = False
                        break
                
                if covers_all:
                    # Perform swap: remove idx, add j
                    selected_indices[i] = j
                    # Update coverage: subtract mask_i, add mask_j
                    for word_i in range(num_words):
                        w = mask_i[word_i]
                        if w:
                            base = word_i * 64
                            ww = w
                            while ww:
                                bit = (ww & -ww).bit_length() - 1
                                ti = base + bit
                                if ti < triple_count:
                                    coverage_count[ti] -= 1
                                ww &= ww - 1
                        w = mask_j[word_i]
                        if w:
                            base = word_i * 64
                            ww = w
                            while ww:
                                bit = (ww & -ww).bit_length() - 1
                                ti = base + bit
                                if ti < triple_count:
                                    coverage_count[ti] += 1
                                ww &= ww - 1
                    found = True
                    improved = True
                    break
            
            if found:
                break
        
        if not improved:
            break
    
    # Final validation and return
    final_blocks = [list(pool[i]) for i in selected_indices]
    
    # Verify
    covered_mask = np.zeros(num_words, dtype=np.uint64)
    for b in final_blocks:
        covered_mask = mask_or(covered_mask, block_to_triple_mask(b))
    
    if not masks_equal(covered_mask, full_universe_mask):
        # Fallback to baseline
        return [list(b) for b in baseline_blocks]
    
    return final_blocks
