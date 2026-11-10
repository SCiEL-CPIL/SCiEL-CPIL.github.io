import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def now():
        return time.time()

    def remaining():
        return deadline - now()

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    nts = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute all k-blocks and their masks
    all_kblocks = list(itertools.combinations(range(v), k))
    block_masks = {}
    block_to_tsubsets = {} # frozenset(block) -> list of tsubset indices
    
    # Also map each t-subset to the list of blocks containing it
    tsubset_to_blocks = defaultdict(list)
    
    for b in all_kblocks:
        m = 0
        ts_list = []
        for ts in itertools.combinations(b, t):
            idx = tsubset_to_idx[ts]
            m |= 1 << idx
            ts_list.append(idx)
        block_masks[frozenset(b)] = m
        block_to_tsubsets[frozenset(b)] = ts_list
        for idx in ts_list:
            tsubset_to_blocks[idx].append(frozenset(b))

    full_mask = (1 << nts) - 1

    def greedy_build(seed):
        rng = random.Random(seed)
        covered = 0
        blocks = []
        
        # Randomize order of blocks to break ties
        block_order = list(all_kblocks)
        rng.shuffle(block_order)
        
        while covered != full_mask:
            if remaining() < 5:
                break
            best_block = None
            best_gain = -1
            # We can't iterate all blocks every time if it's too slow.
            # But 75k blocks * bit_count is fast in C? No, Python big ints.
            # Let's try iterating a subset or using a heuristic.
            # Heuristic: Pick a random uncovered t-subset, extend it.
            
            # Find an uncovered t-subset
            # Instead of scanning all, we can keep a list of uncovered indices?
            # Or just pick a random one and check.
            
            # Better Greedy:
            # 1. Find any uncovered t-subset S.
            # 2. Consider all blocks containing S.
            # 3. Pick the one that covers the most other uncovered t-subsets.
            
            # Find uncovered S
            # We can maintain a list of uncovered indices, but updating it is costly.
            # Let's just pick a random t-subset until we find an uncovered one.
            while True:
                if remaining() < 5:
                    return None
                ts = rng.choice(all_tsubsets)
                idx = tsubset_to_idx[ts]
                if not (covered >> idx) & 1:
                    break
            
            # Get all blocks containing this ts
            candidates = tsubset_to_blocks[idx]
            
            best_gain = -1
            best_b = None
            for b in candidates:
                m = block_masks[b]
                # Count how many bits in m are NOT in covered
                # new_bits = m & ~covered
                # gain = new_bits.bit_count()
                # This is fast enough?
                # 364 candidates * bit_count.
                gain = (m & ~covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_b = b
            
            if best_b:
                blocks.append(list(best_b))
                covered |= block_masks[best_b]
                
        if covered == full_mask:
            return blocks
        return None

    def remove_redundant(blocks):
        # Calculate coverage counts
        counts = np.zeros(nts, dtype=np.int32)
        for b in blocks:
            for idx in block_to_tsubsets[frozenset(b)]:
                counts[idx] += 1
        
        new_blocks = []
        for b in blocks:
            is_critical = False
            for idx in block_to_tsubsets[frozenset(b)]:
                if counts[idx] == 1:
                    is_critical = True
                    break
            if not is_critical:
                # Remove it
                for idx in block_to_tsubsets[frozenset(b)]:
                    counts[idx] -= 1
            else:
                new_blocks.append(b)
        
        return new_blocks

    def local_search(blocks, max_time):
        start_ls = now()
        blocks = remove_redundant(blocks)
        
        improved = True
        while improved and (now() - start_ls) < max_time:
            improved = False
            n_blocks = len(blocks)
            # Shuffle blocks to avoid bias
            random.shuffle(blocks)
            
            for i in range(n_blocks):
                if (now() - start_ls) > max_time:
                    break
                if i >= len(blocks):
                    continue
                
                b_i = blocks[i]
                # Remove b_i
                blocks.pop(i)
                
                # Calculate uncovered
                counts = np.zeros(nts, dtype=np.int32)
                for b in blocks:
                    for idx in block_to_tsubsets[frozenset(b)]:
                        counts[idx] += 1
                
                uncovered = [idx for idx in range(nts) if counts[idx] == 0]
                
                if not uncovered:
                    # We can remove b_i entirely!
                    improved = True
                    continue
                else:
                    # We need to find a replacement
                    # Find blocks that cover all uncovered subsets
                    # Intersection of tsubset_to_blocks for all idx in uncovered
                    
                    # Optimization: If len(uncovered) is large, skip
                    if len(uncovered) > 3:
                        blocks.insert(i, b_i)
                        continue
                    
                    # Find candidate replacement
                    # Start with blocks containing the first uncovered subset
                    candidates = set(tsubset_to_blocks[uncovered[0]])
                    for u_idx in uncovered[1:]:
                        candidates &= set(tsubset_to_blocks[u_idx])
                        if not candidates:
                            break
                    
                    if not candidates:
                        blocks.insert(i, b_i)
                        continue
                    
                    # Try each candidate
                    best_new_block = None
                    best_size = len(blocks) + 1 # Current size + 1 (since we popped one, we need to add one to maintain coverage)
                    # Actually, we want to minimize total size.
                    # If we add one, size is len(blocks) + 1.
                    # But maybe adding this one allows removing others?
                    # Let's just try to add one and see if redundancy removal helps.
                    
                    for c in candidates:
                        # Add c
                        test_blocks = blocks + [list(c)]
                        # Remove redundant
                        test_blocks = remove_redundant(test_blocks)
                        if len(test_blocks) < len(blocks) + 1:
                            # We improved or maintained?
                            # If len(test_blocks) < len(blocks) + 1, it means we removed at least 1 redundant block.
                            # But we started with len(blocks) (after pop).
                            # So if len(test_blocks) <= len(blocks), we didn't increase size.
                            # If len(test_blocks) < len(blocks), we decreased size.
                            
                            # We want to accept if len(test_blocks) < len(blocks) + 1 is not the goal,
                            # we want len(test_blocks) < original_size.
                            # original_size was len(blocks) + 1 (before pop).
                            # So if len(test_blocks) < len(blocks) + 1, it's an improvement or tie.
                            # Let's accept if strictly better or equal (to explore).
                            if len(test_blocks) < len(blocks) + 1:
                                best_new_block = test_blocks
                                best_size = len(test_blocks)
                                break # Take first improvement? Or best?
                                # Taking first is faster.
                    
                    if best_new_block is not None:
                        blocks = best_new_block
                        improved = True
                    else:
                        # Restore b_i
                        blocks.insert(i, b_i)
        
        return blocks

    best_result = None
    best_len = float('inf')
    
    # Phase 1: Greedy + Local Search
    n_restarts = 20
    for seed in range(n_restarts):
        if remaining() < 10:
            break
        res = greedy_build(seed)
        if res is None:
            continue
        # Local search with time budget
        budget = min(5, max(0.5, remaining() - 10))
        res = local_search(res, budget)
        if len(res) < best_len:
            best_len = len(res)
            best_result = res
    
    # Phase 2: Polish best result
    if best_result is not None and remaining() > 5:
        budget = max(0.5, remaining() - 5)
        refined = local_search(list(best_result), budget)
        if len(refined) < best_len:
            best_len = len(refined)
            best_result = refined
            
    return best_result
