import itertools
import random
import time
import numpy as np

_START_TIME = None

def _deadline_ok(budget):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return (time.time() - _START_TIME) < budget

def construct(v, k, t):
    global _START_TIME
    _START_TIME = time.time()
    budget = 145.0  # safety margin before 150s kill

    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    n_triples = len(all_triples)
    triple_index = {tr: i for i, tr in enumerate(all_triples)}

    # Precompute bitmask for each triple
    # triple_mask[i] is a Python int with bit i set
    triple_mask = [1 << i for i in range(n_triples)]
    
    # Precompute mask for each block? 
    # There are C(24,7) = 346104 blocks.
    # We can precompute the mask for every possible block to speed up local search swaps.
    # Memory: 346104 * (size of int). Python ints for 2024 bits are ~ 28 + 2024/8 * 4 bytes? 
    # Actually, Python int overhead is small. Let's just compute on the fly or cache.
    # Computing on the fly: sum of 35 triple masks. 35 additions of big ints. Fast.
    
    def get_block_mask(block):
        m = 0
        for tr in itertools.combinations(block, t):
            idx = triple_index[tr]
            m |= triple_mask[idx]
        return m

    def popcount(x):
        return bin(x).count("1")

    def greedy_cover(seed):
        r = random.Random(seed)
        covered_mask = 0
        blocks = []
        block_masks = []
        
        # We need to track which triples are covered to pick "uncovered" ones quickly?
        # With bitmask, checking if a triple is covered is `covered_mask & triple_mask[idx]`.
        # To pick a random uncovered triple, we can just scan or use a list of uncovered indices.
        # Since n_triples is small (2024), scanning is fine.
        
        # Heuristic: 
        # 1. Find an uncovered triple.
        # 2. Build a block around it to maximize gain.
        
        while True:
            if not _deadline_ok(budget):
                break
                
            # Find uncovered triple
            uncovered_triples = [tr for tr in all_triples if not (covered_mask & triple_mask[triple_index[tr]])]
            if not uncovered_triples:
                break
                
            # Pick random uncovered triple
            start_triple = r.choice(uncovered_triples)
            block = list(start_triple)
            block_set = set(block)
            
            # Greedily add points to maximize new coverage
            current_block_mask = get_block_mask(block)
            
            while len(block) < k:
                best_x = None
                best_gain = -1
                best_new_mask = 0
                
                # Pairs in current block
                block_pairs = list(itertools.combinations(block, t - 1))
                
                for x in points:
                    if x in block_set:
                        continue
                    # Calculate gain of adding x
                    # New triples are (p1, p2, x) for p1, p2 in block
                    gain_mask = 0
                    for pair in block_pairs:
                        tr = (pair[0], pair[1], x)
                        idx = triple_index[tr]
                        gain_mask |= triple_mask[idx]
                    
                    # Gain is number of new triples
                    gain = popcount(gain_mask & ~covered_mask)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_new_mask = gain_mask
                    elif gain == best_gain and best_x is not None:
                        # Tie break: random
                        if r.random() < 0.5:
                            best_x = x
                            best_new_mask = gain_mask
                            
                if best_x is None:
                    break
                block.append(best_x)
                block_set.add(best_x)
                current_block_mask |= best_new_mask

            block.sort()
            
            # Check if this block adds any new coverage
            if popcount(current_block_mask & ~covered_mask) == 0:
                # If it adds nothing, maybe we are stuck? 
                # But we started with an uncovered triple, so it should add at least 1.
                # Unless the "uncovered" check was stale? No, we picked from uncovered.
                # So it must add at least the start_triple.
                pass

            blocks.append(block)
            block_masks.append(current_block_mask)
            covered_mask |= current_block_mask

        return blocks, covered_mask

    def local_search(blocks, covered_mask, max_time=5.0):
        if not _deadline_ok(budget):
            return blocks, covered_mask

        ls_start = time.time()
        current_blocks = [list(b) for b in blocks]
        current_masks = [get_block_mask(b) for b in current_blocks]
        current_covered_mask = covered_mask
        
        # Helper to check if a block is removable
        # A block is removable if all its triples are covered by other blocks.
        # i.e., (current_covered_mask ^ current_masks[i]) & current_masks[i] == 0 ?
        # No. If we remove block i, the new covered mask is current_covered_mask & ~current_masks[i] ?
        # No, that's not right. Other blocks might cover those triples.
        # We need to know if any triple in block i is covered ONLY by block i.
        # This requires counting coverage per triple.
        # Let's maintain a count array for local search.
        
        # Re-initialize counts
        counts = np.zeros(n_triples, dtype=np.int32)
        for mask in current_masks:
            # This is slow to iterate bits. 
            # Alternative: just use the mask to check removal?
            # If we remove block i, we lose coverage for triples in block i that are NOT in any other block.
            # Let's stick to the count array for correctness and speed in removal check.
            pass
            
        # Let's rebuild counts from blocks
        for b in current_blocks:
            for tr in itertools.combinations(b, t):
                counts[triple_index[tr]] += 1

        def is_removable(block_idx):
            b = current_blocks[block_idx]
            for tr in itertools.combinations(b, t):
                if counts[triple_index[tr]] == 1:
                    return False
            return True

        def try_remove(block_idx):
            nonlocal current_blocks, current_masks, current_covered_mask
            if not is_removable(block_idx):
                return False
            b = current_blocks[block_idx]
            for tr in itertools.combinations(b, t):
                counts[triple_index[tr]] -= 1
            current_blocks.pop(block_idx)
            current_masks.pop(block_idx)
            # Recalculate covered mask? 
            # We can just OR all masks.
            # Or keep it updated. Removing a mask is hard with OR.
            # Let's just recalculate covered_mask when needed or keep a count of unique triples?
            # Actually, for the "swap" logic, we need to know what is uniquely covered.
            # The `counts` array is the source of truth for "uniquely covered".
            # We don't strictly need `current_covered_mask` if we have `counts`.
            # But `counts` update is O(35) per block.
            return True

        # Main loop
        while _deadline_ok(budget) and (time.time() - ls_start) < max_time:
            # Phase 1: Remove redundant blocks
            removed_any = False
            for i in range(len(current_blocks) - 1, -1, -1):
                if is_removable(i):
                    try_remove(i)
                    removed_any = True
            
            # Phase 2: Swaps
            # Try to replace a block with a better one.
            # A good swap: Replace block B with B' such that B' covers all triples that B uniquely covers.
            # And B' might cover additional triples.
            # We can try to find B' by sampling.
            
            # Sample random blocks
            # To speed up, we can precompute a list of "good" blocks?
            # Or just generate random blocks and check.
            
            for _ in range(500): # Limit swaps per pass
                if not _deadline_ok(budget) or (time.time() - ls_start) > max_time:
                    break
                if not current_blocks:
                    break
                
                b_idx = random.randrange(len(current_blocks))
                b = current_blocks[b_idx]
                
                # Find triples uniquely covered by b
                unique_triples = []
                for tr in itertools.combinations(b, t):
                    if counts[triple_index[tr]] == 1:
                        unique_triples.append(tr)
                
                if not unique_triples:
                    # This block is redundant? 
                    # If it has no unique triples, it should have been removed in Phase 1.
                    # Unless Phase 1 didn't run or failed? 
                    # If is_removable is true, it is removed.
                    # So if we are here, it has unique triples.
                    continue

                # Try to find a replacement block B' that covers all unique_triples
                # And ideally covers more.
                found_replacement = False
                for _attempt in range(20):
                    # Generate a random block
                    # To make it more likely to cover unique_triples, we can start with one of them
                    # and fill the rest randomly.
                    start_tr = random.choice(unique_triples)
                    new_block = list(start_tr)
                    new_block_set = set(new_block)
                    
                    while len(new_block) < k:
                        x = random.choice(points)
                        if x not in new_block_set:
                            new_block.append(x)
                            new_block_set.add(x)
                    
                    new_block.sort()
                    
                    # Check if new_block covers all unique_triples
                    new_block_triples = set(itertools.combinations(new_block, t))
                    if all(tr in new_block_triples for tr in unique_triples):
                        # Valid replacement candidate
                        # Check if it's better (covers more total triples, or just valid)
                        # For now, accept if valid. 
                        # We can also check if it covers more non-unique triples.
                        
                        # Apply swap
                        # 1. Remove old block
                        for tr in itertools.combinations(b, t):
                            counts[triple_index[tr]] -= 1
                        
                        # 2. Add new block
                        for tr in new_block_triples:
                            counts[triple_index[tr]] += 1
                        
                        # Check if swap maintained validity (it should, by construction of unique_triples)
                        # But what if new_block covers a triple that was ONLY covered by the old block?
                        # We ensured new_block covers ALL unique triples of the old block.
                        # So those are still covered.
                        # Triples not in old block but in new block: counts increase.
                        # Triples in old block but not new block: counts decrease.
                        #   If count was 1, it becomes 0 -> INVALID.
                        #   Wait, we only checked unique_triples of the OLD block.
                        #   If a triple T is in OLD but not in NEW, and count[T] was 1, then T is unique to OLD.
                        #   So T is in unique_triples.
                        #   We required NEW to cover all unique_triples.
                        #   So NEW covers T.
                        #   Contradiction: T is not in NEW.
                        #   So if T is in unique_triples, it MUST be in NEW.
                        #   Therefore, if NEW covers all unique_triples, it covers all triples in OLD that had count 1.
                        #   What about triples in OLD with count > 1?
                        #   If T is in OLD, count[T] > 1. T is not in unique_triples.
                        #   If T is not in NEW, count[T] decreases by 1.
                        #   Since count[T] > 1, new count[T] >= 1. Valid.
                        #   So the swap is valid.
                        
                        current_blocks[b_idx] = new_block
                        current_masks[b_idx] = get_block_mask(new_block)
                        found_replacement = True
                        break
                        
                # If we found a replacement, we might have made other blocks removable.
                # Loop continues.

        return current_blocks, current_covered_mask # Return covered_mask even if not updated perfectly, just for signature

    # Main
    best_blocks = None
    best_covered_mask = None
    
    for restart in range(100): # More restarts possible with bitmask speed?
        if not _deadline_ok(budget):
            break
            
        blocks, covered_mask = greedy_cover(restart)
        
        # Local search
        lb_blocks, lb_covered_mask = local_search(blocks, covered_mask, max_time=2.0)
        
        if best_blocks is None or len(lb_blocks) < len(best_blocks):
            best_blocks = lb_blocks
            best_covered_mask = lb_covered_mask
            # If we found a very good one, spend more time
            if len(best_blocks) <= 76:
                 lb_blocks2, _ = local_search(best_blocks, best_covered_mask, max_time=5.0)
                 if len(lb_blocks2) < len(best_blocks):
                     best_blocks = lb_blocks2
                     best_covered_mask = lb_covered_mask # Approximation

    if best_blocks is None:
        blocks, covered_mask = greedy_cover(0)
        best_blocks = blocks

    return best_blocks
