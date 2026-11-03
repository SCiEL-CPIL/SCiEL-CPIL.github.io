import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Use bitmask for t-subsets for faster lookup
    mask_to_idx = {}
    idx_to_mask = [0] * n_t_subsets
    for i, s in enumerate(all_t_subsets):
        mask = 0
        for p in s:
            mask |= (1 << p)
        mask_to_idx[mask] = i
        idx_to_mask[i] = mask

    # Precompute masks for all (t-1) subsets? No, block changes.
    # But we can precompute masks for all subsets of size < k?
    # Or just compute on the fly.
    
    def get_mask(subset):
        mask = 0
        for p in subset:
            mask |= (1 << p)
        return mask

    def greedy_cover(seed_val):
        random.seed(seed_val)
        # Use a set of uncovered indices for O(1) check?
        # Or a boolean array?
        # A set of indices is good for "is this uncovered?".
        # But we need to count gains.
        # Let's use a numpy array of bools for uncovered?
        # Or just a set of indices.
        uncovered_set = set(range(n_t_subsets))
        blocks = []
        
        while uncovered_set:
            if time.time() > deadline:
                return None # Signal timeout/failure
            
            # Find first uncovered subset to start block
            # Randomly pick an uncovered subset to diversify?
            # Or pick the one with fewest possible completions?
            # Let's pick a random uncovered subset.
            first_uncovered_idx = random.choice(list(uncovered_set))
            start_subset = all_t_subsets[first_uncovered_idx]
            start_point = random.choice(start_subset)
            block = [start_point]
            
            while len(block) < k:
                best_gain = -1
                best_point = -1
                candidates = [x for x in range(v) if x not in block]
                
                # Precompute masks for (t-1) subsets of current block?
                # Block size is small.
                block_set = set(block)
                
                for x in candidates:
                    gain = 0
                    # Iterate over (t-1) subsets of block
                    # We need to form t-subsets containing x.
                    # So we pick (t-1) points from block.
                    for rest in itertools.combinations(block, t - 1):
                        # Form the t-subset
                        # Since x is not in block, and rest is from block, 
                        # the set is rest U {x}.
                        # We need the mask.
                        # We can compute mask of rest, then OR with (1<<x).
                        # To speed up, precompute masks of (t-1) subsets of block?
                        # Let's just compute it.
                        mask_rest = 0
                        for p in rest:
                            mask_rest |= (1 << p)
                        mask_full = mask_rest | (1 << x)
                        
                        if mask_full in mask_to_idx:
                            idx = mask_to_idx[mask_full]
                            if idx in uncovered_set:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point == -1:
                    # Should not happen if uncovered_set is not empty and k < v
                    break
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered_set
            # Get all t-subsets of this block
            for s in itertools.combinations(block, t):
                mask = 0
                for p in s:
                    mask |= (1 << p)
                idx = mask_to_idx[mask]
                if idx in uncovered_set:
                    uncovered_set.remove(idx)
                
        return blocks

    def remove_redundant(blocks):
        if not blocks:
            return blocks
            
        # Count coverage
        counts = [0] * n_t_subsets
        for block in blocks:
            for s in itertools.combinations(block, t):
                mask = 0
                for p in s:
                    mask |= (1 << p)
                idx = mask_to_idx[mask]
                counts[idx] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline:
                    return blocks
                block = blocks[i]
                is_redundant = True
                for s in itertools.combinations(block, t):
                    mask = 0
                    for p in s:
                        mask |= (1 << p)
                    idx = mask_to_idx[mask]
                    if counts[idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for s in itertools.combinations(block, t):
                        mask = 0
                        for p in s:
                            mask |= (1 << p)
                        idx = mask_to_idx[mask]
                        counts[idx] -= 1
                    blocks.pop(i)
                    changed = True
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple restarts
    for seed in range(20):
        if time.time() > deadline:
            break
        
        blocks = greedy_cover(seed)
        if blocks is None:
            continue
            
        # Verify validity
        # Check if all subsets are covered
        covered = [False] * n_t_subsets
        for block in blocks:
            for s in itertools.combinations(block, t):
                mask = 0
                for p in s:
                    mask |= (1 << p)
                idx = mask_to_idx[mask]
                covered[idx] = True
        
        if all(covered):
            blocks = remove_redundant(blocks)
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
        else:
            # Invalid cover, discard
            pass

    if best_blocks is None:
        # Fallback: Return a valid cover, even if large.
        # Simple greedy without randomization might be faster?
        # Or just return all blocks? No.
        # Let's try one more time with seed 0, but force it to finish?
        # If it timed out, it's likely stuck.
        # Let's use a deterministic greedy that is very fast.
        # Actually, if the above failed, something is wrong.
        # Let's return a cover constructed by taking all k-subsets containing 0?
        # No, that misses subsets not containing 0.
        # 
        # A guaranteed valid cover: The set of ALL k-subsets.
        # This is large (75582) but valid.
        # We can try to reduce it with remove_redundant.
        # But generating 75k blocks and running remove_redundant might be slow.
        # 
        # Better fallback: 
        # Use a simple greedy that picks blocks in a fixed order?
        # Or just raise an error? No, must return.
        # 
        # Let's try a simple greedy one more time, but with a tighter loop?
        # Actually, if the previous attempts failed, it's likely the timeout.
        # Let's just return the result of a single fast greedy run.
        blocks = greedy_cover(0)
        if blocks is None:
             # If even this fails, return a trivial valid cover:
             # All blocks.
             return [list(b) for b in itertools.combinations(range(v), k)]
        # Verify
        covered = [False] * n_t_subsets
        for block in blocks:
            for s in itertools.combinations(block, t):
                mask = 0
                for p in s:
                    mask |= (1 << p)
                idx = mask_to_idx[mask]
                covered[idx] = True
        if all(covered):
            return remove_redundant(blocks)
        else:
            return [list(b) for b in itertools.combinations(range(v), k)]

    return best_blocks
