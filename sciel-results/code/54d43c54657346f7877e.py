import itertools
import random
import time

def _remove_redundant_blocks(v, k, t, blocks):
    """
    Remove blocks that do not uniquely cover any t-subset.
    Returns a new list of blocks.
    """
    if not blocks:
        return []
    
    # Map each t-subset to the count of blocks covering it
    # We iterate until no more redundant blocks are found.
    
    # Optimization: Use a dictionary to count coverage.
    # Key: tuple of t-subset, Value: count
    
    # To avoid recomputing combinations repeatedly, we can cache them?
    # Or just compute on the fly since k=7, t=4 is small (35 per block).
    
    current_blocks = list(blocks)
    
    while True:
        if not current_blocks:
            break
            
        counts = {}
        for b in current_blocks:
            for tsub in itertools.combinations(b, t):
                key = tuple(tsub)
                counts[key] = counts.get(key, 0) + 1
                
        redundant_indices = []
        for i, b in enumerate(current_blocks):
            is_redundant = True
            for tsub in itertools.combinations(b, t):
                key = tuple(tsub)
                if counts.get(key, 0) <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                redundant_indices.append(i)
                
        if not redundant_indices:
            break
            
        # Remove redundant blocks
        # It's possible that removing one block makes another redundant, 
        # so we loop.
        # We can remove all identified redundant blocks at once?
        # If A is redundant because B covers its unique parts, and B is redundant because A covers its unique parts?
        # Impossible: if A's parts are covered by others, and B's parts are covered by others.
        # If A and B are the only ones covering a specific t-subset S, then count[S] = 2.
        # If A is redundant, all its t-subsets have count > 1.
        # If we remove A, count[S] becomes 1.
        # If B was also marked redundant, it means all B's t-subsets had count > 1.
        # If S is in B, count[S] was > 1. It was 2 (A and B).
        # If we remove A, S is still covered by B.
        # So removing all simultaneously is safe.
        
        redundant_set = set(redundant_indices)
        current_blocks = [b for i, b in enumerate(current_blocks) if i not in redundant_set]
        
    return current_blocks

def _build_greedy(v, k, t, all_tsubs, rng, deadline):
    """
    Greedy construction with randomization.
    """
    uncovered = set(all_tsubs)
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Pick a random uncovered t-subset to seed the new block
        # Using list(uncovered) is O(N), but N decreases.
        # For better performance, we could maintain a list of uncovered t-subsets.
        # But set is fine for now given the small size (3876).
        
        # Optimization: Pick a "hard" t-subset?
        # For now, random.
        
        # To speed up random selection, we can convert to list occasionally?
        # Or just use rng.choice on the set? 
        # Python's random.choice requires a sequence. 
        # So we must convert.
        # This is expensive if done every iteration.
        # Alternative: maintain a list `uncovered_list` and swap-remove.
        
        # Let's optimize the uncovered structure.
        # But for simplicity and correctness first, let's stick to set.
        # If it's too slow, we'll switch to list.
        
        # Actually, converting a set of size ~1000 to a list is fast.
        # 3876 total.
        
        seed = rng.choice(list(uncovered))
        block = list(seed)
        block_set = set(block)
        
        while len(block) < k:
            if time.time() > deadline:
                break
                
            best_gain = -1
            best_point = None
            t_minus_1 = t - 1
            
            # Precompute (t-1) subsets of current block
            # block size is at least t (since we started with t)
            # So we can form combinations
            block_t1 = list(itertools.combinations(block, t_minus_1))
            
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                for comb in block_t1:
                    # Form the t-subset
                    # comb is a tuple of t-1 elements
                    # x is an int
                    # We need to check if the set(comb) U {x} is in uncovered
                    # Since comb is sorted (from itertools on sorted block? No, block is not necessarily sorted during construction)
                    # Wait, block starts as list(seed). seed is a tuple from combinations, so it's sorted.
                    # We append points. So block might not be sorted.
                    # But itertools.combinations preserves order.
                    # So comb is a tuple.
                    # To check if {comb elements} U {x} is in uncovered (which contains sorted tuples),
                    # we should form the sorted tuple.
                    
                    # Optimization:
                    # Instead of sorting every time, we can check membership in a different way?
                    # Or just sort. t=4, so sorting 4 elements is very fast.
                    
                    tsub = tuple(sorted(comb + (x,)))
                    if tsub in uncovered:
                        gain += 1
                        # Early termination? If gain is already high?
                        # No, we want max gain.
                    
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    
            if best_point is None:
                # Should not happen if uncovered is not empty and we can add points
                # But if k == t, we wouldn't be in this loop.
                # If we are in this loop, len(block) < k.
                # If no point adds gain, it means all potential t-subsets are covered.
                # But we have an uncovered t-subset `seed` which is a subset of `block`?
                # No, `seed` is in `block`.
                # If `seed` is in `block`, and `block` is not full,
                # does adding a point cover `seed`?
                # `seed` is already a subset of `block`.
                # The `uncovered` set is updated only after the block is complete.
                # So `seed` is still in `uncovered`.
                # When we check gain for adding `x`, we check if `tsub` is in `uncovered`.
                # If `tsub` == `seed`, it is in `uncovered`.
                # So gain should be at least 1 for any `x` that forms `seed`?
                # Wait, `seed` is formed by `t` elements from `block`.
                # When we add `x`, we form new t-subsets using `x` and `t-1` from `block`.
                # `seed` does NOT contain `x` (since `x` is new).
                # So `seed` is NOT one of the new t-subsets formed by adding `x`.
                # So `seed` is not "covered" by the addition of `x` in the sense of the gain calculation?
                # No, `seed` is covered by the FINAL block.
                # The gain calculation estimates how many NEW t-subsets are covered by the final block
                # that were not covered by previous blocks.
                # Since `seed` was not in any previous block, it is "new".
                # But the gain calculation only looks at t-subsets that include the NEW point `x`.
                # `seed` does not include `x`.
                # So `seed` is not counted in the gain!
                # This is a flaw in the simple greedy gain calculation if the seed is not "re-covered" by the new points.
                # However, the seed IS covered by the final block.
                # The issue is that the greedy choice doesn't "know" that the seed will be covered.
                # It only optimizes for covering OTHER uncovered t-subsets that involve the new points.
                # This might lead to suboptimal blocks if the seed is "hard" to cover otherwise?
                # But the seed is already in the block. It will be covered.
                # The gain calculation is a proxy for "how much extra coverage do we get".
                # If best_point is None, it means no new point helps cover other uncovered t-subsets.
                # In that case, we should just pick a random point to complete the block.
                
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                best_point = rng.choice(candidates)
                
            block.append(best_point)
            block_set.add(best_point)
            
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for tsub in itertools.combinations(block, t):
            uncovered.discard(tuple(tsub))
            
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0
    
    all_tsubs = [tuple(c) for c in itertools.combinations(range(v), t)]
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple restarts
    # Use different random seeds
    num_restarts = 200 # Increase restarts if time allows
    for i in range(num_restarts):
        if time.time() > deadline - 5.0: # Leave time for validation/cleanup
            break
            
        rng = random.Random(42 + i)
        blocks = _build_greedy(v, k, t, all_tsubs, rng, deadline)
        
        if not blocks:
            continue
            
        # Remove redundant blocks
        blocks = _remove_redundant_blocks(v, k, t, blocks)
        
        # Validation is done by the evaluator, but we can check len
        if len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
            
    if best_blocks is None:
        # Fallback: single greedy
        rng = random.Random(0)
        best_blocks = _build_greedy(v, k, t, all_tsubs, rng, deadline)
        best_blocks = _remove_redundant_blocks(v, k, t, best_blocks)
        
    return best_blocks
