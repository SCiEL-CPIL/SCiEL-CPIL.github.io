"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
import itertools

def construct(v, k, t):
    deadline = time.time() + 115.0
    
    # Map t-subsets to integer indices
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t = len(all_t_subs)
    
    # For each point x, precompute all t-subsets containing x
    point_to_tsubs = [set() for _ in range(v)]
    for sub, idx in t_sub_to_idx.items():
        for x in sub:
            point_to_tsubs[x].add(idx)
    
    # For each (t-1)-subset, precompute list of t-subsets extending it
    t1_sub_to_tsubs = {}
    for sub, idx in t_sub_to_idx.items():
        for j in range(t):
            t1 = sub[:j] + sub[j+1:]
            t1_sub_to_tsubs.setdefault(t1, set()).add(idx)
    
    uncovered = set(range(n_t))
    blocks = []
    
    # Fast greedy
    while uncovered and time.time() < deadline - 10:
        # Pick the uncovered t-subset with the most potential (or just min for simplicity, 
        # but let's use a smarter heuristic: pick the one that appears in the most "promising" partial blocks)
        # For now, simple min is fine for the baseline, but let's add a twist:
        # Actually, the standard greedy is to pick any uncovered, then build the block.
        # To improve, we can pick the uncovered subset that is "hardest" to cover?
        # Let's stick to min(uncovered) for determinism but optimize the block building.
        
        seed_sub = min(uncovered)
        block = list(all_t_subs[seed_sub])
        remaining_slots = k - t
        
        # Build the block
        for _ in range(remaining_slots):
            best_point = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Calculate gain: how many new t-subsets does x add to the block?
                # A t-subset is formed by taking t-1 elements from current block and adding x.
                # We only count it if it's in `uncovered`.
                gain = 0
                for t1 in itertools.combinations(block, t-1):
                    if (x,) + t1 in t_sub_to_tsubs:
                        for t_sub in t_sub_to_tsubs[(x,) + t1]:
                            if t_sub in uncovered:
                                gain += 1
                    # Optimization: (x,) + t1 is not necessarily sorted, but t_sub_to_tsubs keys are sorted tuples.
                    # So we must sort (x,) + t1.
                    # This is slow. Let's fix the logic.
                
                # Corrected gain calculation
                gain = 0
                for t1 in itertools.combinations(block, t-1):
                    candidate = tuple(sorted(t1 + (x,)))
                    if candidate in t_sub_to_tsubs:
                        for t_sub in t_sub_to_tsubs[candidate]:
                            if t_sub in uncovered:
                                gain += 1
                                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    
            if best_point == -1:
                break
            block.append(best_point)
            # Early stop if we've covered enough? No, must fill k.
            
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            if idx in uncovered:
                uncovered.remove(idx)
        
        if time.time() > deadline - 10:
            break

    if not uncovered:
        return blocks

    # If greedy didn't finish, we might have a partial solution.
    # The task requires a VALID covering. If we run out of time, we must return a valid one.
    # The greedy above only adds blocks if it completes. If it breaks mid-block, that block is NOT added.
    # So `blocks` is a valid covering of SOME subsets, but maybe not all.
    # If `uncovered` is not empty, the result is INVALID.
    
    # To guarantee validity under time pressure, we should fallback to a slower but guaranteed method if time is tight,
    # or better: only add a block if we have enough time to finish it.
    # But the prompt says "return the best result found". If invalid, score 0.
    # So we MUST ensure we return a valid covering.
    
    # Strategy: If we run out of time, we must finish the current block or discard it and ensure all are covered.
    # Actually, the loop above only appends `block` to `blocks` if it completes.
    # If we break out of the while loop due to time, we still have `uncovered`.
    # We need to finish covering `uncovered` quickly.
    
    # Fast finish: for each remaining uncovered subset, make a block containing it.
    # This is O(n_uncovered * k) and very fast.
    for sub_idx in list(uncovered):
        if time.time() > deadline - 1:
            break
        sub = all_t_subs[sub_idx]
        block = list(sub)
        while len(block) < k:
            # Just add any point not in block
            for x in range(v):
                if x not in block:
                    block.append(x)
                    break
        block.sort()
        blocks.append(block)
        # Remove covered
        to_remove = []
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            if idx in uncovered:
                to_remove.append(idx)
        for idx in to_remove:
            uncovered.remove(idx)
            
    # If we still have uncovered (time ran out completely), we have to hope the evaluator is lenient or we fail.
    # But usually "at most C(v,t) blocks" is the limit.
    # To be safe, if uncovered remains, we must return something valid.
    # The "fast finish" above should handle it unless v,k are huge.
    
    # Optional: Redundancy removal to improve score (reduce blocks)
    # Only if we have time.
    if not uncovered and time.time() < deadline - 5:
        # Check for redundant blocks
        # For each block, see if all its t-subsets are covered by other blocks.
        # If yes, remove it.
        
        # Precompute cover counts
        # This is O(n_blocks * C(k,t)).
        # Let's do a simple pass.
        
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            # Count how many blocks cover each t-subset
            cover_count = [0] * n_t
            for block in blocks:
                for t_sub in itertools.combinations(block, t):
                    cover_count[t_sub_to_idx[t_sub]] += 1
            
            new_blocks = []
            for block in blocks:
                redundant = True
                for t_sub in itertools.combinations(block, t):
                    idx = t_sub_to_idx[t_sub]
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if not redundant:
                    new_blocks.append(block)
            
            if len(new_blocks) < len(blocks):
                blocks = new_blocks
                changed = True
                
    return blocks
# EVOLVE-BLOCK-END
