import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s buffer for safety
    
    # 1. Precomputation
    # Map t-subsets to IDs
    t_subs = list(itertools.combinations(range(v), t))
    t_to_id = {ts: i for i, ts in enumerate(t_subs)}
    n_t = len(t_subs)
    full_mask = (1 << n_t) - 1
    
    # Generate all k-subsets (blocks)
    blocks = list(itertools.combinations(range(v), k))
    n_k = len(blocks)
    
    # Precompute masks for each block
    # block_masks[i] is a bitset of t-subset IDs covered by block i
    block_masks = [0] * n_k
    
    # Precompute which blocks cover each t-subset
    # t_to_blocks[j] is a list of block indices that cover t-subset j
    t_to_blocks = [[] for _ in range(n_t)]
    
    for i, block in enumerate(blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            tid = t_to_id[t_sub]
            mask |= (1 << tid)
            t_to_blocks[tid].append(i)
        block_masks[i] = mask
        
    # Check time
    if time.time() > deadline - 20:
        # Fallback: simple greedy
        return greedy_cover_fallback(v, k, t, block_masks, t_to_id, full_mask)

    # 2. Greedy Initialization
    def greedy_init():
        covered = 0
        solution = []
        remaining = n_t
        while covered != full_mask:
            best_block = -1
            best_gain = 0
            # Optimization: only check blocks that might add something?
            # For now, check all.
            for i in range(n_k):
                gain = (block_masks[i] & ~covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = i
                if gain == 0:
                    continue
                # Early exit if we found a perfect gain? Max gain is k choose t? No, depends on context.
                # Max possible gain is number of uncovered subsets in this block.
            if best_block == -1:
                break # Should not happen if covering exists
            solution.append(best_block)
            covered |= block_masks[best_block]
            remaining -= best_gain
            if time.time() > deadline - 10:
                break
        return solution

    # 3. Local Search
    def local_search(solution):
        current_mask = 0
        for b in solution:
            current_mask |= block_masks[b]
        
        solution_set = set(solution)
        
        while True:
            if time.time() > deadline - 5:
                break
            changed = False
            # Shuffle to avoid bias?
            # random.shuffle(solution)
            
            for i, b in enumerate(list(solution)): # Iterate over copy
                # Try removing b
                new_mask = current_mask ^ block_masks[b]
                
                if new_mask == full_mask:
                    # Valid removal
                    solution.pop(i)
                    solution_set.remove(b)
                    current_mask = new_mask
                    changed = True
                    break # Restart pass
                else:
                    # Find uncovered t-subsets
                    uncovered_mask = full_mask & ~new_mask
                    # Get list of uncovered t-IDs
                    # Efficient way to get set bits:
                    # Since n_t is ~2000, we can iterate or use bit tricks.
                    # Let's just iterate through the bits of uncovered_mask.
                    # Or better, keep a list of active t-ids?
                    # For simplicity, extract IDs.
                    unc_ids = []
                    m = uncovered_mask
                    while m:
                        lsb = m & -m
                        tid = lsb.bit_length() - 1
                        unc_ids.append(tid)
                        m ^= lsb
                    
                    if not unc_ids:
                        continue # Should not happen if new_mask != full_mask? 
                                 # Wait, if new_mask != full_mask, uncovered_mask is not 0.
                    
                    # Find intersection of t_to_blocks for all unc_ids
                    # Start with the smallest list
                    unc_ids.sort(key=lambda x: len(t_to_blocks[x]))
                    candidates = set(t_to_blocks[unc_ids[0]])
                    
                    for tid in unc_ids[1:]:
                        candidates.intersection_update(t_to_blocks[tid])
                        if not candidates:
                            break
                    
                    # Filter out current block and blocks already in solution
                    # We want a block NOT in solution to replace b?
                    # Actually, we can replace b with b? No, that's no change.
                    # We can replace b with another block already in solution?
                    # If b' is in solution, replacing b with b' means removing b and keeping b'.
                    # This is equivalent to removing b if b' covers everything b covered?
                    # No, b' might not cover everything b covered.
                    # If we replace b with b', the new mask is (current_mask ^ mask[b]) | mask[b'].
                    # We know (current_mask ^ mask[b]) misses unc_ids.
                    # So we need mask[b'] to cover unc_ids.
                    # If b' is in solution, it's fine.
                    # But we must ensure we don't add duplicates if we just "add" b'.
                    # Here we are "replacing" b with b'.
                    # So b is removed, b' is added.
                    # If b' == b, no change.
                    # If b' is in solution, we just remove b.
                    # Wait, if b' is in solution, and we remove b, does the covering hold?
                    # Yes, if b' covers unc_ids.
                    # So the logic holds.
                    
                    # Filter candidates
                    valid_candidates = [c for c in candidates if c != b]
                    
                    if valid_candidates:
                        # Pick a candidate
                        # Heuristic: Pick one that maximizes additional coverage?
                        # Or just random?
                        # Let's pick one that covers the most *other* uncovered things?
                        # There are no other uncovered things.
                        # Let's pick one that is "most central" or just random.
                        # Random might help escape local minima.
                        # Deterministic might be faster.
                        # Let's pick the one with the highest "degree" or just first.
                        # Let's try to pick one that is NOT in solution to increase diversity?
                        # Or one that IS in solution to allow deletion?
                        # If we pick one in solution, we effectively just delete b.
                        # If we pick one NOT in solution, we swap b for a new block.
                        # Swapping might enable future deletions.
                        # Let's prefer blocks NOT in solution to explore new space.
                        new_blocks = [c for c in valid_candidates if c not in solution_set]
                        if new_blocks:
                            chosen = random.choice(new_blocks)
                        else:
                            chosen = random.choice(valid_candidates)
                        
                        # Perform replacement
                        solution[i] = chosen
                        solution_set.remove(b)
                        solution_set.add(chosen)
                        current_mask = new_mask | block_masks[chosen]
                        changed = True
                        break # Restart pass
            
            if not changed:
                break
                
        return solution

    # Run greedy
    best_solution = greedy_init()
    best_count = len(best_solution)
    
    # Run local search
    current_solution = local_search(best_solution)
    if len(current_solution) < best_count:
        best_solution = current_solution
        best_count = len(best_solution)
    
    # Try a few more random restarts if time permits?
    # Given the precomputation cost, maybe just one good run is enough.
    # But greedy is deterministic.
    # We can add noise to greedy.
    
    # Let's add a randomized greedy variant
    def greedy_init_randomized():
        covered = 0
        solution = []
        while covered != full_mask:
            best_blocks = []
            max_gain = 0
            for i in range(n_k):
                gain = (block_masks[i] & ~covered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_blocks = [i]
                elif gain == max_gain and gain > 0:
                    best_blocks.append(i)
            if not best_blocks:
                break
            # Add some noise: pick randomly from top 10%?
            # Or just random from best_blocks
            chosen = random.choice(best_blocks)
            solution.append(chosen)
            covered |= block_masks[chosen]
            if time.time() > deadline - 10:
                break
        return solution

    # Run a few randomized greedy + local search
    for _ in range(5):
        if time.time() > deadline - 15:
            break
        sol = greedy_init_randomized()
        sol = local_search(sol)
        if len(sol) < best_count:
            best_solution = sol
            best_count = len(sol)

    return [list(blocks[b]) for b in best_solution]

def greedy_cover_fallback(v, k, t, block_masks, t_to_id, full_mask):
    # Very simple fallback, just in case precomputation takes too long
    # But precomputation is done before this call in the main flow?
    # No, this is called if precomputation takes too long?
    # Actually, the check is after precomputation.
    # So we have block_masks.
    covered = 0
    solution = []
    while covered != full_mask:
        best_block = -1
        best_gain = 0
        for i, mask in enumerate(block_masks):
            gain = (mask & ~covered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block = i
        if best_block == -1:
            break
        solution.append(best_block)
        covered |= block_masks[best_block]
    return solution
