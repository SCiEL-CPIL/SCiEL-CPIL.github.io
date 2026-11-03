import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Precompute mapping from 4-subset to index
    # There are C(19, 4) = 3876 subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Helper to get index of a t-subset
    def get_idx(sub):
        return t_to_idx[sub]

    # Greedy Construction
    # We track uncovered subsets using a set of indices for simplicity first, 
    # but a bitset (Python int) is faster for intersection and popcount.
    
    # Initialize bitset of uncovered subsets
    # Bit i is 1 if subset i is uncovered
    uncovered_mask = (1 << num_t_subsets) - 1
    
    blocks = []
    
    # To speed up greedy, we precompute the list of t-subsets for any potential block?
    # No, we compute on the fly.
    
    while uncovered_mask:
        if time.time() > deadline - 10.0:
            break
            
        # Start a new block with the first uncovered subset
        # Find the index of the lowest set bit
        first_uncovered_idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
        first_uncovered_sub = all_t_subsets[first_uncovered_idx]
        
        current_block = list(first_uncovered_sub)
        
        while len(current_block) < k:
            best_gain = -1
            best_point = -1
            
            # Calculate gain for each candidate point
            # Gain = number of currently uncovered t-subsets that will be covered by adding x
            # A t-subset s is covered by B U {x} if s is a subset of B U {x}
            # s must contain x? No, s is a t-subset of the final k-block.
            # If we add x, the new block is B' = B U {x}.
            # The t-subsets covered by B' are all t-subsets of B'.
            # We want to count how many of these are currently in uncovered_mask.
            
            # Optimization: 
            # The t-subsets of B' are:
            # 1. t-subsets of B (already covered by previous steps? No, B is partial)
            #    Actually, any t-subset of B is a t-subset of B'.
            # 2. t-subsets of B' that include x.
            #    These are formed by taking x and (t-1) elements from B.
            
            # Let's just generate all t-subsets of B U {x} and check the mask.
            # Size of B is at most k-1 = 6.
            # Number of t-subsets in a 7-set is 35.
            # This is small.
            
            for x in range(v):
                if x in current_block:
                    continue
                
                # Construct the hypothetical block
                hyp_block = current_block + [x]
                # Sort hyp_block to ensure combinations match the sorted keys in t_to_idx
                hyp_block_sorted = sorted(hyp_block)
                
                # Generate all t-subsets of hyp_block
                # Note: hyp_block has len(current_block)+1 elements
                gain = 0
                for sub in itertools.combinations(hyp_block_sorted, t):
                    idx = get_idx(sub)
                    if (uncovered_mask >> idx) & 1:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            current_block.append(best_point)
        
        # Finalize block
        current_block.sort()
        blocks.append(current_block)
        
        # Update uncovered mask
        # Remove all t-subsets of current_block from uncovered_mask
        for sub in itertools.combinations(current_block, t):
            idx = get_idx(sub)
            uncovered_mask &= ~(1 << idx)
            
    # If we broke out of greedy early due to time, we might not have a valid covering.
    # We need to ensure validity. If not valid, we must finish the greedy.
    # But 115s is plenty for greedy. The greedy is O(C(v,t) * v * C(k,t)).
    # C(19,4)=3876. v=19. C(7,4)=35.
    # 3876 * 19 * 35 ~ 2.5M ops. Very fast.
    
    # Now, Local Search to reduce block count
    # Data structures for local search
    # counts[i] = number of blocks covering t-subset i
    counts = [0] * num_t_subsets
    
    # Initialize counts
    for b in blocks:
        for sub in itertools.combinations(b, t):
            counts[get_idx(sub)] += 1
    
    # Helper to check if a block is redundant
    def is_redundant(block):
        for sub in itertools.combinations(block, t):
            if counts[get_idx(sub)] <= 1:
                return False
        return True

    # Helper to check if a swap is valid
    # Swap B_out with B_in.
    # Valid if for every sub in B_out, counts[sub] > 1 OR sub in B_in
    def is_valid_swap(b_out, b_in):
        # Precompute set of subs in b_in for O(1) lookup?
        # b_in has size 7. subs are 35.
        # We can just check membership in the generated list.
        b_in_subs = set(itertools.combinations(b_in, t))
        
        for sub in itertools.combinations(b_out, t):
            idx = get_idx(sub)
            if counts[idx] <= 1:
                # It is covered only by b_out (in the current set)
                # So it MUST be covered by b_in
                if sub not in b_in_subs:
                    return False
        return True

    # Perform swaps
    # We will run this loop until time is up or no improvements are found in a while
    # Strategy: 
    # 1. Try to remove redundant blocks.
    # 2. If none, try random swaps that maintain validity.
    #    After a swap, check for redundancies again.
    
    # To make swaps more effective, we can prioritize B_out that are "critical" 
    # or just random. Random is simple and often works well with many iterations.
    
    # Precompute all possible blocks?
    # C(19,7) = 50388.
    # We can iterate through all possible blocks for B_in if needed, but random sampling is faster per step.
    
    # Let's use a list of all possible blocks to pick B_in from?
    # Or just generate random blocks?
    # Random blocks might be poor. 
    # Better: Pick B_out. Find B_in that covers the most "critical" subsets of B_out.
    # Critical subsets of B_out are those with counts[sub] == 1.
    # If no critical subsets, the block is redundant (handled in step 1).
    # So if we are in swap phase, every block has at least one critical subset.
    
    # Let's implement a "Guided Swap":
    # 1. Identify critical subsets of B_out (those with count 1).
    # 2. We need B_in to cover ALL of them.
    # 3. If no single B_in can cover all critical subsets of B_out, the swap is impossible.
    # 4. If possible, pick one.
    
    # This constraint is very strong. 
    # Alternative: Relax validity? No, must remain valid covering.
    
    # Let's try a different local search: 
    # Instead of 1-for-1 swaps that maintain validity, 
    # we can use a "Penalty" method or just try to find a B_in that covers the union of critical subsets.
    
    # Actually, if counts[sub] == 1 for some sub in B_out, then B_in MUST contain that sub.
    # If B_out has multiple disjoint critical subsets, B_in must contain their union.
    # If the union size > k, swap is impossible.
    
    # Let's just do random valid swaps.
    
    # To speed up is_valid_swap, we can precompute the set of t-subsets for each block?
    # There are 50388 blocks. Storing 35 tuples for each is ~1.7M tuples. Feasible.
    # But we don't know which blocks will be in B_in.
    
    # Let's stick to the simple check.
    
    # Optimization: 
    # If we find a redundant block, remove it immediately.
    # This reduces the size of the covering.
    
    # We will alternate between:
    # 1. Scan for redundant blocks. Remove if found.
    # 2. If no redundant blocks, perform a random valid swap.
    
    # To ensure we explore enough, we will do many swaps.
    
    # Precompute all 7-subsets of 19 points?
    all_blocks = list(itertools.combinations(range(v), k))
    # 50388 blocks.
    
    # For faster lookup of "does B_in cover sub?", we can use the t_to_idx map.
    
    # Let's run the local search
    # We need to be careful with time.
    
    # To make the search more effective, we can use "Simulated Annealing" style acceptance?
    # No, we must maintain validity. So only valid swaps are allowed.
    # The goal is to reach a state where a block becomes redundant.
    
    # Heuristic for B_out selection:
    # Pick a block with the highest number of "critical" subsets (count == 1)?
    # Or the lowest? 
    # If a block has many critical subsets, it's hard to swap out.
    # If a block has few critical subsets, it's easier to swap out.
    # So pick B_out with the MINIMUM number of critical subsets (but > 0, since if 0 it's redundant).
    
    # Let's implement this.
    
    while time.time() < deadline - 2.0:
        # Step 1: Remove redundant blocks
        removed_any = False
        i = 0
        while i < len(blocks):
            if is_redundant(blocks[i]):
                # Remove block
                b_out = blocks[i]
                for sub in itertools.combinations(b_out, t):
                    counts[get_idx(sub)] -= 1
                blocks.pop(i)
                removed_any = True
                # Do not increment i, check the next block at same index
            else:
                i += 1
        
        if removed_any:
            continue
        
        # Step 2: If no redundant blocks, try a swap
        if len(blocks) == 0:
            break
            
        # Find B_out with min critical subsets
        min_crit_count = float('inf')
        b_out_idx = -1
        b_out = None
        
        # Sample a few blocks to find a good candidate to save time?
        # Or scan all? Scanning all is O(N * 35). N is ~150. 150*35 = 5250 ops. Fast.
        for i, b in enumerate(blocks):
            crit_count = 0
            for sub in itertools.combinations(b, t):
                if counts[get_idx(sub)] == 1:
                    crit_count += 1
            if crit_count < min_crit_count:
                min_crit_count = crit_count
                b_out_idx = i
                b_out = b
        
        # If min_crit_count is 0, it should have been removed in Step 1.
        # But due to race conditions or logic, ensure it's > 0.
        if min_crit_count == 0:
            continue
            
        # Try to find a valid B_in
        # We need B_in to cover all critical subsets of B_out.
        # Let C_out be the set of critical subsets of B_out.
        crit_subs = []
        for sub in itertools.combinations(b_out, t):
            if counts[get_idx(sub)] == 1:
                crit_subs.append(sub)
                
        # If crit_subs is empty, B_out is redundant (shouldn't happen)
        if not crit_subs:
            continue
            
        # Find a block B_in (from all_blocks or random) that contains all crit_subs
        # And is not already in blocks (to avoid duplicate blocks, though distinctness is required)
        # Actually, B_in must be distinct from existing blocks? 
        # "blocks must be distinct". Yes.
        
        # We can iterate through all_blocks? 50k iterations per swap.
        # 50k * 35 checks = 1.7M ops per swap.
        # If we do 1000 swaps, that's 1.7B ops. Too slow in Python.
        
        # We need a faster way to find B_in.
        # Or we can just pick a random B_in and check validity.
        # But random B_in is unlikely to cover specific crit_subs.
        
        # Alternative: Construct B_in?
        # B_in must contain the union of crit_subs.
        # Let U = union of crit_subs.
        # If len(U) > k, no such B_in exists.
        # If len(U) <= k, we can fill the rest of B_in with random points.
        
        # Let's try this construction approach.
        
        # 1. Compute U = union of crit_subs
        # 2. If len(U) > k, skip this B_out.
        # 3. Else, generate candidate B_in by adding random points to U until size k.
        # 4. Check if B_in is valid (covers all crit_subs - guaranteed by construction) 
        #    AND is valid swap (covers all other subs in B_out that are not critical? 
        #    Wait, is_valid_swap checks ALL subs in B_out.
        #    If sub in B_out is NOT critical (count > 1), it's already covered by another block.
        #    So B_in does NOT need to cover it.
        #    So the ONLY constraint on B_in is that it must cover all CRITICAL subs of B_out.
        
        # So, if we construct B_in to contain U (union of critical subs), the swap is ALWAYS valid 
        # with respect to the critical subs.
        # What about the non-critical subs? They are covered by other blocks, so removing B_out 
        # doesn't uncover them.
        
        # So, ANY block B_in that contains U is a valid swap partner for B_out!
        
        # We just need to ensure B_in is distinct from existing blocks.
        
        # Algorithm for Swap:
        # 1. Identify critical subs of B_out.
        # 2. Compute U = union.
        # 3. If len(U) > k, try another B_out.
        # 4. Generate a random block B_in containing U.
        # 5. If B_in is already in blocks, regenerate.
        # 6. Perform swap: Remove B_out, Add B_in.
        # 7. Update counts.
        
        # This is much faster!
        
        # Execute Swap
        U = set()
        for sub in crit_subs:
            U.update(sub)
            
        if len(U) > k:
            # Cannot swap this block. Try next iteration (will pick different B_out)
            continue
            
        # Generate B_in
        # We need to add k - len(U) random points from v - len(U) available points.
        available_points = [p for p in range(v) if p not in U]
        num_to_add = k - len(U)
        
        # Shuffle and pick
        random.shuffle(available_points)
        B_in = list(U) + available_points[:num_to_add]
        B_in.sort()
        
        # Check if B_in is already in blocks
        # To make this O(1), we can maintain a set of blocks?
        # Blocks are lists. Convert to tuple for set.
        # Let's maintain a set `block_set` for O(1) lookup.
        
        # We didn't initialize block_set. Let's add it.
        # But we are in the middle of the code. 
        # We can just check `if B_in in blocks`. List of lists comparison is O(N*k).
        # N=150, k=7. 1050 ops. Fast enough.
        
        if B_in in blocks:
            # Try a few more random permutations
            found_new = False
            for _ in range(10):
                random.shuffle(available_points)
                B_in_try = list(U) + available_points[:num_to_add]
                B_in_try.sort()
                if B_in_try not in blocks:
                    B_in = B_in_try
                    found_new = True
                    break
            if not found_new:
                continue
                
        # Perform Swap
        # Remove B_out
        for sub in itertools.combinations(b_out, t):
            counts[get_idx(sub)] -= 1
            
        # Add B_in
        for sub in itertools.combinations(B_in, t):
            counts[get_idx(sub)] += 1
            
        blocks[b_out_idx] = B_in

    return blocks
