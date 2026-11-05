import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Pre-compute all t-subsets (triples)
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # Pre-compute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Pre-compute bitmask for each block
    # Note: num_triples for C(24,3) is 2024. This requires >64 bits, so we use Python integers (arbitrary precision).
    block_masks = [0] * num_blocks
    for bi, b in enumerate(all_blocks):
        mask = 0
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            mask |= (1 << idx)
        block_masks[bi] = mask
        
    # Pre-compute for each triple, the list of blocks that cover it
    triple_to_blocks = [[] for _ in range(num_triples)]
    for bi in range(num_blocks):
        mask = block_masks[bi]
        m = mask
        while m:
            lb = m & -m
            idx = lb.bit_length() - 1
            triple_to_blocks[idx].append(bi)
            m ^= lb
            
    # Helper to get list of set bits in a mask (for small masks or iteration)
    def get_bits(mask):
        bits = []
        m = mask
        while m:
            lb = m & -m
            bits.append(lb.bit_length() - 1)
            m ^= lb
        return bits

    def compute_counts(state):
        counts = [0] * num_triples
        for bi in state:
            mask = block_masks[bi]
            m = mask
            while m:
                lb = m & -m
                idx = lb.bit_length() - 1
                counts[idx] += 1
                m ^= lb
        return counts

    def remove_redundant(state, counts):
        changed = True
        while changed:
            changed = False
            for i in range(len(state) - 1, -1, -1):
                bi = state[i]
                mask = block_masks[bi]
                m = mask
                redundant = True
                while m:
                    lb = m & -m
                    idx = lb.bit_length() - 1
                    if counts[idx] <= 1:
                        redundant = False
                        break
                    m ^= lb
                
                if redundant:
                    m = mask
                    while m:
                        lb = m & -m
                        idx = lb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lb
                    state.pop(i)
                    changed = True
                    break
        return state, counts

    def greedy_construct(seed):
        random.seed(seed)
        uncovered_mask = (1 << num_triples) - 1
        current_blocks = []
        
        while uncovered_mask != 0:
            # Pick a random uncovered triple to break symmetry/ties
            # Instead of always picking the lowest bit, pick randomly from uncovered
            # This helps diversify the initial solution
            bits = get_bits(uncovered_mask)
            if not bits:
                break
            target_idx = random.choice(bits)
            
            candidates = triple_to_blocks[target_idx]
            best_gain = -1
            best_block_idx = -1
            best_block_mask = 0
            
            for bi in candidates:
                bm = block_masks[bi]
                gain = (uncovered_mask & bm).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = bi
                    best_block_mask = bm
                elif gain == best_gain and random.random() < 0.5:
                    best_block_idx = bi
                    best_block_mask = bm
                    
            if best_block_idx != -1:
                current_blocks.append(best_block_idx)
                uncovered_mask &= ~best_block_mask
                
        return current_blocks

    # --- New Strategy: VNS with Redundancy Penalty ---
    
    # Cost function: We want to minimize block count, but we also penalize redundancy 
    # to allow the search to "trade" redundancy for size reduction later.
    # However, the primary goal is size. 
    # Let's define the objective as: Size * 10000 + Redundancy.
    # If we swap a block and increase size by 1 but reduce redundancy by 1000, 
    # the cost increases. But if we swap and size stays same but redundancy drops, cost drops.
    # Actually, for covering design, we MUST maintain coverage. 
    # So the constraint is: All triples covered.
    # The objective is: Minimize Size.
    # The "Redundancy Penalty" approach usually works when we relax the constraint temporarily 
    # or when we use it to guide the search towards "tighter" solutions that are easier to compress.
    
    # Let's stick to a robust Local Search that optimizes Size directly but uses 
    # "Smart Swaps" based on overlap.
    
    def evaluate_state(state):
        # Returns (is_valid, size, redundancy_count)
        if not state:
            return (True, 0, 0)
        counts = compute_counts(state)
        uncovered = 0
        redundancy = 0
        for c in counts:
            if c == 0:
                uncovered += 1
            elif c > 1:
                redundancy += (c - 1)
        return (uncovered == 0, len(state), redundancy)

    def vns_search(initial_state, initial_counts, time_limit):
        state = initial_state[:]
        counts = initial_counts[:]
        state, counts = remove_redundant(state, counts)
        
        best_state = state[:]
        best_size = len(state)
        best_counts = counts[:]
        
        # Neighborhoods: 1-swap, 2-swap (if time permits)
        # For 1-swap: Replace block i with block j.
        # We need j to be "compatible" enough to not break coverage too badly, 
        # or we repair afterwards.
        
        # To speed up, we pre-compute for each block in the current state, 
        # the "unique" triples it covers (count == 1).
        # If a block has no unique triples, it's a candidate for removal/swap.
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try 1-Swaps
            # Identify blocks that are "redundant" (all their triples are covered by others)
            # or "low value" (cover few unique triples).
            
            # Let's try to swap a block B_i with a block B_j not in state.
            # Condition for a good swap:
            # 1. B_i has some unique triples (otherwise just remove it).
            # 2. B_j covers most of B_i's unique triples (so we don't create many holes).
            # 3. B_j covers many triples already covered by other blocks (increasing redundancy 
            #    is okay if it allows removing another block later? No, we just swap).
            
            # Actually, a simpler and effective VNS move for Set Cover:
            # "Exchange Move": Remove block i, add block j.
            # This is valid if the new set still covers everything.
            # If not, it's invalid.
            
            # We can speed this up by checking if B_j covers all "unique" triples of B_i.
            # If yes, the swap is valid (coverage maintained).
            # If B_j covers ALL unique triples of B_i, then removing B_i and adding B_j 
            # keeps all previously covered triples covered.
            
            # Let's find blocks that can be swapped out for "better" ones.
            # "Better" in what sense? 
            # If we swap B_i for B_j, and B_j covers MORE unique triples of the *remaining* set?
            # No, the size is constant.
            # The goal is to reach a state where we can REMOVE a block.
            # So we want to increase redundancy of other blocks to make them removable.
            
            # Strategy: 
            # 1. Find a block B_i in state.
            # 2. Find a block B_j not in state that covers all unique triples of B_i.
            # 3. If such B_j exists, swap B_i -> B_j.
            # 4. This swap might make other blocks redundant.
            # 5. Run remove_redundant.
            # 6. If size decreased, accept.
            
            # To avoid checking all pairs (N^2), we can prioritize B_i with many unique triples?
            # No, we want B_i with FEW unique triples, so it's easier to find a B_j covering them.
            
            # Let's iterate through blocks in state.
            # For each block i, compute its unique triples.
            # If unique triples is empty, remove it (handled by remove_redundant).
            # If unique triples is non-empty, look for a block j that covers them.
            
            # Optimization: 
            # Instead of random swaps, let's do a deterministic "Greedy Swap" pass.
            
            # 1. Compute unique triples for each block in state.
            unique_triples_per_block = {}
            for i, bi in enumerate(state):
                mask = block_masks[bi]
                unique_mask = 0
                m = mask
                while m:
                    lb = m & -m
                    idx = lb.bit_length() - 1
                    if counts[idx] == 1:
                        unique_mask |= (1 << idx)
                    m ^= lb
                unique_triples_per_block[i] = unique_mask
            
            # 2. For each block i, try to find a swap.
            # We'll pick a random order to diversify.
            indices = list(range(len(state)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > time_limit:
                    break
                if i >= len(state): # State might have changed? No, we only swap, size same until removal.
                    continue
                
                bi = state[i]
                unique_mask = unique_triples_per_block[i]
                
                if unique_mask == 0:
                    # Should have been removed by remove_redundant, but just in case
                    continue
                
                # Find a block j that covers unique_mask.
                # To be efficient, we can pick a random triple from unique_mask and look at candidates.
                # But we need to cover ALL unique triples.
                # This is a sub-problem. 
                # Approximation: Pick the triple in unique_mask with the fewest candidate blocks 
                # (excluding bi), and check if any of those candidates cover the rest of unique_mask.
                
                bits = get_bits(unique_mask)
                if not bits:
                    continue
                    
                # Pick the "hardest" triple to cover
                min_candidates = float('inf')
                target_tri = -1
                for tri_idx in bits:
                    cands = [c for c in triple_to_blocks[tri_idx] if c != bi]
                    if len(cands) < min_candidates:
                        min_candidates = len(cands)
                        target_tri = tri_idx
                        if min_candidates == 0:
                            break
                
                if target_tri == -1 or min_candidates == 0:
                    continue
                
                # Check candidates for target_tri
                found_swap = False
                # Limit candidates to check for speed
                cands_to_check = triple_to_blocks[target_tri]
                # Filter out bi
                cands_to_check = [c for c in cands_to_check if c != bi]
                
                # Shuffle candidates to try different ones
                random.shuffle(cands_to_check)
                
                for bj in cands_to_check:
                    # Check if bj covers all of unique_mask
                    bj_mask = block_masks[bj]
                    if (unique_mask & ~bj_mask) == 0:
                        # bj covers all unique triples of bi
                        # Perform swap: remove bi, add bj
                        # But wait, if bj is already in state, we can't add it twice.
                        if bj in state:
                            continue
                        
                        # Execute swap
                        state[i] = bj
                        
                        # Update counts
                        # Remove bi
                        m = block_masks[bi]
                        while m:
                            lb = m & -m
                            idx = lb.bit_length() - 1
                            counts[idx] -= 1
                            m ^= lb
                        # Add bj
                        m = block_masks[bj]
                        while m:
                            lb = m & -m
                            idx = lb.bit_length() - 1
                            counts[idx] += 1
                            m ^= lb
                        
                        # This swap is valid (coverage maintained).
                        # Now, this increased redundancy for triples covered by bj (which were already covered by others).
                        # So some blocks might now be redundant.
                        state, counts = remove_redundant(state, counts)
                        
                        if len(state) < best_size:
                            best_size = len(state)
                            best_state = state[:]
                            best_counts = counts[:]
                            improved = True
                            # Recompute unique_triples_per_block for the new state
                            unique_triples_per_block = {}
                            for i_new, bi_new in enumerate(state):
                                mask = block_masks[bi_new]
                                um = 0
                                m = mask
                                while m:
                                    lb = m & -m
                                    idx = lb.bit_length() - 1
                                    if counts[idx] == 1:
                                        um |= (1 << idx)
                                    m ^= lb
                                unique_triples_per_block[i_new] = um
                            # Restart the inner loop? Or just continue with new indices.
                            # Since state changed, indices are invalid. Break out.
                            break
                        else:
                            # No improvement, revert swap? 
                            # In VNS, if we don't improve, we usually keep the move if it's "equivalent" 
                            # or revert. Reverting is safer to avoid drifting into bad local minima.
                            # Revert
                            state[i] = bi
                            m = block_masks[bj]
                            while m:
                                lb = m & -m
                                idx = lb.bit_length() - 1
                                counts[idx] -= 1
                                m ^= lb
                            m = block_masks[bi]
                            while m:
                                lb = m & -m
                                idx = lb.bit_length() - 1
                                counts[idx] += 1
                                m ^= lb
                            
                            # Try next candidate
                            continue
                        
                        if found_swap:
                            break
                    # If we broke out of the candidate loop due to improvement, break outer
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                # No 1-swap improvement found.
                # Try a "Kick": Randomly replace 1 block with a random block, repair, optimize.
                # This helps escape local minima where no valid 1-swap exists.
                if len(state) > 1:
                    idx = random.randint(0, len(state) - 1)
                    bi_old = state[idx]
                    
                    # Pick a random block to replace with
                    # To make it effective, pick a block that shares some points with bi_old
                    # or just random.
                    bi_new = random.randint(0, num_blocks - 1)
                    if bi_new == bi_old:
                        continue
                    
                    # Swap
                    state[idx] = bi_new
                    
                    # Update counts
                    m = block_masks[bi_old]
                    while m:
                        lb = m & -m
                        idx_tri = lb.bit_length() - 1
                        counts[idx_tri] -= 1
                        m ^= lb
                    m = block_masks[bi_new]
                    while m:
                        lb = m & -m
                        idx_tri = lb.bit_length() - 1
                        counts[idx_tri] += 1
                        m ^= lb
                    
                    # Repair: Find uncovered triples and add blocks
                    uncovered_mask = 0
                    for i_tri in range(num_triples):
                        if counts[i_tri] == 0:
                            uncovered_mask |= (1 << i_tri)
                            
                    while uncovered_mask != 0:
                        if time.time() > time_limit:
                            break
                        bits = get_bits(uncovered_mask)
                        if not bits:
                            break
                        target_idx = random.choice(bits)
                        candidates = triple_to_blocks[target_idx]
                        best_gain = -1
                        best_block_idx = -1
                        best_block_mask = 0
                        for bi in candidates:
                            bm = block_masks[bi]
                            gain = (uncovered_mask & bm).bit_count()
                            if gain > best_gain:
                                best_gain = gain
                                best_block_idx = bi
                                best_block_mask = bm
                        if best_block_idx != -1:
                            state.append(best_block_idx)
                            m = best_block_mask
                            while m:
                                lb = m & -m
                                idx_tri = lb.bit_length() - 1
                                counts[idx_tri] += 1
                                m ^= lb
                            uncovered_mask &= ~best_block_mask
                        else:
                            break
                            
                    # Optimize
                    state, counts = remove_redundant(state, counts)
                    
                    if len(state) < best_size:
                        best_size = len(state)
                        best_state = state[:]
                        best_counts = counts[:]
                        improved = True
                    else:
                        # Revert to best_state? Or keep current?
                        # Keeping current allows further exploration from this "worse" state.
                        # But if it's much worse, we might get stuck.
                        # Let's keep current state for the next iteration's VNS.
                        pass
            
            # If we found an improvement, the loop continues with the new state.
            # If not, we might be stuck. The kick handles this.

        return best_state, best_size

    # Main Loop
    best_blocks = None
    best_len = float('inf')
    
    # Run several initial greedy restarts
    for restart in range(15):
        if time.time() > deadline - 20:
            break
        state = greedy_construct(42 + restart)
        counts = compute_counts(state)
        state, counts = remove_redundant(state, counts)
        
        if len(state) < best_len:
            best_len = len(state)
            best_blocks = state[:]

    # Run VNS on the best initial solution
    if best_blocks:
        # Give VNS most of the time
        vns_time_limit = deadline - 5
        best_blocks, best_len = vns_search(best_blocks, compute_counts(best_blocks), vns_time_limit)
        
        # Final cleanup
        final_counts = compute_counts(best_blocks)
        final_blocks, _ = remove_redundant(best_blocks[:], final_counts)
        
        # Convert indices to actual blocks
        return [list(all_blocks[bi]) for bi in final_blocks]
    else:
        # Fallback (should not happen)
        return []

