import time
import random
import itertools
import sys

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin before 150s

    # Precompute all t-subsets and their indices
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    full_mask = (1 << N) - 1

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Precompute bitmasks for each block
    # mask[i] has bit j set if all_t[j] is a subset of all_blocks[i]
    block_masks = [0] * num_blocks
    for i, b in enumerate(all_blocks):
        mask = 0
        for s in itertools.combinations(b, t):
            ti = t_idx[s]
            mask |= (1 << ti)
        block_masks[i] = mask

    # Helper: popcount for int
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    def greedy_solve(seed_val):
        random.seed(seed_val)
        covered = 0
        used = []
        remaining = set(range(num_blocks))
        
        # To speed up greedy, we can sort blocks by initial potential?
        # Or just iterate.
        # Optimization: We can maintain a list of active blocks.
        
        while covered != full_mask:
            if time.time() > deadline - 5:
                break
            
            best_gain = -1
            best_block = -1
            # We can iterate over all remaining blocks.
            # To introduce randomness in tie-breaking, we can shuffle or just pick random among max.
            
            # Optimization: If we have many blocks, iterating 75k in Python loop is ~2ms.
            # 300 iterations * 2ms = 600ms. Acceptable.
            
            max_gain_val = 0
            candidates = []
            
            for bi in remaining:
                gain = popcount(block_masks[bi] & ~covered)
                if gain > max_gain_val:
                    max_gain_val = gain
                    candidates = [bi]
                elif gain == max_gain_val:
                    candidates.append(bi)
            
            if not candidates:
                break
                
            # Random choice from best candidates
            bi = random.choice(candidates)
            used.append(bi)
            remaining.remove(bi)
            covered |= block_masks[bi]
            
        return used

    def repair_and_optimize(blocks, deadline):
        if not blocks:
            return blocks
        
        current = list(blocks)
        current_set = set(current)
        
        # Pre-compute OR of all blocks? 
        # OR is expensive to recompute if we remove one.
        # But N is small. 
        # Let's maintain `total_covered` and a frequency count?
        # No, bitmask OR is fast.
        
        def get_total_covered(blks):
            c = 0
            for b in blks:
                c |= block_masks[b]
            return c

        # Initial total covered
        total_cov = get_total_covered(current)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            n = len(current)
            
            # Pass 1: Try to remove redundant blocks
            # A block is redundant if its mask is subset of the OR of the others.
            # OR(others) = total_cov if block is not unique? 
            # Actually, if we remove b, new_cov = OR(current \ {b}).
            # b is redundant if block_masks[b] is subset of OR(current \ {b}).
            # Note: OR(current) = OR(current \ {b}) | block_masks[b].
            # So block_masks[b] is subset of OR(current \ {b}) iff OR(current \ {b}) == total_cov.
            
            for i in range(len(current)):
                bi = current[i]
                # Calculate OR of others
                # Efficient way: 
                # If we keep a "count of coverage" for each bit, we can check if any bit has count 1.
                # But bitsets don't have easy "count 1" without iterating bits.
                # Iterating 11628 bits is fast (~0.1ms).
                
                # Let's just compute OR of others.
                # This is O(n) bitwise ops.
                # n ~ 300. 300 * (11628/64) ops.
                # This might be slow if done for every i in every pass.
                
                # Optimization: 
                # Keep an array `cover_counts` of size N? 
                # N=11628.
                # When we remove a block, we decrement counts for its bits.
                # If min(counts) > 0, we can remove.
                
                # Let's switch to count-based representation for the repair phase.
                # This is a structural change within the function.
                pass

        # --- Revised Repair Strategy using Counts ---
        # We convert `current` list of indices to a list of masks.
        # We maintain `counts` array of length N.
        
        counts = [0] * N
        for bi in current:
            mask = block_masks[bi]
            # Iterate set bits in mask
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        
        def try_remove_and_repair(current_list, counts):
            """
            Attempts to improve the list. Returns (new_list, new_counts, improved_flag).
            """
            improved = False
            # 1. Try removing any block that has no exclusive bits
            for i in range(len(current_list)):
                bi = current_list[i]
                mask = block_masks[bi]
                # Check if any bit in mask has count == 1
                has_exclusive = False
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        has_exclusive = True
                        break
                    m ^= lsb
                    
                if not has_exclusive:
                    # Safe to remove
                    current_list.pop(i)
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                    improved = True
                    break # Restart scan
            
            if improved:
                return current_list, counts, True
            
            # 2. Try swapping
            # Pick a random block to try to replace
            if len(current_list) > 0:
                i = random.randint(0, len(current_list) - 1)
                bi = current_list[i]
                mask = block_masks[bi]
                
                # Identify exclusive bits
                exclusive_bits = []
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        exclusive_bits.append(idx)
                    m ^= lsb
                
                if exclusive_bits:
                    # We need a new block that covers all exclusive_bits
                    # Find candidates
                    candidates = []
                    # We can iterate all blocks not in current_list
                    # To speed up, we can pre-filter?
                    # For now, iterate all.
                    current_set = set(current_list)
                    
                    best_candidate = -1
                    best_score = -1
                    
                    # Optimization: Only check blocks that cover the first exclusive bit
                    first_ex = exclusive_bits[0]
                    # We don't have inverse index of bit->blocks precomputed for speed in this scope?
                    # We can precompute it globally.
                    
                    # Let's assume we can iterate. 
                    # 75k blocks.
                    # Check if block covers first_ex.
                    # Then check if covers all other exclusive_bits.
                    
                    for nb in range(num_blocks):
                        if nb in current_set:
                            continue
                        
                        n_mask = block_masks[nb]
                        if not (n_mask & (1 << first_ex)):
                            continue
                        
                        # Check all exclusive bits
                        ok = True
                        for ex in exclusive_bits:
                            if not (n_mask & (1 << ex)):
                                ok = False
                                break
                        if not ok:
                            continue
                        
                        # It's a valid candidate.
                        # Score: How many "critical" bits (count==1) does it cover that are NOT in current block?
                        # Or better: How much does it increase the "minimum count" of the system?
                        # Simple score: Number of bits where count is 1 that it covers.
                        # If it covers many count-1 bits, it makes the solution more robust.
                        
                        score = 0
                        nm = n_mask
                        while nm:
                            lsb = nm & -nm
                            idx = lsb.bit_length() - 1
                            if counts[idx] == 1:
                                score += 1
                            nm ^= lsb
                        
                        if score > best_score:
                            best_score = score
                            best_candidate = nb
                            
                    if best_candidate != -1:
                        # Perform swap
                        # Remove old
                        current_list.pop(i)
                        m = mask
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            counts[idx] -= 1
                            m ^= lsb
                        
                        # Add new
                        current_list.append(best_candidate)
                        nm = block_masks[best_candidate]
                        while nm:
                            lsb = nm & -nm
                            idx = lsb.bit_length() - 1
                            counts[idx] += 1
                            nm ^= lsb
                        
                        return current_list, counts, True
                        
            return current_list, counts, False

        # Run the loop
        while time.time() < deadline - 2:
            current, counts, improved = try_remove_and_repair(current, counts)
            if not improved:
                # If we couldn't improve in a full pass, break?
                # The function returns immediately on improvement.
                # If it returns False, it means it tried a swap and failed?
                # My `try_remove_and_repair` logic:
                # 1. Checks all for removal. If found, removes and returns True.
                # 2. If no removal, tries ONE random swap. If successful, returns True.
                # 3. If swap fails, returns False.
                
                # So if it returns False, we should probably try again?
                # Or break if we are stuck?
                # Randomization might help. Let's just keep going until time runs out or we decide to break.
                # But if we are stuck, we might loop forever doing nothing.
                # Let's add a counter for failed attempts.
                
                # Actually, the function returns immediately after one action.
                # So `improved=False` means "I tried a swap and it failed".
                # We should probably try another swap.
                # So we just loop.
                pass

        return current

    best_blocks = None
    best_size = float('inf')
    
    # Run multiple restarts
    # Estimate time per restart:
    # Greedy: ~1-2s
    # Repair: ~1-5s
    # Total ~5s.
    # We have 145s. So ~20-25 restarts.
    
    for seed in range(100):
        if time.time() > deadline - 10:
            break
        
        blocks = greedy_solve(seed)
        
        # Quick check: is it valid?
        # Greedy ensures it is.
        
        size = len(blocks)
        if size < best_size:
            best_size = size
            best_blocks = list(blocks)
            print(f"Seed {seed}: Greedy size {size}", file=sys.stderr)
            
            # Try to improve
            improved_blocks = repair_and_optimize(blocks, deadline - 2)
            imp_size = len(improved_blocks)
            
            if imp_size < best_size:
                best_size = imp_size
                best_blocks = improved_blocks
                print(f"Seed {seed}: Improved size {imp_size}", file=sys.stderr)
            
            # If we found a very good one (e.g., < 340), spend more time on it?
            # For now, just move to next restart.

    if not best_blocks:
        # Fallback
        best_blocks = greedy_solve(0)
        
    final_blocks = [list(all_blocks[bi]) for bi in best_blocks]
    return final_blocks
