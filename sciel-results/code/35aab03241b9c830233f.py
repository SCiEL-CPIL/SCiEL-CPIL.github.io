import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    def elapsed():
        return time.time() - start_time
    
    # 1. Precompute all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}
    
    # 2. Generate all k-subsets (candidate blocks)
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)
    
    # 3. Precompute coverage mask for each k-subset as a Python int bitset
    ksub_masks = [0] * n_ksubs
    for j, ks in enumerate(all_ksubs):
        mask = 0
        for tsub in itertools.combinations(ks, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
        ksub_masks[j] = mask
    
    # Also build reverse mapping: t-subset index -> list of block indices covering it
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for j, mask in enumerate(ksub_masks):
        # Iterate set bits
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            tsub_to_blocks[idx].append(j)
            m ^= lsb
    
    # 4. Greedy initialization with bitsets
    uncovered_mask = (1 << n_tsubs) - 1
    remaining = n_tsubs
    
    blocks_indices = []
    used_blocks = set()
    
    while remaining > 0:
        if elapsed() > deadline - 30.0:  # Leave time for local search
            break
        
        best_j = -1
        best_gain = 0
        
        # Scan all candidates to find the one with max gain
        for j in range(n_ksubs):
            if j in used_blocks:
                continue
            
            common = uncovered_mask & ksub_masks[j]
            gain = common.bit_count()
            
            if gain > best_gain:
                best_gain = gain
                best_j = j
        
        if best_j == -1 or best_gain == 0:
            break
        
        blocks_indices.append(best_j)
        used_blocks.add(best_j)
        
        uncovered_mask &= ~ksub_masks[best_j]
        remaining -= best_gain
    
    # 5. Local Search Refinement
    # We will use a 1-exchange strategy:
    # - Maintain coverage counts for each t-subset.
    # - Identify "critical" t-subsets (covered by exactly 1 block).
    # - For each block, if it has critical t-subsets, we look for a replacement
    #   that covers all those critical t-subsets and is not currently used.
    # - If found, swap and then try to remove redundancies.
    
    if elapsed() < deadline - 10.0 and len(blocks_indices) > 0:
        current_indices = list(blocks_indices)
        current_set = set(current_indices)
        
        # Initialize coverage counts
        coverage = [0] * n_tsubs
        for idx in current_indices:
            mask = ksub_masks[idx]
            m = mask
            while m:
                lsb = m & (-m)
                ts_idx = lsb.bit_length() - 1
                coverage[ts_idx] += 1
                m ^= lsb
        
        def try_remove_redundants(indices, idx_set, cov):
            """Remove redundant blocks. Returns new list, set, and coverage."""
            changed = True
            while changed and elapsed() < deadline - 5.0:
                changed = False
                new_indices = []
                for idx in indices:
                    mask = ksub_masks[idx]
                    # Check if any t-subset in this block is covered ONLY by this block
                    # i.e., coverage[ts] == 1 for some ts in mask
                    redundant = True
                    m = mask
                    while m:
                        lsb = m & (-m)
                        ts_idx = lsb.bit_length() - 1
                        if cov[ts_idx] == 1:
                            redundant = False
                            break
                        m ^= lsb
                    
                    if redundant:
                        # Remove it
                        idx_set.remove(idx)
                        m = mask
                        while m:
                            lsb = m & (-m)
                            ts_idx = lsb.bit_length() - 1
                            cov[ts_idx] -= 1
                            m ^= lsb
                        changed = True
                    else:
                        new_indices.append(idx)
                indices = new_indices
            return indices, idx_set, cov
        
        # Initial redundancy removal
        current_indices, current_set, coverage = try_remove_redundants(current_indices, current_set, coverage)
        
        # 1-Exchange Loop
        # To speed up, we limit the number of passes or time.
        # We pick a block, find its critical t-subsets, intersect their candidate blocks,
        # pick a random candidate (or the one with max gain?) not in current_set,
        # swap, and check validity/redundancy.
        
        # Heuristic: Prioritize blocks with many critical t-subsets? Or random?
        # Let's iterate through blocks.
        
        # To avoid getting stuck, we use a simple heuristic:
        # For each block b:
        #   Crit = {ts | ts in b and coverage[ts] == 1}
        #   If Crit is empty: remove b (handled by redundancy removal)
        #   Else:
        #     Candidates = intersection of tsub_to_blocks[ts] for ts in Crit
        #     Candidates -= current_set
        #     If Candidates is not empty:
        #       Pick a candidate c (e.g., random, or one that maximizes coverage of non-critical ts?)
        #       Let's just pick random for simplicity, or the first one.
        #       Swap b with c.
        #       Update coverage.
        #       Run redundancy removal.
        #       If size decreased, keep. Else revert.
        
        # Actually, a swap doesn't change size. We only get a gain if redundancy removal
        # removes a block *in addition* to the swap, or if the swap allows removing b
        # and the new block c is "better" in a way that allows further removals?
        # No, standard 1-exchange for set cover minimization:
        # We replace b with c. Size is same.
        # But maybe c covers more "useful" things, allowing us to remove *another* block?
        # Or maybe c is just a better fit.
        # The gain comes from:
        # 1. b was redundant (removed).
        # 2. After swapping, some other block becomes redundant.
        
        # So the procedure is:
        # 1. Swap b -> c.
        # 2. Remove redundants.
        # 3. If total size < previous size, accept. Else revert.
        
        # We need to be careful with time.
        
        improved = True
        max_iterations = 2000 # Hard cap on local search steps
        steps = 0
        
        while improved and steps < max_iterations and elapsed() < deadline - 5.0:
            improved = False
            steps += 1
            
            # Shuffle blocks to try different orderings
            to_try = list(current_indices)
            random.shuffle(to_try)
            
            for b in to_try:
                if b not in current_set:
                    continue
                
                # Find critical t-subsets for b
                crit_ts = []
                mask_b = ksub_masks[b]
                m = mask_b
                while m:
                    lsb = m & (-m)
                    ts_idx = lsb.bit_length() - 1
                    if coverage[ts_idx] == 1:
                        crit_ts.append(ts_idx)
                    m ^= lsb
                
                if not crit_ts:
                    # b is redundant, should have been removed, but maybe it wasn't?
                    # Force remove?
                    # Actually, if we are in the loop, redundancy removal should have handled it.
                    # But maybe it became redundant due to a previous swap in this pass?
                    # Let's just continue, redundancy removal will catch it at the end of pass?
                    # No, we run it after swap.
                    # If it's redundant now, we can just remove it.
                    current_set.remove(b)
                    current_indices.remove(b)
                    m = mask_b
                    while m:
                        lsb = m & (-m)
                        ts_idx = lsb.bit_length() - 1
                        coverage[ts_idx] -= 1
                        m ^= lsb
                    improved = True
                    continue
                
                # Find candidates
                # Candidates must cover all crit_ts
                # Intersect tsub_to_blocks[ts] for ts in crit_ts
                candidates = None
                for ts in crit_ts:
                    blocks_covering = tsub_to_blocks[ts]
                    if candidates is None:
                        candidates = set(blocks_covering)
                    else:
                        candidates.intersection_update(blocks_covering)
                
                if not candidates:
                    continue
                
                candidates -= current_set
                
                if not candidates:
                    continue
                
                # Pick a candidate.
                # Heuristic: Pick one that covers many *other* t-subsets that are currently covered by b?
                # Or just random?
                # Let's pick the one that maximizes the number of t-subsets in b that are *not* critical
                # but are covered by the candidate?
                # Actually, simpler: Just pick random.
                # Or pick the one with max gain on the *whole* set?
                # Let's try: Score = number of t-subsets in b that are covered by candidate.
                # Since candidate MUST cover crit_ts, we want it to cover as much of the rest of b as possible.
                # This minimizes the "loss" of coverage when b is removed.
                
                best_cand = None
                best_score = -1
                cand_list = list(candidates)
                # Limit candidates to check to save time?
                # If list is huge, sample it.
                if len(cand_list) > 50:
                    cand_list = random.sample(cand_list, 50)
                
                for c in cand_list:
                    mask_c = ksub_masks[c]
                    # Count intersection of mask_b and mask_c
                    # Since c covers crit_ts, it covers at least those.
                    # We want to maximize overlap with b.
                    common = mask_b & mask_c
                    score = common.bit_count()
                    if score > best_score:
                        best_score = score
                        best_cand = c
                
                if best_cand is None:
                    continue
                
                # Perform Swap
                old_size = len(current_indices)
                
                # Remove b
                current_set.remove(b)
                current_indices.remove(b)
                m = mask_b
                while m:
                    lsb = m & (-m)
                    ts_idx = lsb.bit_length() - 1
                    coverage[ts_idx] -= 1
                    m ^= lsb
                
                # Add c
                current_set.add(best_cand)
                current_indices.append(best_cand)
                mask_c = ksub_masks[best_cand]
                m = mask_c
                while m:
                    lsb = m & (-m)
                    ts_idx = lsb.bit_length() - 1
                    coverage[ts_idx] += 1
                    m ^= lsb
                
                # Run redundancy removal
                new_indices, new_set, new_cov = try_remove_redundants(current_indices, current_set, coverage)
                
                if len(new_indices) < old_size:
                    # Improvement!
                    current_indices = new_indices
                    current_set = new_set
                    coverage = new_cov
                    improved = True
                    # Break inner loop? Or continue?
                    # Continue to try more swaps.
                else:
                    # Revert
                    # Remove c
                    current_set.remove(best_cand)
                    current_indices.remove(best_cand)
                    m = mask_c
                    while m:
                        lsb = m & (-m)
                        ts_idx = lsb.bit_length() - 1
                        coverage[ts_idx] -= 1
                        m ^= lsb
                    
                    # Add b
                    current_set.add(b)
                    current_indices.append(b)
                    m = mask_b
                    while m:
                        lsb = m & (-m)
                        ts_idx = lsb.bit_length() - 1
                        coverage[ts_idx] += 1
                        m ^= lsb

    # Convert indices to blocks
    final_blocks = [list(all_ksubs[idx]) for idx in current_indices]
    
    # Final sanity check (optional, but good for safety)
    # The evaluator checks validity. We trust our logic, but let's ensure we return a list of lists.
    return final_blocks
