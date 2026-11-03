import itertools
import time
import random
import numpy as np
import heapq
from collections import Counter

def construct(v, k, t):
    """
    Finds a covering design C(v, k, t) using heap-based greedy + enhanced local search.
    v=19, k=7, t=4
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    # Precompute all t-subsets and their indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute for each block, the list of t-subset indices it covers
    block_tsubs = []
    for block in all_blocks:
        tsubs = []
        for ts in itertools.combinations(block, t):
            tsubs.append(tsub_to_idx[ts])
        block_tsubs.append(np.array(tsubs, dtype=np.int32))
    
    # Precompute tsub_to_blocks: list of block indices for each t-subset
    tsub_to_blocks = [[] for _ in range(n_tsubsets)]
    for bi, tsubs in enumerate(block_tsubs):
        for ts_idx in tsubs:
            tsub_to_blocks[ts_idx].append(bi)
    
    # Convert to sets for faster membership testing in swap validation
    # Actually, for swap validation we check if ts_idx is in block_tsubs[bi_new]
    # Let's precompute a set version for each block's tsubsets for O(1) lookup
    block_tsub_sets = [set(tsubs.tolist()) for tsubs in block_tsubs]
    
    # --- Heap-based Greedy Construction ---
    def greedy_construct_heap(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        blocks_used = []
        
        # gain[bi] = number of uncovered t-subsets in block bi
        gain = np.zeros(n_blocks, dtype=np.int32)
        gain.fill(len(block_tsubs[0]))  # C(k,t) for all blocks initially
        
        # Heap: (negative_gain, block_index) for max-heap behavior
        # We'll use lazy deletion: push updated gains, skip stale entries
        heap = []
        for bi in range(n_blocks):
            heapq.heappush(heap, (-gain[bi], bi))
        
        n_uncovered = n_tsubsets
        
        while n_uncovered > 0:
            if time.time() > deadline - 80:
                break
                
            # Pop until we find a valid (non-stale) entry
            chosen_idx = None
            while heap:
                neg_g, bi = heapq.heappop(heap)
                g = -neg_g
                if g == gain[bi] and g > 0:
                    chosen_idx = bi
                    break
                # If gain[bi] changed, this entry is stale; discard
                elif g > gain[bi]:
                    # Stale: old gain was higher than current
                    continue
                elif g < gain[bi]:
                    # This shouldn't happen if we always push new gains
                    continue
                else:
                    # g == gain[bi] but gain[bi] == 0, skip
                    continue
            
            if chosen_idx is None:
                break
                
            blocks_used.append(chosen_idx)
            
            # Update uncovered and gains
            tsubs = block_tsubs[chosen_idx]
            newly_covered = tsubs[uncovered[tsubs]]
            
            if len(newly_covered) > 0:
                n_uncovered -= len(newly_covered)
                uncovered[newly_covered] = False
                
                # Decrement gain for all blocks that cover any of the newly_covered t-subsets
                affected_blocks = []
                for ts_idx in newly_covered:
                    affected_blocks.extend(tsub_to_blocks[ts_idx])
                
                if affected_blocks:
                    # Use Counter for speed on small lists
                    cnt = Counter(affected_blocks)
                    # Mark chosen block first so it won't be re-pushed
                    gain[chosen_idx] = 0
                    for bi, c in cnt.items():
                        if bi == chosen_idx:
                            continue
                        gain[bi] -= c
                        if gain[bi] > 0:
                            heapq.heappush(heap, (-gain[bi], bi))
                    # Re-push chosen with 0 is unnecessary; it stays 0
                else:
                    gain[chosen_idx] = 0
            
            # Ensure chosen is 0 (in case affected was empty)
            if gain[chosen_idx] != 0:
                gain[chosen_idx] = 0
            
        return blocks_used
    
    # Get initial solution with multiple seeds, keep the best
    best_initial = None
    for seed in [123, 456, 789, 101, 202]:
        if time.time() > deadline - 75:
            break
        init = greedy_construct_heap(seed)
        if init and (best_initial is None or len(init) < len(best_initial)):
            best_initial = init
    
    initial_blocks = best_initial
    if not initial_blocks:
        return [list(b) for b in all_blocks[:100]]
    
    # --- Enhanced Local Search ---
    blocks = list(initial_blocks)
    cover_count = np.zeros(n_tsubsets, dtype=np.int32)
    
    for bi in blocks:
        for ts_idx in block_tsubs[bi]:
            cover_count[ts_idx] += 1
    
    current_cost = len(blocks)
    best_cost = current_cost
    best_blocks = blocks.copy()
    
    # Precompute for each block, the set of t-subsets it covers (for fast membership)
    # Already have block_tsub_sets
    
    # Helper: get unique t-subsets for a block (those with cover_count == 1)
    def get_unique_tsubs(bi):
        tsubs = block_tsubs[bi]
        return [ts_idx for ts_idx in tsubs if cover_count[ts_idx] == 1]
    
    # SA Parameters
    T = 50.0
    T_min = 0.5
    alpha = 0.9995
    
    rng = random.Random(42)
    
    # Track best swap candidates: for each block, which other blocks could replace it
    # Precompute candidate pools for swap moves
    # For a block bi_old with unique t-subsets U, bi_new must cover all of U
    # We can precompute for each subset of t-subsets, which blocks cover them
    # But that's too expensive. Instead, use random sampling with validation.
    
    iteration = 0
    no_improve_count = 0
    max_no_improve = 5000
    
    while T > T_min and time.time() < deadline - 10:
        iteration += 1
        
        if time.time() > deadline - 10:
            break
            
        move_type = rng.random()
        
        if move_type < 0.25:
            # Try to remove a redundant block
            if not blocks:
                continue
                
            i = rng.randint(0, len(blocks) - 1)
            bi = blocks[i]
            
            # Check if redundant
            redundant = True
            for ts_idx in block_tsubs[bi]:
                if cover_count[ts_idx] == 1:
                    redundant = False
                    break
                    
            if redundant:
                blocks.pop(i)
                for ts_idx in block_tsubs[bi]:
                    cover_count[ts_idx] -= 1
                current_cost -= 1
                no_improve_count = 0
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = blocks.copy()
            else:
                no_improve_count += 1
                
        elif move_type < 0.55:
            # 2-for-1 swap: replace two blocks with one that covers their union of unique tsubs
            if len(blocks) < 2:
                continue
            i1 = rng.randint(0, len(blocks) - 1)
            i2 = rng.randint(0, len(blocks) - 1)
            if i2 == i1:
                i2 = (i2 + 1) % len(blocks)
            bi1 = blocks[i1]
            bi2 = blocks[i2]
            # Unique tsubs of each
            u1 = [ts for ts in block_tsubs[bi1] if cover_count[ts] == 1]
            u2 = [ts for ts in block_tsubs[bi2] if cover_count[ts] == 1]
            if not u1 and not u2:
                # Both redundant; remove one
                blocks.pop(i2)
                for ts in block_tsubs[bi2]:
                    cover_count[ts] -= 1
                current_cost -= 1
                no_improve_count = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = blocks.copy()
                continue
            # Union of unique tsubs that must be covered by replacement
            union_unique = list(set(u1) | set(u2))
            if not union_unique:
                continue
            # Find a block covering all union_unique, not equal to bi1 or bi2
            first = union_unique[0]
            cands = tsub_to_blocks[first]
            if len(cands) > 30:
                cands = rng.sample(cands, 30)
            else:
                cands = list(cands)
            rng.shuffle(cands)
            found = None
            for bn in cands:
                if bn == bi1 or bn == bi2:
                    continue
                bset = block_tsub_sets[bn]
                if all(ts in bset for ts in union_unique):
                    found = bn
                    break
            if found is None:
                no_improve_count += 1
                continue
            # Perform: remove bi1 and bi2, add found
            # Remove the larger index first to keep indices valid
            if i1 > i2:
                i1, i2 = i2, i1
                bi1, bi2 = bi2, bi1
            # Remove bi2 (at i2)
            for ts in block_tsubs[bi2]:
                cover_count[ts] -= 1
            blocks.pop(i2)
            # Remove bi1 (now at i1 if i1 < i2, else i1-1)
            if i1 < i2:
                idx1 = i1
            else:
                idx1 = i1 - 1
            for ts in block_tsubs[bi1]:
                cover_count[ts] -= 1
            blocks.pop(idx1)
            # Add found
            blocks.append(found)
            for ts in block_tsubs[found]:
                cover_count[ts] += 1
            current_cost -= 1  # net: -2 + 1 = -1
            no_improve_count = 0
            if current_cost < best_cost:
                best_cost = current_cost
                best_blocks = blocks.copy()
                
        elif move_type < 0.85:
            # Try a swap: replace bi_old with bi_new
            if not blocks:
                continue
                
            i = rng.randint(0, len(blocks) - 1)
            bi_old = blocks[i]
            
            unique_tsubs = get_unique_tsubs(bi_old)
            
            if not unique_tsubs:
                # Redundant block; try to remove it instead
                blocks.pop(i)
                for ts_idx in block_tsubs[bi_old]:
                    cover_count[ts_idx] -= 1
                current_cost -= 1
                no_improve_count = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = blocks.copy()
                continue
            
            # Find candidate blocks that cover all unique_tsubs
            # Strategy: sample random blocks and check
            # Better: for the first unique t-subset, get all blocks covering it,
            # then filter for those covering the rest
            first_ts = unique_tsubs[0]
            candidates = tsub_to_blocks[first_ts]
            
            if not candidates:
                continue
            
            # Shuffle and try up to 20 candidates
            rng.shuffle(candidates)
            swapped = False
            for bi_new in candidates[:20]:
                if bi_new == bi_old:
                    continue
                    
                # Check if bi_new covers all unique_tsubs
                valid = True
                bts_set = block_tsub_sets[bi_new]
                for ts_idx in unique_tsubs:
                    if ts_idx not in bts_set:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Perform swap
                for ts_idx in block_tsubs[bi_old]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_tsubs[bi_new]:
                    cover_count[ts_idx] += 1
                    
                blocks[i] = bi_new
                swapped = True
                break
            
            if not swapped:
                no_improve_count += 1
            else:
                # Swap succeeded; check if it enables removals
                # Try to remove redundant blocks after swap
                for j in range(len(blocks)):
                    if time.time() > deadline - 10:
                        break
                    bj = blocks[j]
                    redundant = True
                    for ts_idx in block_tsubs[bj]:
                        if cover_count[ts_idx] == 1:
                            redundant = False
                            break
                    if redundant:
                        blocks.pop(j)
                        for ts_idx in block_tsubs[bj]:
                            cover_count[ts_idx] -= 1
                        current_cost -= 1
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_blocks = blocks.copy()
                        no_improve_count = 0
                        break  # Restart the removal scan
        
        else:
            # Add a random block (only when hot)
            if T < 5.0:
                no_improve_count += 1
                continue
                
            bi_new = rng.randint(0, n_blocks - 1)
            if bi_new in blocks:
                continue
                
            blocks.append(bi_new)
            for ts_idx in block_tsubs[bi_new]:
                cover_count[ts_idx] += 1
            current_cost += 1
            no_improve_count += 1
        
        # Update temperature
        T *= alpha
        
        # If stuck too long, do a small restart: perturb the solution
        if no_improve_count > max_no_improve and T > T_min * 10:
            # Remove 3-5 random non-essential blocks and re-add
            n_remove = min(3, len(blocks) // 10)
            for _ in range(n_remove):
                if not blocks:
                    break
                i = rng.randint(0, len(blocks) - 1)
                bi = blocks[i]
                # Only remove if not critical
                critical = False
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        critical = True
                        break
                if not critical:
                    blocks.pop(i)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    current_cost -= 1
            no_improve_count = 0
        
        # Early termination if we've reached a very good solution
        if best_cost <= 135:
            break
    
    # Final Polish: aggressive reduction
    final_blocks = best_blocks.copy()
    cover_count_final = np.zeros(n_tsubsets, dtype=np.int32)
    for bi in final_blocks:
        for ts_idx in block_tsubs[bi]:
            cover_count_final[ts_idx] += 1
    
    # Multiple passes of removal
    for _ in range(3):
        i = 0
        while i < len(final_blocks):
            bi = final_blocks[i]
            redundant = True
            for ts_idx in block_tsubs[bi]:
                if cover_count_final[ts_idx] == 1:
                    redundant = False
                    break
            if redundant:
                final_blocks.pop(i)
                for ts_idx in block_tsubs[bi]:
                    cover_count_final[ts_idx] -= 1
            else:
                i += 1
    
    # 2-for-1 polish passes
    for _ in range(200):
        if time.time() > deadline - 3:
            break
        if len(final_blocks) < 2:
            break
        i1 = rng.randint(0, len(final_blocks) - 1)
        i2 = rng.randint(0, len(final_blocks) - 1)
        if i2 == i1:
            i2 = (i2 + 1) % len(final_blocks)
        bi1 = final_blocks[i1]
        bi2 = final_blocks[i2]
        u1 = [ts for ts in block_tsubs[bi1] if cover_count_final[ts] == 1]
        u2 = [ts for ts in block_tsubs[bi2] if cover_count_final[ts] == 1]
        if not u1 and not u2:
            # Remove one redundant
            final_blocks.pop(i2)
            for ts in block_tsubs[bi2]:
                cover_count_final[ts] -= 1
            continue
        union_unique = list(set(u1) | set(u2))
        if not union_unique:
            continue
        first = union_unique[0]
        cands = tsub_to_blocks[first]
        if len(cands) > 20:
            cands = rng.sample(cands, 20)
        else:
            cands = list(cands)
        rng.shuffle(cands)
        found = None
        for bn in cands:
            if bn == bi1 or bn == bi2:
                continue
            bset = block_tsub_sets[bn]
            if all(ts in bset for ts in union_unique):
                found = bn
                break
        if found is None:
            continue
        if i1 > i2:
            i1, i2 = i2, i1
            bi1, bi2 = bi2, bi1
        for ts in block_tsubs[bi2]:
            cover_count_final[ts] -= 1
        final_blocks.pop(i2)
        idx1 = i1 if i1 < i2 else i1 - 1
        for ts in block_tsubs[bi1]:
            cover_count_final[ts] -= 1
        final_blocks.pop(idx1)
        final_blocks.append(found)
        for ts in block_tsubs[found]:
            cover_count_final[ts] += 1
        # Removal pass after 2-for-1
        j = 0
        while j < len(final_blocks):
            bj = final_blocks[j]
            redundant = all(cover_count_final[ts] > 1 for ts in block_tsubs[bj])
            if redundant:
                final_blocks.pop(j)
                for ts in block_tsubs[bj]:
                    cover_count_final[ts] -= 1
            else:
                j += 1
    
    # Try swap-based reduction on final
    for _ in range(500):
        if time.time() > deadline - 2:
            break
        if not final_blocks:
            break
        i = rng.randint(0, len(final_blocks) - 1)
        bi_old = final_blocks[i]
        unique_tsubs = [ts_idx for ts_idx in block_tsubs[bi_old] if cover_count_final[ts_idx] == 1]
        if not unique_tsubs:
            final_blocks.pop(i)
            for ts_idx in block_tsubs[bi_old]:
                cover_count_final[ts_idx] -= 1
            continue
        first_ts = unique_tsubs[0]
        candidates = tsub_to_blocks[first_ts]
        if not candidates:
            continue
        rng.shuffle(candidates)
        swapped = False
        for bi_new in candidates[:10]:
            if bi_new == bi_old:
                continue
            bts_set = block_tsub_sets[bi_new]
            valid = all(ts_idx in bts_set for ts_idx in unique_tsubs)
            if not valid:
                continue
            for ts_idx in block_tsubs[bi_old]:
                cover_count_final[ts_idx] -= 1
            for ts_idx in block_tsubs[bi_new]:
                cover_count_final[ts_idx] += 1
            final_blocks[i] = bi_new
            swapped = True
            break
        if swapped:
            # Try removal again
            j = 0
            while j < len(final_blocks):
                bj = final_blocks[j]
                redundant = all(cover_count_final[ts_idx] > 1 for ts_idx in block_tsubs[bj])
                if redundant:
                    final_blocks.pop(j)
                    for ts_idx in block_tsubs[bj]:
                        cover_count_final[ts_idx] -= 1
                else:
                    j += 1
    
    if len(final_blocks) < best_cost:
        best_blocks = final_blocks
        best_cost = len(final_blocks)
    
    return [list(all_blocks[bi]) for bi in best_blocks]
