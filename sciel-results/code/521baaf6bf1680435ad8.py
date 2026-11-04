import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin

    # Pre-computation
    # 1. Map every t-subset to an index
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # 2. Generate all possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # 3. Map each block to the list of triple indices it covers
    block_triples = []
    for b in all_blocks:
        tris = [triple_to_idx[tri] for tri in itertools.combinations(b, t)]
        block_triples.append(tris)
        
    # 4. Map each triple to the list of block indices that cover it
    triple_to_blocks = defaultdict(list)
    for b_idx, tris in enumerate(block_triples):
        for tri_idx in tris:
            triple_to_blocks[tri_idx].append(b_idx)

    def greedy_construct():
        uncovered = set(range(n_triples))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triple
            tri_idx = random.choice(list(uncovered))
            
            # Get all blocks covering this triple
            candidates = triple_to_blocks[tri_idx]
            
            best_block = None
            best_gain = -1
            
            for b_idx in candidates:
                gain = 0
                for tri in block_triples[b_idx]:
                    if tri in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    if gain == len(block_triples[b_idx]):
                        break
            
            if best_block is not None:
                blocks.append(best_block)
                for tri in block_triples[best_block]:
                    if tri in uncovered:
                        uncovered.remove(tri)
            else:
                break
                
        return blocks

    def optimize_blocks(blocks):
        if not blocks:
            return []
            
        # counts[tri_idx] = number of blocks in `blocks` covering tri_idx
        counts = [0] * n_triples
        for b_idx in blocks:
            for tri in block_triples[b_idx]:
                counts[tri] += 1
                
        active = [True] * len(blocks)
        
        def is_redundant(i):
            if not active[i]:
                return False
            b_idx = blocks[i]
            for tri in block_triples[b_idx]:
                if counts[tri] <= 1:
                    return False
            return True

        def remove_block(i):
            if not active[i]:
                return
            active[i] = False
            b_idx = blocks[i]
            for tri in block_triples[b_idx]:
                counts[tri] -= 1

        def add_block(b_idx):
            blocks.append(b_idx)
            i_in = len(blocks) - 1
            active.append(True)
            for tri in block_triples[b_idx]:
                counts[tri] += 1
            return i_in

        # Phase 1: Greedy Redundancy Removal
        changed = True
        while changed:
            changed = False
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            for i in indices:
                if not active[i]:
                    continue
                if is_redundant(i):
                    remove_block(i)
                    changed = True

        # Phase 2: Local Search with 1-for-1 Swaps and 2-for-1 Swaps
        # We run this until deadline
        while time.time() < deadline:
            pass_found = False
            
            # 2.1 Try 2-for-1 Swaps (Remove A, B, Add C)
            # Sample a few pairs of active blocks
            active_indices = [i for i in range(len(blocks)) if active[i]]
            if len(active_indices) >= 2:
                # Limit attempts to keep it fast
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    i1, i2 = random.sample(active_indices, 2)
                    b1 = blocks[i1]
                    b2 = blocks[i2]
                    
                    # Temporarily remove b1 and b2
                    # We need to track which triples become uncovered (count goes to 0)
                    # And which are just reduced (count > 0)
                    # Actually, for 2-for-1, we remove both. 
                    # Any triple that was covered ONLY by b1 or b2 (or both, but count was 1 or 2) 
                    # might become uncovered.
                    
                    # Efficient check:
                    # 1. Identify triples that will become uncovered (count becomes 0)
                    # 2. Check if a single block covers all of them.
                    
                    # Let's simulate removal
                    # We need to know counts before removal.
                    # Let's collect the "critical" triples.
                    critical_triples = []
                    
                    # Triples in b1
                    for tri in block_triples[b1]:
                        if counts[tri] == 1:
                            # Only covered by b1 (since b2 is different, if b2 also covered it, count >= 2)
                            # Wait, if b2 also covers tri, count is at least 2.
                            # So if count is 1, it's unique to b1.
                            critical_triples.append(tri)
                        elif counts[tri] == 2:
                            # Covered by b1 and one other. If that other is b2, it becomes 0.
                            # We need to check if b2 covers it.
                            if b2 in triple_to_blocks[tri]:
                                critical_triples.append(tri)
                    
                    # Triples in b2
                    for tri in block_triples[b2]:
                        if counts[tri] == 1:
                            critical_triples.append(tri)
                        elif counts[tri] == 2:
                            if b1 in triple_to_blocks[tri]:
                                critical_triples.append(tri)
                        
                    # Remove duplicates in critical_triples?
                    # It's possible a triple is in both b1 and b2 and count was 2.
                    # It will appear twice. We should unique it.
                    critical_set = set(critical_triples)
                    
                    if not critical_set:
                        # Both blocks are redundant? Should have been caught in Phase 1.
                        # Or they are mutually redundant but not individually?
                        # If count is 2 for all, and they cover each other's unique parts?
                        # If critical_set is empty, removing both leaves all triples covered.
                        # This is a 2-for-0 swap! Even better.
                        remove_block(i1)
                        remove_block(i2)
                        pass_found = True
                        break # Restart loop with new state
                    
                    # Find a block C that covers all triples in critical_set
                    # Start with blocks covering the first critical triple
                    first_tri = next(iter(critical_set))
                    candidates = triple_to_blocks[first_tri]
                    
                    valid_c = None
                    for c_idx in candidates:
                        # Skip if c is b1 or b2
                        if c_idx == b1 or c_idx == b2:
                            continue
                        # Check if c is already in active solution?
                        # If c is already in solution, adding it does nothing (count increases, but we are removing b1,b2).
                        # Wait, if c is in solution, it's already counted in `counts`.
                        # Our `critical_set` logic assumed we remove b1,b2.
                        # If c is in solution, it contributes to counts.
                        # So if c covers a critical triple, that triple's count would be >= 1 (from c) + (from b1/b2).
                        # If count was 1, it couldn't be covered by c (else count >= 2).
                        # If count was 2, and covered by b1 and c, then removing b1 leaves c. Count becomes 1. Not critical (uncovered).
                        # So if a triple is in critical_set, it is NOT covered by any other active block.
                        # Therefore, any valid replacement C must NOT be an existing active block (unless it's b1 or b2 which we are removing, but we skip those).
                        # So we just need to find a block C (not b1, not b2) that covers all critical_set.
                        
                        # Optimization: Check if c covers all critical_set
                        # critical_set size is small usually.
                        c_tris = set(block_triples[c_idx])
                        if c_tris.issuperset(critical_set):
                            valid_c = c_idx
                            break
                    
                    if valid_c is not None:
                        # Perform swap
                        remove_block(i1)
                        remove_block(i2)
                        add_block(valid_c)
                        pass_found = True
                        break # Restart loop

            if pass_found:
                # After a 2-for-1, run redundancy removal again
                changed = True
                while changed:
                    changed = False
                    indices = list(range(len(blocks)))
                    random.shuffle(indices)
                    for i in indices:
                        if not active[i]:
                            continue
                        if is_redundant(i):
                            remove_block(i)
                            changed = True
                continue

            # 2.2 Try 1-for-1 Swaps
            # Pick a block, try to replace it with another that maintains validity.
            # Accept if it reduces "potential redundancy" or just random walk to escape local optima?
            # Since 1-for-1 keeps count same, we use it to restructure.
            # Heuristic: Accept if the new block covers more triples that are "low count" (closer to 1).
            
            active_indices = [i for i in range(len(blocks)) if active[i]]
            if not active_indices:
                break
                
            i_out = random.choice(active_indices)
            b_out = blocks[i_out]
            
            # Temporarily remove b_out
            active[i_out] = False
            for tri in block_triples[b_out]:
                counts[tri] -= 1
                
            uncovered_tris = []
            for tri in block_triples[b_out]:
                if counts[tri] == 0:
                    uncovered_tris.append(tri)
            
            if not uncovered_tris:
                # b_out was redundant. Restore and let redundancy removal handle it? 
                # Or just keep it removed?
                # If we keep it removed, count decreases. 
                # But we are in swap phase. Let's just restore and let next iteration's redundancy check catch it.
                active[i_out] = True
                for tri in block_triples[b_out]:
                    counts[tri] += 1
                continue
            
            # Find candidate b_in
            # Must cover all uncovered_tris
            candidates = set(triple_to_blocks[uncovered_tris[0]])
            for tri in uncovered_tris[1:]:
                candidates.intersection_update(triple_to_blocks[tri])
                if not candidates:
                    break
            
            if candidates:
                # Filter out b_out
                valid_candidates = [b for b in candidates if b != b_out]
                
                if valid_candidates:
                    # Score candidates by how many "extra" triples they cover that are currently count 1 or 0?
                    # Or just random pick to explore.
                    b_in = random.choice(valid_candidates)
                    
                    # Check if b_in is already in active solution?
                    # If so, adding it increases counts, but we are replacing b_out.
                    # Net effect: b_out removed, b_in count increased.
                    # This is a valid move.
                    
                    # Accept if it doesn't increase "waste"?
                    # Let's just accept it to allow structural change.
                    active[i_out] = True # Revert removal of b_out for now? 
                    # No, we want to swap.
                    # We already decremented counts for b_out.
                    # We need to increment for b_in.
                    
                    # But wait, if b_in is already in the solution (active), we just increment its count.
                    # We don't add a new block entry.
                    # If b_in is NOT in the solution, we add a new entry.
                    
                    # To keep list management simple, let's only swap with blocks NOT in the active set.
                    # If we allow swapping with existing blocks, we just change counts.
                    # Let's restrict to new blocks for now to keep `blocks` list growing/shrinking cleanly?
                    # No, `blocks` list is just a container.
                    
                    # Let's check if b_in is in active set.
                    # We need a set of active block indices for fast lookup.
                    active_set = set(blocks[i] for i in range(len(blocks)) if active[i])
                    
                    if b_in in active_set:
                        # Just update counts
                        for tri in block_triples[b_in]:
                            counts[tri] += 1
                        # b_out is already removed (counts decremented, active False)
                        pass_found = True
                    else:
                        # Add new block
                        add_block(b_in)
                        pass_found = True
                    
                    if pass_found:
                        # Run redundancy removal
                        changed = True
                        while changed:
                            changed = False
                            indices = list(range(len(blocks)))
                            random.shuffle(indices)
                            for i in indices:
                                if not active[i]:
                                    continue
                                if is_redundant(i):
                                    remove_block(i)
                                    changed = True
                        continue
                else:
                    # No valid swap, revert
                    active[i_out] = True
                    for tri in block_triples[b_out]:
                        counts[tri] += 1
            else:
                # Revert
                active[i_out] = True
                for tri in block_triples[b_out]:
                    counts[tri] += 1
            
            # If no swap found, break inner loop? 
            # We just do one swap attempt per outer loop iteration.
            # The outer loop continues until deadline.

        # Compact the blocks list
        final_blocks = [blocks[i] for i in range(len(blocks)) if active[i]]
        return final_blocks

    best_blocks = None
    best_len = float('inf')
    
    # Main Loop
    while time.time() < deadline:
        # Greedy Init
        cand = greedy_construct()
        
        # Optimize
        opt_cand = optimize_blocks(cand)
        
        if len(opt_cand) < best_len:
            best_len = len(opt_cand)
            best_blocks = opt_cand
            
        if best_len <= 69:
            break
            
    if best_blocks is None:
        # Fallback: return greedy if optimization failed or didn't run
        cand = greedy_construct()
        best_blocks = cand
        best_len = len(cand)

    # Convert block indices back to tuples of points
    return [list(all_blocks[b_idx]) for b_idx in best_blocks]
