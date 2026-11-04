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
            
            # Evaluate candidates
            # To speed up, we can sample if candidates list is huge?
            # Candidates for a triple is ~6000. 6000 * 35 operations is ~200k ops.
            # This is fine.
            for b_idx in candidates:
                gain = 0
                for tri in block_triples[b_idx]:
                    if tri in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    if gain == len(block_triples[b_idx]): # Perfect cover of all its triples
                        # Can break early? No, another block might have higher gain?
                        # Max gain is 35. If we hit 35, we can't do better.
                        break
            
            if best_block is not None:
                blocks.append(best_block)
                # Update uncovered
                for tri in block_triples[best_block]:
                    if tri in uncovered:
                        uncovered.remove(tri)
            else:
                # Should not happen
                break
                
        return blocks

    def optimize_blocks(blocks):
        # blocks is a list of block indices
        if not blocks:
            return []
            
        # counts[tri_idx] = number of blocks in `blocks` covering tri_idx
        counts = [0] * n_triples
        for b_idx in blocks:
            for tri in block_triples[b_idx]:
                counts[tri] += 1
                
        active = [True] * len(blocks)
        
        # Helper to check if block i is redundant
        def is_redundant(i):
            b_idx = blocks[i]
            for tri in block_triples[b_idx]:
                if counts[tri] <= 1:
                    return False
            return True

        # Helper to remove block i
        def remove_block(i):
            active[i] = False
            b_idx = blocks[i]
            for tri in block_triples[b_idx]:
                counts[tri] -= 1

        # Helper to add block b_idx (returns new index in blocks list? No, we modify list)
        # We will maintain `blocks` as a list of indices.
        
        changed = True
        while changed:
            changed = False
            # Try to remove redundant blocks
            # Shuffle order to handle dependencies
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            
            for i in indices:
                if not active[i]:
                    continue
                if is_redundant(i):
                    remove_block(i)
                    changed = True
                    
        # Now try swaps
        # We need to be careful with indices if we add/remove from `blocks` list.
        # Let's work with a set of block indices for the current solution?
        # No, list is fine if we don't remove from the middle, or we compact at the end.
        # For swap, we pick a block to remove, find a replacement, add it.
        
        # To allow adding new blocks, we should not fix the list size.
        # Let's represent the current solution as a list of block indices.
        
        # Re-initialize for swap phase
        # We need to rebuild counts if we modified `blocks`? 
        # The removal phase above modified `counts` and `active`.
        # Let's compact `blocks` and `counts`?
        # Actually, let's just use the `active` array and `blocks` list.
        # When we add a block, we append to `blocks` and set `active` to True.
        
        # Swap Loop
        no_improvement_count = 0
        
        while time.time() < deadline:
            # Try to find a redundant block again (in case swap created one)
            # We can just run the removal pass again
            pass_removal = True
            while pass_removal:
                pass_removal = False
                indices = list(range(len(blocks)))
                random.shuffle(indices)
                for i in indices:
                    if not active[i]:
                        continue
                    if is_redundant(i):
                        remove_block(i)
                        pass_removal = True
                        changed = True
                        
            if not changed:
                # Try a swap
                active_indices = [i for i in range(len(blocks)) if active[i]]
                if not active_indices:
                    break
                    
                i_out = random.choice(active_indices)
                b_out = blocks[i_out]
                
                # Temporarily remove b_out
                active[i_out] = False
                for tri in block_triples[b_out]:
                    counts[tri] -= 1
                    
                # Find uncovered triples
                uncovered_tris = []
                for tri_idx in range(n_triples):
                    if counts[tri_idx] == 0:
                        uncovered_tris.append(tri_idx)
                        
                # If too many uncovered, revert
                if len(uncovered_tris) > 10:
                    # Revert
                    active[i_out] = True
                    for tri in block_triples[b_out]:
                        counts[tri] += 1
                    no_improvement_count += 1
                    if no_improvement_count > 100:
                        break # Restart optimization
                    continue
                
                # Find candidate b_in that covers all uncovered_tris
                if not uncovered_tris:
                    # This means b_out was redundant, but we missed it?
                    # Or it became redundant.
                    # We already did removal pass. So this shouldn't happen.
                    # But if it does, we just keep it removed.
                    # Actually, if uncovered_tris is empty, we successfully removed b_out.
                    # We should have caught this in is_redundant.
                    # Let's assume it's handled.
                    pass
                else:
                    # Intersect block lists for each uncovered triple
                    # Start with the list for the first triple
                    candidates = set(triple_to_blocks[uncovered_tris[0]])
                    
                    for tri in uncovered_tris[1:]:
                        candidates.intersection_update(triple_to_blocks[tri])
                        if not candidates:
                            break
                    
                    if candidates:
                        # Filter out b_out and any block already in solution
                        current_block_set = set(blocks[i] for i in range(len(blocks)) if active[i])
                        valid_candidates = [b for b in candidates if b != b_out and b not in current_block_set]
                        
                        if valid_candidates:
                            b_in = random.choice(valid_candidates)
                            # Add b_in
                            blocks.append(b_in)
                            i_in = len(blocks) - 1
                            active.append(True)
                            for tri in block_triples[b_in]:
                                counts[tri] += 1
                            
                            # We made a move. Reset no_improvement counter
                            no_improvement_count = 0
                            changed = True
                            # Loop back to try removals
                            continue
                        else:
                            # Revert
                            active[i_out] = True
                            for tri in block_triples[b_out]:
                                counts[tri] += 1
                            no_improvement_count += 1
                    else:
                        # Revert
                        active[i_out] = True
                        for tri in block_triples[b_out]:
                            counts[tri] += 1
                        no_improvement_count += 1
                        
                    if no_improvement_count > 100:
                        break

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
            
        # If we hit the lower bound, break
        if best_len <= 69:
            break
            
    # Convert block indices back to tuples of points
    # best_blocks is a list of indices into all_blocks
    return [list(all_blocks[b_idx]) for b_idx in best_blocks]
