import itertools
import random
import time
from collections import defaultdict
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
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
            # Pick the uncovered triple with the fewest covering blocks (most constrained)
            min_count = float('inf')
            min_triple = None
            # Sample a subset if uncovered is large to speed up
            if len(uncovered) > 1000:
                sample = random.sample(list(uncovered), 1000)
            else:
                sample = list(uncovered)
            
            for tri_idx in sample:
                cnt = len(triple_to_blocks[tri_idx])
                if cnt < min_count:
                    min_count = cnt
                    min_triple = tri_idx
            
            if min_triple is None:
                break
                
            tri_idx = min_triple
            
            # Get all blocks covering this triple
            candidates = triple_to_blocks[tri_idx]
            
            best_block = None
            best_gain = -1
            
            # Evaluate candidates
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

        changed = True
        while changed:
            changed = False
            # Try to remove redundant blocks
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            
            for i in indices:
                if not active[i]:
                    continue
                if is_redundant(i):
                    remove_block(i)
                    changed = True
                    
        # Swap Loop
        no_improvement_count = 0
        
        while time.time() < deadline:
            # Try to find a redundant block again
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
                    active[i_out] = True
                    for tri in block_triples[b_out]:
                        counts[tri] += 1
                    no_improvement_count += 1
                    if no_improvement_count > 100:
                        break
                    continue
                
                # Find candidate b_in that covers all uncovered_tris
                if not uncovered_tris:
                    pass
                else:
                    # Intersect block lists for each uncovered triple
                    candidates = set(triple_to_blocks[uncovered_tris[0]])
                    
                    for tri in uncovered_tris[1:]:
                        candidates.intersection_update(triple_to_blocks[tri])
                        if not candidates:
                            break
                    
                    if candidates:
                        current_block_set = set(blocks[i] for i in range(len(blocks)) if active[i])
                        valid_candidates = [b for b in candidates if b != b_out and b not in current_block_set]
                        
                        if valid_candidates:
                            b_in = random.choice(valid_candidates)
                            blocks.append(b_in)
                            i_in = len(blocks) - 1
                            active.append(True)
                            for tri in block_triples[b_in]:
                                counts[tri] += 1
                            
                            no_improvement_count = 0
                            changed = True
                            continue
                        else:
                            active[i_out] = True
                            for tri in block_triples[b_out]:
                                counts[tri] += 1
                            no_improvement_count += 1
                    else:
                        active[i_out] = True
                        for tri in block_triples[b_out]:
                            counts[tri] += 1
                        no_improvement_count += 1
                        
                    if no_improvement_count > 100:
                        break

        # Compact the blocks list
        final_blocks = [blocks[i] for i in range(len(blocks)) if active[i]]
        return final_blocks

    def local_search(blocks):
        """
        Perform local search by swapping blocks to reduce count or maintain count with better structure.
        """
        if not blocks:
            return []
            
        counts = [0] * n_triples
        for b_idx in blocks:
            for tri in block_triples[b_idx]:
                counts[tri] += 1
                
        n = len(blocks)
        current_size = n
        
        while time.time() < deadline:
            # Try to find a block to remove
            removed = False
            for i in range(n):
                if time.time() > deadline:
                    break
                b_idx = blocks[i]
                # Check if removing this block breaks coverage
                removable = True
                for tri in block_triples[b_idx]:
                    if counts[tri] == 1:
                        removable = False
                        break
                if removable:
                    # Remove it
                    for tri in block_triples[b_idx]:
                        counts[tri] -= 1
                    blocks.pop(i)
                    n -= 1
                    current_size -= 1
                    removed = True
                    break
            
            if removed:
                continue
            
            # Try to swap: remove block i, add block j
            swapped = False
            for i in range(n):
                if time.time() > deadline:
                    break
                b_idx = blocks[i]
                # Temporarily remove
                for tri in block_triples[b_idx]:
                    counts[tri] -= 1
                    
                # Find uncovered
                uncovered = []
                for tri_idx in range(n_triples):
                    if counts[tri_idx] == 0:
                        uncovered.append(tri_idx)
                        
                if len(uncovered) > 20:
                    # Revert
                    for tri in block_triples[b_idx]:
                        counts[tri] += 1
                    continue
                
                if not uncovered:
                    # Block was redundant, but we missed it? Just remove.
                    blocks.pop(i)
                    n -= 1
                    current_size -= 1
                    swapped = True
                    break
                
                # Find replacement
                candidates = set(triple_to_blocks[uncovered[0]])
                for tri in uncovered[1:]:
                    candidates.intersection_update(triple_to_blocks[tri])
                    if not candidates:
                        break
                
                if candidates:
                    # Filter out existing blocks
                    existing = set(blocks)
                    valid = [b for b in candidates if b not in existing and b != b_idx]
                    if valid:
                        # Pick one that maximizes new coverage? Or just random
                        # To be faster, just pick random
                        new_b = random.choice(valid)
                        blocks[i] = new_b
                        for tri in block_triples[new_b]:
                            counts[tri] += 1
                        swapped = True
                        break
                    else:
                        # Revert
                        for tri in block_triples[b_idx]:
                            counts[tri] += 1
                else:
                    # Revert
                    for tri in block_triples[b_idx]:
                        counts[tri] += 1
            
            if not swapped and not removed:
                break
                
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Main Loop
    while time.time() < deadline:
        # Greedy Init
        cand = greedy_construct()
        
        # Optimize
        opt_cand = optimize_blocks(cand)
        
        # Local Search
        ls_cand = local_search(opt_cand)
        
        if len(ls_cand) < best_len:
            best_len = len(ls_cand)
            best_blocks = ls_cand
            
        if best_len <= 69:
            break

    if best_blocks is None:
        # Fallback
        cand = greedy_construct()
        best_blocks = cand

    return [list(all_blocks[b_idx]) for b_idx in best_blocks]

