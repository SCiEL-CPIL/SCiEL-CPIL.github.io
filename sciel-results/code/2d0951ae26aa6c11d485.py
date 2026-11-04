import itertools
import random
import time
import math
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # 1. Pre-computation
    # Generate all possible blocks (k-subsets of v)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Generate all possible t-subsets
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # Map each block to the indices of the triples it covers
    block_triple_indices = []
    # Map each triple to the list of block indices that cover it
    triple_to_blocks = defaultdict(list)
    
    for b_idx, block in enumerate(all_blocks):
        covered_tris = []
        for tri in itertools.combinations(block, t):
            t_idx = triple_to_idx[tri]
            covered_tris.append(t_idx)
            triple_to_blocks[t_idx].append(b_idx)
        block_triple_indices.append(covered_tris)
        
    # Convert triple_to_blocks to lists of ints for faster intersection if needed, 
    # but we will use set intersection logic carefully.
    # For efficiency in the repair step, we want fast intersection of lists.
    # Since the lists can be long (each triple is in C(24-3, 7-3) = C(21,4) = 5985 blocks),
    # intersection of 5985-element lists is non-trivial but manageable if the number of uncovered triples is small.
    
    # 2. Initial Greedy Construction
    def greedy_construct():
        uncovered = set(range(num_triples))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triple to start
            start_tri_idx = random.choice(list(uncovered))
            start_tri = all_triples[start_tri_idx]
            
            block = list(start_tri)
            
            # Greedily add points to maximize coverage of uncovered triples
            while len(block) < k:
                current_block_set = set(block)
                candidates = [x for x in range(v) if x not in current_block_set]
                
                best_gain = -1
                best_candidates = []
                
                for x in candidates:
                    gain = 0
                    # Check triples formed by x and t-1 elements from current block
                    for pair in itertools.combinations(block, t-1):
                        tri = tuple(sorted(pair + (x,)))
                        tri_idx = triple_to_idx[tri]
                        if tri_idx in uncovered:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                        
                if best_candidates:
                    block.append(random.choice(best_candidates))
                else:
                    block.append(random.choice(candidates))
                    
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered
            for tri_idx in block_triple_indices[all_blocks.index(tuple(block))]:
                if tri_idx in uncovered:
                    uncovered.remove(tri_idx)
                    
        return blocks

    # Get index of a block tuple
    block_tuple_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    current_blocks = greedy_construct()
    current_block_indices = [block_tuple_to_idx[b] for b in current_blocks]
    
    # Initialize counts
    counts = [0] * num_triples
    for b_idx in current_block_indices:
        for tri_idx in block_triple_indices[b_idx]:
            counts[tri_idx] += 1
            
    best_block_indices = current_block_indices[:]
    best_len = len(current_block_indices)
    
    # 3. Optimization Loop
    # We use a random walk with "Remove and Repair" moves.
    # Move: Pick a block B. Remove it. If invalid, find a block B' that covers all exposed triples.
    # If found, replace B with B'. If B' allows further removals, do them.
    
    no_improve_count = 0
    steps = 0
    
    while time.time() < deadline:
        if no_improve_count > 5000:
            # Random restart or kick
            # Add a random block to escape local minimum
            rand_b_idx = random.randint(0, num_blocks - 1)
            if rand_b_idx not in current_block_indices:
                current_block_indices.append(rand_b_idx)
                for tri_idx in block_triple_indices[rand_b_idx]:
                    counts[tri_idx] += 1
            no_improve_count = 0
            steps = 0
            continue
            
        # Pick a random block to try to remove/swap
        pos = random.randint(0, len(current_block_indices) - 1)
        b_remove_idx = current_block_indices[pos]
        
        # Tentatively remove
        removed_tris = []
        exposed_tris = []
        for tri_idx in block_triple_indices[b_remove_idx]:
            counts[tri_idx] -= 1
            if counts[tri_idx] == 0:
                exposed_tris.append(tri_idx)
            removed_tris.append(tri_idx)
        
        # Remove from list
        current_block_indices.pop(pos)
        
        if not exposed_tris:
            # Pure removal!
            if len(current_block_indices) < best_len:
                best_len = len(current_block_indices)
                best_block_indices = current_block_indices[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
            # Chain: check if more blocks are now redundant
            # Simple check: just let the loop run again
            continue
        else:
            # Need to repair. Find a block that covers ALL exposed_tris.
            # Intersection of Triple_to_Blocks for all exposed_tris
            
            # Optimization: Start with the triple that has the fewest covering blocks
            exposed_tris_sorted = sorted(exposed_tris, key=lambda t: len(triple_to_blocks[t]))
            
            # Start with the list for the first triple
            candidates = set(triple_to_blocks[exposed_tris_sorted[0]])
            
            for t in exposed_tris_sorted[1:]:
                if not candidates:
                    break
                # Intersect with the next triple's blocks
                # To speed up, we can iterate over the smaller set
                if len(candidates) > len(triple_to_blocks[t]):
                    # Iterate over the list, check membership in candidates
                    new_candidates = set()
                    cand_list = triple_to_blocks[t]
                    cand_set = candidates
                    for c in cand_list:
                        if c in cand_set:
                            new_candidates.add(c)
                    candidates = new_candidates
                else:
                    # Iterate over candidates, check membership in list (list is faster to check? No, set is O(1))
                    # Actually, converting the list to a set might be expensive if done repeatedly.
                    # Let's assume sets are fine.
                    cand_set = candidates
                    list_blocks = triple_to_blocks[t]
                    new_candidates = set()
                    for c in list_blocks:
                        if c in cand_set:
                            new_candidates.add(c)
                    candidates = new_candidates
                    
            if candidates:
                # We have valid replacement blocks
                # Choose one that is not currently in the set if possible, 
                # or one that maximizes "potential" (e.g., covers many triples with count 1)
                
                # Filter out blocks already in current_block_indices to avoid duplicates
                # Note: Duplicates are allowed in the problem statement? 
                # "blocks must be distinct" -> Yes, distinct blocks.
                current_set = set(current_block_indices)
                valid_candidates = [c for c in candidates if c not in current_set]
                
                if not valid_candidates:
                    # All candidates are already in the set. This means we are stuck in a cycle or the set is very dense.
                    # We can still pick one, but it won't change the set. 
                    # Actually, if the block is already there, adding it back does nothing to counts? 
                    # Wait, we removed it. If we add the same one back, we return to previous state.
                    # This is a "null" move. 
                    # Let's pick a random one from valid_candidates if available, else from candidates.
                    if candidates:
                         chosen_b = random.choice(list(candidates))
                    else:
                         chosen_b = b_remove_idx # Revert
                else:
                    # Heuristic: Pick the block that covers the most "critical" triples (count==1)
                    # Or just random to diversify
                    # Let's use a weighted random choice based on coverage of low-count triples
                    # For speed, just random
                    chosen_b = random.choice(valid_candidates)
                
                # Add the new block
                current_block_indices.append(chosen_b)
                for tri_idx in block_triple_indices[chosen_b]:
                    counts[tri_idx] += 1
                    
                # Now, check if this addition made any *other* blocks redundant?
                # Unlikely to make others redundant immediately, but possible if it creates a "clique" of redundancy?
                # No, adding a block only increases counts. It cannot make a block redundant (which requires count==1).
                # So no chain reaction from addition.
                
                # However, the removal of b_remove_idx might have made *other* blocks redundant?
                # No, removing a block decreases counts. It cannot make a block redundant (count==1).
                # So only the removed block was at risk.
                
                # Check if we improved
                if len(current_block_indices) < best_len:
                    best_len = len(current_block_indices)
                    best_block_indices = current_block_indices[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # No valid repair found. Revert.
                current_block_indices.insert(pos, b_remove_idx)
                for tri_idx in removed_tris:
                    counts[tri_idx] += 1
                no_improve_count += 1

        steps += 1
        if steps % 1000 == 0 and time.time() > deadline - 5:
            break

    # Final Verification and Cleanup
    # Rebuild counts for best_block_indices to be safe
    best_counts = [0] * num_triples
    for b_idx in best_block_indices:
        for tri_idx in block_triple_indices[b_idx]:
            best_counts[tri_idx] += 1
            
    # Remove any redundant blocks from the best set found
    final_block_indices = []
    for b_idx in best_block_indices:
        is_redundant = True
        for tri_idx in block_triple_indices[b_idx]:
            if best_counts[tri_idx] == 1:
                is_redundant = False
                break
        if not is_redundant:
            final_block_indices.append(b_idx)
            for tri_idx in block_triple_indices[b_idx]:
                best_counts[tri_idx] -= 1
    # Repeat cleanup to catch chained redundancies
    changed = True
    while changed:
        changed = False
        for i in range(len(final_block_indices)):
            b_idx = final_block_indices[i]
            is_redundant = True
            for tri_idx in block_triple_indices[b_idx]:
                if best_counts[tri_idx] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                final_block_indices.pop(i)
                for tri_idx in block_triple_indices[b_idx]:
                    best_counts[tri_idx] -= 1
                changed = True
                break

    # Convert indices back to lists of integers
    final_blocks = [list(all_blocks[b_idx]) for b_idx in final_block_indices]
    
    return final_blocks
