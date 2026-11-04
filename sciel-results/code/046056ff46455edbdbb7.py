import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin for 120s limit
    
    # Pre-compute all t-subsets for the universe
    # For v=24, k=7, t=3, this is C(24,3) = 2024 triples.
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    
    # Map triple to index for faster counting
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # Helper to get indices of triples in a block
    def get_block_triple_indices(block):
        return [triple_to_idx[tri] for tri in itertools.combinations(block, t)]

    # 1. Greedy Construction with Ratio Heuristic
    # We build blocks one by one. For each new block, we want to maximize
    # the number of *newly* covered triples.
    # To make this efficient, we pick a starting triple, then greedily add points
    # that maximize the coverage of *uncovered* triples involving those points.
    
    def greedy_construct():
        uncovered = set(range(num_triples))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triple to start the block
            start_tri_idx = random.choice(list(uncovered))
            start_tri = all_triples[start_tri_idx]
            
            block = list(start_tri)
            
            while len(block) < k:
                current_block_set = set(block)
                candidates = [x for x in range(v) if x not in current_block_set]
                
                best_gain = -1
                best_candidates = []
                
                for x in candidates:
                    # Calculate gain: number of triples in new_block that are in uncovered
                    # We only need to check triples involving x and pairs from block
                    gain = 0
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
                    # Random tie-breaking to explore different structures
                    block.append(random.choice(best_candidates))
                else:
                    block.append(random.choice(candidates))
                    
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered
            for tri_idx in get_block_triple_indices(block):
                if tri_idx in uncovered:
                    uncovered.remove(tri_idx)
                    
        return blocks

    # Initial solution
    current_blocks = greedy_construct()
    best_blocks = current_blocks[:]
    best_len = len(current_blocks)
    
    # State for optimization
    # counts[i] is the number of blocks covering triple i
    counts = [0] * num_triples
    for b in current_blocks:
        for tri_idx in get_block_triple_indices(b):
            counts[tri_idx] += 1
    
    # Optimization Loop:
    # 1. Aggressively remove redundant blocks.
    # 2. Try to swap a block with another that allows removal of more blocks or keeps count same but is "better" (harder to define, so we focus on count reduction).
    # Actually, a simpler and very effective local search for covering designs is:
    # - Remove any block that doesn't cover any unique triple (redundant).
    # - Try to replace a block with a different one that allows the removal of another block? 
    #   This is complex. 
    # - Alternative: "Shift" moves. Move a point from block A to block B? No, blocks must be size k.
    # - Let's stick to: 
    #   a) Remove redundant blocks.
    #   b) If stuck, try to find a block that can be removed if we *add* another block? 
    #      This increases count temporarily. 
    #      Actually, a better move for reducing count is finding two blocks B1, B2 
    #      such that replacing B1 with B3 allows B2 to be removed? 
    #      
    #      Let's use a simpler heuristic that works well:
    #      "Greedy Removal with Repair":
    #      1. Remove a random block.
    #      2. If validity is broken, try to repair by adding a block that covers the most uncovered triples.
    #      3. If the repair adds only 1 block (net 0 change) or 0 blocks (net -1 change), keep it.
    #      4. If it adds >1, revert.
    #      This effectively performs 1-for-1 swaps or pure removals.

    def try_remove_and_repair(blocks, counts):
        """
        Attempts to remove a random block and repair if necessary.
        Returns (new_blocks, new_counts, success)
        """
        if len(blocks) <= 1:
            return blocks, counts, False
        
        idx = random.randint(0, len(blocks) - 1)
        b_remove = blocks[idx]
        b_indices = get_block_triple_indices(b_remove)
        
        # Tentatively remove
        for tri_idx in b_indices:
            counts[tri_idx] -= 1
        
        new_blocks = blocks[:idx] + blocks[idx+1:]
        
        # Check validity
        uncovered_triples = []
        for i in range(num_triples):
            if counts[i] == 0:
                uncovered_triples.append(i)
        
        if not uncovered_triples:
            # Pure removal! Great.
            return new_blocks, counts, True
        
        # Need to repair. Find a block that covers the most uncovered triples.
        # To be efficient, we can just pick a random uncovered triple and build a block around it?
        # Or scan all possible blocks? C(24,7) is 346k. Scanning all is too slow per step.
        # Instead, use a greedy repair:
        # Pick an uncovered triple, build a block greedily to cover as many uncovered as possible.
        
        # Pick random uncovered triple
        target_idx = random.choice(uncovered_triples)
        target_tri = all_triples[target_idx]
        
        repair_block = list(target_tri)
        while len(repair_block) < k:
            current_set = set(repair_block)
            candidates = [x for x in range(v) if x not in current_set]
            best_gain = -1
            best_cands = []
            for x in candidates:
                gain = 0
                for pair in itertools.combinations(repair_block, t-1):
                    tri = tuple(sorted(pair + (x,)))
                    tri_idx = triple_to_idx[tri]
                    if counts[tri_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_cands = [x]
                elif gain == best_gain:
                    best_cands.append(x)
            if best_cands:
                repair_block.append(random.choice(best_cands))
            else:
                repair_block.append(random.choice(candidates))
        
        repair_block.sort()
        repair_block = tuple(repair_block)
        
        # Calculate if this repair is worth it.
        # It adds 1 block. We removed 1 block. Net change 0.
        # But we might have removed a block that was "bad" and added a "good" one?
        # Or we might have failed to cover everything.
        
        # Check if repair_block covers all uncovered_triples
        repair_indices = get_block_triple_indices(repair_block)
        repair_set = set(repair_indices)
        
        # We need to check if all uncovered_triples are in repair_set
        # And also, does adding this block create new redundancy that allows further removal?
        # For now, just accept if it restores validity.
        
        is_valid_repair = True
        for tri_idx in uncovered_triples:
            if tri_idx not in repair_set:
                is_valid_repair = False
                break
        
        if is_valid_repair:
            # Apply repair
            for tri_idx in repair_indices:
                counts[tri_idx] += 1
            new_blocks.append(repair_block)
            return new_blocks, counts, True
        else:
            # Revert removal
            for tri_idx in b_indices:
                counts[tri_idx] += 1
            return blocks, counts, False

    # Run optimization
    steps = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        # Phase 1: Remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(current_blocks)):
                b = current_blocks[i]
                b_indices = get_block_triple_indices(b)
                can_remove = True
                for tri_idx in b_indices:
                    if counts[tri_idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for tri_idx in b_indices:
                        counts[tri_idx] -= 1
                    current_blocks.pop(i)
                    changed = True
                    break # Restart loop
        
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = current_blocks[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Phase 2: Try Remove and Repair
        # Limit number of attempts per cycle to avoid wasting time if stuck
        attempts = 0
        max_attempts = 50
        while attempts < max_attempts and time.time() < deadline:
            new_blocks, new_counts, success = try_remove_and_repair(current_blocks, counts)
            if success:
                current_blocks = new_blocks
                counts = new_counts
                attempts = 0 # Reset attempts if we made progress (even if net 0, it's a move)
                # Check for improvement
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks[:]
                    no_improve_count = 0
            else:
                attempts += 1
                # If we fail a lot, maybe try a random restart or different strategy?
                # For now, just break out to do redundancy removal again.
                if attempts > 100:
                    break
        
        # If we've been stuck for a while, try a "kick"
        # A kick could be adding a random block to escape local minima, then removing redundancies.
        if no_improve_count > 50 and time.time() < deadline - 5:
            # Add a random block
            new_block = tuple(sorted(random.sample(range(v), k)))
            if new_block not in current_blocks:
                current_blocks.append(new_block)
                for tri_idx in get_block_triple_indices(new_block):
                    counts[tri_idx] += 1
                no_improve_count = 0 # Reset counter to allow search in new area

    # Final cleanup
    # Rebuild counts for best_blocks to be safe
    best_counts = [0] * num_triples
    for b in best_blocks:
        for tri_idx in get_block_triple_indices(b):
            best_counts[tri_idx] += 1
            
    final_blocks = []
    for b in best_blocks:
        b_indices = get_block_triple_indices(b)
        can_remove = True
        for tri_idx in b_indices:
            if best_counts[tri_idx] == 1:
                can_remove = False
                break
        if not can_remove:
            final_blocks.append(b)
            for tri_idx in b_indices:
                best_counts[tri_idx] -= 1
        
    return [list(b) for b in final_blocks]
