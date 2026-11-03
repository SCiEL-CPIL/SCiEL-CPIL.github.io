"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math

def _precompute_data(v, k, t):
    """
    Precompute all t-subsets, all k-blocks, and the mapping between them.
    Returns:
    - all_subsets: list of all t-subsets as tuples
    - subset_to_idx: dict mapping frozenset(t-subset) to index
    - all_blocks: list of all k-blocks as tuples
    - block_triples: dict mapping block tuple to list of triple indices it covers
    """
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {frozenset(s): i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    block_triples = {}
    for b in all_blocks:
        # For each block, find the indices of the t-subsets it contains
        triples = []
        for c in itertools.combinations(b, t):
            idx = subset_to_idx[frozenset(c)]
            triples.append(idx)
        block_triples[b] = triples
        
    return all_subsets, subset_to_idx, all_blocks, block_triples

def _greedy_cover(v, k, t, rng, all_subsets, subset_to_idx, all_blocks, block_triples):
    """
    Greedy construction of an initial covering.
    """
    uncovered = set(range(len(all_subsets)))
    blocks = []
    block_set = set()
    
    while uncovered:
        # Pick a random uncovered triple as seed
        seed_idx = rng.choice(list(uncovered))
        seed = all_subsets[seed_idx]
        seed_set = set(seed)
        
        # Find the best block containing the seed that covers the most uncovered triples
        best_block = None
        best_gain = -1
        
        # Iterate through all blocks, filter those containing the seed
        # Optimization: We can't easily filter all_blocks by seed without a precomputed index.
        # For v=24, k=7, total blocks is ~173k. Filtering is O(173k). 
        # Doing this for each step of greedy might be slow if we do it many times.
        # However, greedy usually takes < 100 steps. 100 * 173k = 17M ops, which is fine in Python if optimized.
        # Let's try to be smarter: just check candidates? No, let's just scan.
        
        for b in all_blocks:
            if seed_set.issubset(set(b)):
                gain = 0
                for idx in block_triples[b]:
                    if idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
        
        if best_block is None:
            # Should not happen if covering number is finite
            break
            
        block_set.add(best_block)
        blocks.append(best_block)
        
        # Update uncovered
        for idx in block_triples[best_block]:
            uncovered.discard(idx)
            
    return blocks

def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0  # leave 5s buffer

    # Precompute
    all_subsets, subset_to_idx, all_blocks, block_triples = _precompute_data(v, k, t)
    n_subsets = len(all_subsets)
    
    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng, all_subsets, subset_to_idx, all_blocks, block_triples)
    
    # State: counts of coverage for each triple
    counts = [0] * n_subsets
    for b in blocks:
        for idx in block_triples[b]:
            counts[idx] += 1
            
    best_blocks = [list(b) for b in blocks]
    best_count = len(blocks)
    
    # Simulated Annealing parameters
    T0 = 10.0
    T_min = 0.01
    cooling_rate = 0.995
    temperature = T0
    
    no_improve_iter = 0
    max_no_improve = 100
    
    while time.time() - start_time < time_limit:
        
        # 1. Try to remove redundant blocks (always beneficial if valid)
        removed_any = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            b = blocks[i]
            redundant = True
            for idx in block_triples[b]:
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for idx in block_triples[b]:
                    counts[idx] -= 1
                blocks.pop(i)
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
                removed_any = True
                no_improve_iter = 0
                break
                
        if removed_any:
            temperature = max(T_min, temperature * cooling_rate)
            continue
            
        # 2. Try Swap: Replace one block with another
        # We want to minimize the number of blocks, but we are constrained to validity.
        # A swap keeps the count constant but might enable future removals.
        # A "Merge" (2 blocks -> 1) reduces count by 1.
        
        # 2a. Try Merge: Replace 2 blocks with 1
        merged = False
        if len(blocks) >= 2:
            # Try a few random pairs
            for _ in range(5):
                i, j = rng.sample(range(len(blocks)), 2)
                b1 = blocks[i]
                b2 = blocks[j]
                
                # Critical triples for b1 (count == 1) and b2 (count == 1)
                # If we remove b1 and b2, we lose coverage for critical triples.
                # The new block must cover all critical triples of b1 and b2.
                
                critical_set = set()
                for idx in block_triples[b1]:
                    if counts[idx] == 1:
                        critical_set.add(idx)
                for idx in block_triples[b2]:
                    if counts[idx] == 1:
                        critical_set.add(idx)
                        
                if not critical_set:
                    # If no critical triples, we can just remove one of them? 
                    # No, the redundancy check in step 1 would have caught that.
                    # If critical_set is empty, it means all triples in b1 and b2 are covered by others.
                    # So we can remove both? No, we replace 2 with 1.
                    # Actually if critical_set is empty, we can just remove one block (redundant).
                    # Step 1 should have handled this. But maybe counts changed?
                    # If critical_set is empty, any valid block covering the union of points?
                    # No, we just need to keep the covering valid.
                    # If all triples in b1 are covered by others, b1 is redundant.
                    # So this case shouldn't persist.
                    continue

                # Find points in critical triples
                points_needed = set()
                for idx in critical_set:
                    points_needed.update(all_subsets[idx])
                    
                if len(points_needed) > k:
                    continue
                    
                # Find a block that contains points_needed
                # And ideally covers other triples to compensate for lost coverage of b1/b2
                best_new_block = None
                best_score = -1
                
                # Heuristic: We want the new block to cover as many triples from b1 U b2 as possible
                # to minimize the "gap" in coverage, although we don't strictly need to cover them 
                # if they are covered by other blocks.
                
                target_triples = set(block_triples[b1]) | set(block_triples[b2])
                
                for b in all_blocks:
                    if points_needed.issubset(set(b)):
                        # Check if this block is valid to replace b1, b2
                        # It must cover all critical triples (guaranteed by points_needed check? 
                        # No, points_needed is the union of points. The block must contain the specific triples.
                        # If the block contains the points, does it contain the triples?
                        # Yes, if it contains the points of a triple, it contains that triple.
                        
                        # Score: number of target_triples covered by this block
                        # Plus penalty if it doesn't cover some criticals? (Already checked)
                        
                        covered_count = 0
                        for idx in block_triples[b]:
                            if idx in target_triples:
                                covered_count += 1
                                
                        # Also prefer blocks that are "dense" in coverage of currently under-covered areas?
                        # Let's stick to maximizing overlap with b1 U b2.
                        
                        if covered_count > best_score:
                            best_score = covered_count
                            best_new_block = b
                            
                if best_new_block is not None:
                    # Check if swap is valid
                    # Remove b1, b2
                    for idx in block_triples[b1]:
                        counts[idx] -= 1
                    for idx in block_triples[b2]:
                        counts[idx] -= 1
                        
                    # Add best_new_block
                    for idx in block_triples[best_new_block]:
                        counts[idx] += 1
                        
                    # Check validity
                    valid = True
                    for idx in range(n_subsets):
                        if counts[idx] == 0:
                            valid = False
                            break
                            
                    if valid:
                        # Apply merge
                        # Remove i and j (be careful with indices)
                        if i > j:
                            i, j = j, i
                        # Remove j first, then i (since i > j, removing j doesn't shift i)
                        blocks.pop(j)
                        blocks.pop(i)
                        blocks.append(best_new_block)
                        
                        if len(blocks) < best_count:
                            best_count = len(blocks)
                            best_blocks = [list(b) for b in blocks]
                        merged = True
                        no_improve_iter = 0
                        break
                    else:
                        # Restore
                        for idx in block_triples[best_new_block]:
                            counts[idx] -= 1
                        for idx in block_triples[b2]:
                            counts[idx] += 1
                        for idx in block_triples[b1]:
                            counts[idx] += 1
        
        if merged:
            temperature = max(T_min, temperature * cooling_rate)
            continue
            
        # 2b. Try Swap (1 for 1) to escape local minima
        swapped = False
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            
            # Critical triples for old_block
            critical = [idx for idx in block_triples[old_block] if counts[idx] == 1]
            
            if not critical:
                # If no critical triples, we can swap to any block?
                # No, we should try to swap to a block that helps future removals.
                # But for now, let's just try to find a block that covers the same "territory" better.
                # Actually, if no criticals, the block is redundant. Step 1 should have removed it.
                # So this case is rare.
                pass
            else:
                points_needed = set()
                for idx in critical:
                    points_needed.update(all_subsets[idx])
                    
                if len(points_needed) <= k:
                    # Find a new block containing points_needed
                    best_new_block = None
                    best_gain = -1
                    
                    for b in all_blocks:
                        if points_needed.issubset(set(b)):
                            if b == old_block:
                                continue
                            # Calculate gain: how many NEW triples does this block cover that were not covered by old_block?
                            # Or rather, how does it change the count distribution?
                            # We want to minimize the "fragility" (number of triples with count 1).
                            
                            # Simulate swap
                            # Remove old_block
                            # Add new_block
                            
                            # Heuristic: Count how many triples in new_block have count == 0 (after removal of old_block)?
                            # This is complex. Let's use a simpler metric.
                            # Score = - (number of critical triples in new_block that are NOT in old_block? No)
                            
                            # Let's just pick a random valid block containing points_needed.
                            # Or pick the one that maximizes coverage of triples with count == 1 (other than criticals)?
                            
                            # Simple heuristic: Random selection from valid candidates.
                            # To make it deterministic-ish for reproducibility but varied by rng:
                            # Let's collect a few candidates and pick one.
                            
                            # Optimization: Instead of scanning all 173k blocks every time,
                            # we can precompute for each set of points (or subset) the blocks containing them?
                            # That's too big.
                            
                            # Let's just scan. 173k iterations is fast in Python if simple.
                            # But doing it every SA step might be slow.
                            # Let's limit the search or use a heuristic.
                            
                            # Heuristic: Pick the block that shares the most points with old_block?
                            # Or pick the block that covers the most "uncovered" triples?
                            
                            # Let's try to find a block that is "different" from old_block to explore space.
                            
                            # For speed, let's just take the first few valid ones and pick randomly.
                            candidates = []
                            for b in all_blocks:
                                if points_needed.issubset(set(b)):
                                    if b != old_block:
                                        candidates.append(b)
                                        if len(candidates) > 10:
                                            break
                                            
                            if candidates:
                                new_block = rng.choice(candidates)
                                
                                # Apply swap
                                for idx in block_triples[old_block]:
                                    counts[idx] -= 1
                                for idx in block_triples[new_block]:
                                    counts[idx] += 1
                                    
                                # Check validity
                                valid = True
                                for idx in range(n_subsets):
                                    if counts[idx] == 0:
                                        valid = False
                                        break
                                        
                                if valid:
                                    blocks[i] = new_block
                                    swapped = True
                                    no_improve_iter = 0
                                else:
                                    # Restore
                                    for idx in block_triples[new_block]:
                                        counts[idx] -= 1
                                    for idx in block_triples[old_block]:
                                        counts[idx] += 1
                                    # Try next random i? Or break?
                                    # Let's break and try again next iteration
                                    break
                            
        if swapped:
            temperature = max(T_min, temperature * cooling_rate)
            continue
            
        # 3. Perturbation: If stuck, do a random restart of a few blocks
        if no_improve_iter > max_no_improve:
            # Remove 2-3 random blocks and greedily re-add
            if len(blocks) > 3:
                old_blocks = list(blocks)
                old_counts = list(counts)
                
                num_remove = min(rng.randint(2, 3), len(blocks) - 1)
                remove_indices = rng.sample(range(len(blocks)), num_remove)
                remove_indices.sort(reverse=True)
                
                for i in remove_indices:
                    b = blocks[i]
                    for idx in block_triples[b]:
                        counts[idx] -= 1
                    blocks.pop(i)
                    
                # Greedily add blocks to cover uncovered
                uncovered = set(idx for idx in range(n_subsets) if counts[idx] == 0)
                
                while uncovered:
                    seed_idx = rng.choice(list(uncovered))
                    seed = all_subsets[seed_idx]
                    seed_set = set(seed)
                    
                    best_block = None
                    best_gain = -1
                    for b in all_blocks:
                        if seed_set.issubset(set(b)):
                            gain = 0
                            for idx in block_triples[b]:
                                if idx in uncovered:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_block = b
                                
                    if best_block is None:
                        break
                        
                    if best_block not in blocks:
                        blocks.append(best_block)
                        for idx in block_triples[best_block]:
                            counts[idx] += 1
                            uncovered.discard(idx)
                    else:
                        # If best_block is already there, pick next best?
                        # For simplicity, break and restore if stuck
                        break
                        
                if not uncovered:
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = [list(b) for b in blocks]
                    no_improve_iter = 0
                    temperature = max(T_min, temperature * 0.9)
                else:
                    # Restore
                    blocks = old_blocks
                    counts = old_counts
                    no_improve_iter += 1
                    temperature = max(T_min, temperature * 0.9)
            else:
                no_improve_iter += 1
        else:
            no_improve_iter += 1
            temperature = max(T_min, temperature * cooling_rate)

    return best_blocks
# EVOLVE-BLOCK-END
