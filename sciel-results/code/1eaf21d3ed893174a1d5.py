import itertools
import random
import time
import math
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin before 120s limit

    # Precomputation
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)

    # Generate all blocks
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)

    # Map block index to list of triple indices it covers
    block_triples = []
    for b in all_blocks:
        ts = [triple_to_idx[triple] for triple in itertools.combinations(b, t)]
        block_triples.append(ts)

    # Map triple index to list of block indices covering it
    triple_blocks = [[] for _ in range(num_triples)]
    for b_idx, ts in enumerate(block_triples):
        for t_idx in ts:
            triple_blocks[t_idx].append(b_idx)

    # Precompute frequency of each triple across all blocks
    freq = np.bincount([t_idx for ts in block_triples for t_idx in ts], minlength=num_triples).astype(np.float64)
    inv_freq = 1.0 / (freq + 1e-9)
    
    # Precompute base gain (number of triples per block) - constant for all blocks
    base_gain = np.array([len(ts) for ts in block_triples], dtype=np.int32)

    def greedy_randomized(seed, mode='standard'):
        """
        Fast greedy with random tie-breaking.
        mode: 'standard' - max gain, random tie-break
              'weighted' - max gain * inv_freq (weighted by rarity)
              'random'   - pick random block among those with positive gain
        """
        random.seed(seed)
        gain = base_gain.copy().astype(np.int32)
        is_uncovered = np.ones(num_triples, dtype=np.bool_)

        selected_blocks = []

        while True:
            if time.time() > deadline - 5:
                break

            if not is_uncovered.any():
                break

            if mode == 'random':
                # Pick a random block that has positive gain
                pos = np.where(gain > 0)[0]
                if len(pos) == 0:
                    break
                b_idx = pos[random.randint(0, len(pos) - 1)]
            else:
                max_gain = np.max(gain)
                if max_gain == 0:
                    break
                candidates = np.where(gain == max_gain)[0]
                b_idx = candidates[random.randint(0, len(candidates) - 1)]

            selected_blocks.append(b_idx)

            for t_idx in block_triples[b_idx]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    for other_b_idx in triple_blocks[t_idx]:
                        gain[other_b_idx] -= 1

        return selected_blocks

    def reduce_cover(selected_blocks):
        """Remove redundant blocks from a valid cover."""
        if not selected_blocks:
            return []
        
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in selected_blocks:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1

        active = [True] * len(selected_blocks)
        result = []

        # Greedy removal: repeatedly find and remove a redundant block
        # Use a worklist approach for efficiency
        for i, b_idx in enumerate(selected_blocks):
            redundant = True
            for t_idx in block_triples[b_idx]:
                if cover_count[t_idx] <= 1:
                    redundant = False
                    break
            if redundant:
                active[i] = False
                for t_idx in block_triples[b_idx]:
                    cover_count[t_idx] -= 1
            else:
                result.append(b_idx)

        # Second pass: after removing some, others may become redundant
        changed = True
        while changed:
            changed = False
            new_result = []
            for b_idx in result:
                redundant = True
                for t_idx in block_triples[b_idx]:
                    if cover_count[t_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for t_idx in block_triples[b_idx]:
                        cover_count[t_idx] -= 1
                    changed = True
                else:
                    new_result.append(b_idx)
            result = new_result

        return result

    def local_search(cover, seed, time_budget=10.0):
        """
        Local search with:
        1. Remove-and-repair: remove a block, if coverage breaks, try to repair
        2. Block swap: replace one block with another to maintain coverage
        """
        random.seed(seed)
        if not cover:
            return cover

        # First, remove redundant blocks
        cover = reduce_cover(cover)
        
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in cover:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1

        best_cover = list(cover)
        best_len = len(best_cover)
        
        end_time = time.time() + time_budget
        
        while time.time() < end_time and time.time() < deadline - 2:
            if not cover:
                break

            # Choose move type: 70% remove-and-repair, 30% swap
            move_type = random.random()
            
            if move_type < 0.7:
                # Remove-and-repair
                idx = random.randint(0, len(cover) - 1)
                b_remove = cover[idx]
                
                # Temporarily remove
                cover.pop(idx)
                
                # Update cover counts
                for t_idx in block_triples[b_remove]:
                    cover_count[t_idx] -= 1
                
                uncovered = np.where(cover_count == 0)[0]
                
                if len(uncovered) == 0:
                    # Coverage is still complete, keep the removal
                    best_cover = list(cover)
                    best_len = len(cover)
                    continue
                else:
                    # Coverage is broken, try to repair
                    uncovered_list = uncovered.tolist()
                    uncovered_set = set(uncovered_list)
                    
                    # Strategy 1: Find a single block that covers ALL uncovered triples
                    # Strategy 2: Find a block that covers the MOST uncovered triples, then add more if needed
                    
                    best_repair = None
                    best_repair_gain = -1
                    
                    # Sample candidate blocks for repair
                    num_samples = min(50, num_blocks)
                    sampled = random.sample(range(num_blocks), num_samples)
                    
                    for b_add in sampled:
                        if b_add in cover:
                            continue
                        
                        # Count how many uncovered triples this block covers
                        gain_count = 0
                        for t_idx in block_triples[b_add]:
                            if cover_count[t_idx] == 0:
                                gain_count += 1
                        
                        if gain_count > best_repair_gain:
                            best_repair_gain = gain_count
                            best_repair = b_add
                        
                        if gain_count == len(uncovered_list):
                            # Perfect repair found
                            break
                    
                    if best_repair is not None and best_repair_gain == len(uncovered_list):
                        # Perfect single-block repair
                        cover.append(best_repair)
                        for t_idx in block_triples[best_repair]:
                            cover_count[t_idx] += 1
                        best_cover = list(cover)
                        best_len = len(cover)
                        continue
                    elif best_repair is not None and best_repair_gain > 0:
                        # Partial repair: add the best block, then try to cover remaining
                        cover.append(best_repair)
                        for t_idx in block_triples[best_repair]:
                            cover_count[t_idx] += 1
                        
                        # Check if we need more blocks
                        still_uncovered = np.where(cover_count == 0)[0]
                        if len(still_uncovered) == 0:
                            best_cover = list(cover)
                            best_len = len(cover)
                        else:
                            # Try to add one more block
                            still_unc_set = set(still_uncovered.tolist())
                            found_second = False
                            for _ in range(30):
                                b2 = random.randint(0, num_blocks - 1)
                                if b2 in cover:
                                    continue
                                covers_all = True
                                for t_idx in block_triples[b2]:
                                    if t_idx in still_unc_set:
                                        still_unc_set.discard(t_idx)
                                    if not still_unc_set:
                                        break
                                if not still_unc_set:
                                    cover.append(b2)
                                    for t_idx in block_triples[b2]:
                                        cover_count[t_idx] += 1
                                    found_second = True
                                    break
                            
                            if found_second:
                                best_cover = list(cover)
                                best_len = len(cover)
                            else:
                                # Rollback: remove the partial repair
                                cover.pop()
                                for t_idx in block_triples[best_repair]:
                                    cover_count[t_idx] -= 1
                                # Restore original block
                                cover.insert(idx, b_remove)
                                for t_idx in block_triples[b_remove]:
                                    cover_count[t_idx] += 1
                    else:
                        # No good repair found, restore
                        cover.insert(idx, b_remove)
                        for t_idx in block_triples[b_remove]:
                            cover_count[t_idx] += 1
            else:
                # Swap move: remove one block, add another
                if len(cover) < 2:
                    continue
                    
                idx = random.randint(0, len(cover) - 1)
                b_remove = cover[idx]
                
                # Try random replacement blocks
                found_swap = False
                for _ in range(20):
                    b_add = random.randint(0, num_blocks - 1)
                    if b_add == b_remove or b_add in cover:
                        continue
                    
                    # Check if replacing b_remove with b_add maintains coverage
                    # A triple is still covered if it's covered by b_add OR by another block
                    valid = True
                    for t_idx in block_triples[b_remove]:
                        if cover_count[t_idx] <= 1:
                            # This triple is only covered by b_remove
                            # Check if b_add covers it
                            covered_by_add = False
                            for t2 in block_triples[b_add]:
                                if t2 == t_idx:
                                    covered_by_add = True
                                    break
                            if not covered_by_add:
                                valid = False
                                break
                    
                    if valid:
                        # Perform the swap
                        cover[idx] = b_add
                        for t_idx in block_triples[b_remove]:
                            cover_count[t_idx] -= 1
                        for t_idx in block_triples[b_add]:
                            cover_count[t_idx] += 1
                        found_swap = True
                        break
                
                if found_swap:
                    best_cover = list(cover)
                    best_len = len(cover)

        return best_cover

    best_cover = None
    seed = 0
    num_restarts = 0
    
    # We will cycle through different greedy modes
    modes = ['standard', 'weighted', 'random', 'standard']
    
    while True:
        if time.time() > deadline - 15:
            break

        mode = modes[num_restarts % len(modes)]
        cover = greedy_randomized(seed, mode=mode)
        cover = reduce_cover(cover)

        if best_cover is None or len(cover) < len(best_cover):
            best_cover = cover
            if time.time() < deadline - 20:
                ls_cover = local_search(best_cover, seed + 1000, time_budget=5.0)
                if len(ls_cover) < len(best_cover):
                    best_cover = ls_cover

        seed += 1
        num_restarts += 1

        if num_restarts > 150:
            break

    # Final local search on the best cover
    if best_cover is not None:
        remaining_time = deadline - time.time() - 2
        if remaining_time > 5:
            ls_cover = local_search(best_cover, seed + 2000, time_budget=remaining_time)
            if len(ls_cover) < len(best_cover):
                best_cover = ls_cover

    # Convert indices back to blocks
    if best_cover is None:
        cover = greedy_randomized(0, mode='standard')
        return [list(all_blocks[b_idx]) for b_idx in cover]

    return [list(all_blocks[b_idx]) for b_idx in best_cover]
