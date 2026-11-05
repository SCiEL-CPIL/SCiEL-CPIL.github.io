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
        mode: 'standard' - max gain, random tie-break
              'ratio'    - max gain/k (ratio heuristic)
              'weighted' - max gain * inv_freq (weighted by rarity)
        """
        random.seed(seed)
        gain = base_gain.copy()
        is_uncovered = np.ones(num_triples, dtype=np.bool_)

        selected_blocks = []

        while is_uncovered.any():
            if time.time() > deadline - 5:
                break

            max_gain = np.max(gain)
            if max_gain == 0:
                break

            candidates = np.where(gain == max_gain)[0]
            
            if mode == 'ratio':
                # For ratio mode, we want to prefer blocks with fewer overlaps
                # We approximate this by looking at the total frequency of the triples in the block
                # Lower total frequency = less overlap = better ratio
                best_score = -1
                best_candidate = candidates[0]
                # Sample a few candidates if there are many
                if len(candidates) > 10:
                    sampled = random.sample(candidates.tolist(), 10)
                else:
                    sampled = candidates.tolist()
                
                for c in sampled:
                    # Calculate overlap penalty: sum of frequencies of triples in this block
                    # Higher frequency means more other blocks cover these triples
                    # So we want low frequency for "exclusive" coverage
                    # Actually, the ratio heuristic in set cover is usually gain/cost.
                    # Here cost is constant (k). So it's equivalent to max gain.
                    # However, a better proxy for "efficiency" is gain / (sum of inv_freq of covered triples)
                    # or simply gain * (product of inv_freq) ... 
                    # Let's stick to a simpler refinement: 
                    # If we use 'ratio', let's actually implement gain / sqrt(sum_freq) to penalize dense areas
                    block_freq_sum = 0
                    for t_idx in block_triples[c]:
                        block_freq_sum += freq[t_idx]
                    # Score: gain / (log(1+freq_sum) + 1)
                    score = gain[c] / (math.log1p(block_freq_sum) + 1.0)
                    if score > best_score:
                        best_score = score
                        best_candidate = c
                
                b_idx = best_candidate
            else:
                b_idx = random.choice(candidates.tolist())

            selected_blocks.append(b_idx)

            for t_idx in block_triples[b_idx]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    for other_b_idx in triple_blocks[t_idx]:
                        gain[other_b_idx] -= 1

        return selected_blocks

    def reduce_cover(selected_blocks):
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in selected_blocks:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1

        active = [True] * len(selected_blocks)

        changed = True
        while changed:
            changed = False
            for i, b_idx in enumerate(selected_blocks):
                if not active[i]:
                    continue
                redundant = True
                for t_idx in block_triples[b_idx]:
                    if cover_count[t_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    active[i] = False
                    for t_idx in block_triples[b_idx]:
                        cover_count[t_idx] -= 1
                    changed = True

        return [b_idx for i, b_idx in enumerate(selected_blocks) if active[i] if active[i]]

    def local_search(cover, seed, time_budget=10.0):
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
        max_iterations = 1000
        
        for _ in range(max_iterations):
            if time.time() > end_time:
                break
            if time.time() > deadline - 2:
                break

            if not cover:
                break

            # Pick a random block to try removing
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
                # Don't restore
                continue
            else:
                # Coverage is broken, try to add a block that covers all uncovered triples
                uncovered_set = set(uncovered.tolist())
                found_fix = False
                
                # Try random blocks
                for _ in range(30):
                    b_add = random.randint(0, num_blocks - 1)
                    if b_add in cover:
                        continue
                    
                    # Check if b_add covers all uncovered triples
                    covers_all = True
                    for t_idx in block_triples[b_add]:
                        if t_idx in uncovered_set:
                            uncovered_set.discard(t_idx)
                        if not uncovered_set:
                            break
                    
                    if not uncovered_set:
                        # b_add covers all uncovered triples
                        cover.append(b_add)
                        # Update cover counts
                        for t_idx in block_triples[b_add]:
                            cover_count[t_idx] += 1
                        found_fix = True
                        break
                
                if not found_fix:
                    # Restore the removed block
                    cover.insert(idx, b_remove)
                    for t_idx in block_triples[b_remove]:
                        cover_count[t_idx] += 1

        return best_cover

    best_cover = None
    seed = 0
    num_restarts = 0
    
    # We will cycle through different greedy modes
    modes = ['standard', 'ratio', 'weighted', 'standard']
    
    while True:
        if time.time() > deadline - 15:
            break

        mode = modes[num_restarts % len(modes)]
        cover = greedy_randomized(seed, mode=mode)
        cover = reduce_cover(cover)

        if best_cover is None or len(cover) < len(best_cover):
            best_cover = cover
            # If we found a new best, do a quick local search immediately?
            # Or save it for later. Let's do a quick one if we have time.
            if time.time() < deadline - 20:
                ls_cover = local_search(best_cover, seed + 1000, time_budget=5.0)
                if len(ls_cover) < len(best_cover):
                    best_cover = ls_cover

        seed += 1
        num_restarts += 1

        # If we've done many restarts, stop
        if num_restarts > 100:
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
        # Fallback: standard greedy
        cover = greedy_randomized(0, mode='standard')
        return [list(all_blocks[b_idx]) for b_idx in cover]

    return [list(all_blocks[b_idx]) for b_idx in best_cover]
