import itertools
import random
import time
import math
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

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

    def greedy_randomized(seed, mode='standard', noise_scale=0.0):
        """
        mode: 'standard' - max gain, random tie-break
              'ratio'    - max gain / log(1+freq_sum)
              'weighted' - max gain * inv_freq (weighted by rarity)
              'smooth'   - max gain - noise (simulated annealing style)
        """
        random.seed(seed)
        gain = base_gain.copy().astype(np.float64)
        is_uncovered = np.ones(num_triples, dtype=np.bool_)

        selected_blocks = []
        current_temp = 1.0 if mode == 'smooth' else 0.0
        cooling = 0.95 if mode == 'smooth' else 1.0

        while is_uncovered.any():
            if time.time() > deadline - 5:
                break

            max_gain = np.max(gain)
            if max_gain == 0:
                break

            # Calculate scores based on mode
            if mode == 'ratio':
                # Compute scores for candidates with max gain to avoid full scan if possible?
                # But gain changes. Let's just find max gain first.
                candidates = np.where(gain == max_gain)[0]
                if len(candidates) > 0:
                    # For ratio, we want to penalize blocks that cover "dense" areas
                    # We approximate by summing frequencies of the triples in the block
                    best_score = -1
                    best_candidate = candidates[0]
                    # Sample a few candidates if there are many to save time
                    sample_size = min(20, len(candidates))
                    sampled = random.sample(candidates.tolist(), sample_size)
                    
                    for c in sampled:
                        block_freq_sum = 0
                        for t_idx in block_triples[c]:
                            block_freq_sum += freq[t_idx]
                        score = gain[c] / (math.log1p(block_freq_sum) + 1.0)
                        if score > best_score:
                            best_score = score
                            best_candidate = c
                    b_idx = best_candidate
                else:
                    break
            elif mode == 'weighted':
                # Weighted gain: gain * avg_inv_freq of covered triples
                # To make this efficient, we can't easily vectorize the product/sum per block
                # without a big matrix. Let's stick to a simpler heuristic:
                # Prioritize blocks covering rare triples.
                # We can approximate by just using standard greedy but with a bias.
                # Or, calculate score for top K max-gain blocks.
                candidates = np.where(gain == max_gain)[0]
                if len(candidates) > 0:
                    best_score = -1
                    best_candidate = candidates[0]
                    sample_size = min(20, len(candidates))
                    sampled = random.sample(candidates.tolist(), sample_size)
                    
                    for c in sampled:
                        # Sum of inv_freq for uncovered triples in this block
                        score = 0
                        for t_idx in block_triples[c]:
                            if is_uncovered[t_idx]:
                                score += inv_freq[t_idx]
                        # Combine with gain
                        score = gain[c] + 0.1 * score
                        if score > best_score:
                            best_score = score
                            best_candidate = c
                    b_idx = best_candidate
                else:
                    break
            elif mode == 'smooth':
                # Soft-max style: pick block with highest gain, but allow lower gain blocks
                # with some probability based on temperature
                # Score = gain - temp * random_noise
                # This effectively creates a distribution over gains.
                # We can just add noise to gain and pick max.
                noisy_gain = gain + random.gauss(0, current_temp)
                b_idx = np.argmax(noisy_gain)
                current_temp *= cooling
            else:
                # Standard: random among max gain
                candidates = np.where(gain == max_gain)[0]
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

        return [b_idx for i, b_idx in enumerate(selected_blocks) if active[i]]

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
        max_iterations = 2000
        
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
                for _ in range(50):
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
    modes = ['standard', 'ratio', 'weighted', 'smooth', 'standard', 'ratio']
    
    while True:
        if time.time() > deadline - 15:
            break

        mode = modes[num_restarts % len(modes)]
        cover = greedy_randomized(seed, mode=mode)
        cover = reduce_cover(cover)

        if best_cover is None or len(cover) < len(best_cover):
            best_cover = cover
            # If we found a new best, do a quick local search immediately?
            if time.time() < deadline - 20:
                ls_cover = local_search(best_cover, seed + 1000, time_budget=5.0)
                if len(ls_cover) < len(best_cover):
                    best_cover = ls_cover

        seed += 1
        num_restarts += 1

        # If we've done many restarts, stop
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
        # Fallback: standard greedy
        cover = greedy_randomized(0, mode='standard')
        return [list(all_blocks[b_idx]) for b_idx in cover]

    return [list(all_blocks[b_idx]) for b_idx in best_cover]
