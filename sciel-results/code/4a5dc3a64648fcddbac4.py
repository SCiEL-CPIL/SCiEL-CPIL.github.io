import itertools
import random
import time
import math
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

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

    def greedy_weighted(seed):
        """Weighted greedy: prefer blocks that cover rare triples."""
        random.seed(seed)
        gain = base_gain.copy().astype(np.float64)
        is_uncovered = np.ones(num_triples, dtype=np.bool_)

        selected_blocks = []

        while is_uncovered.any():
            if time.time() > deadline - 10:
                break

            # Score each block: sum of inv_freq for uncovered triples it covers
            # This favors blocks that cover rare triples
            best_score = -1
            best_candidate = -1
            
            # Sample candidates for speed
            sample_size = min(num_blocks, 500)
            candidates = random.sample(range(num_blocks), sample_size) if num_blocks > sample_size else range(num_blocks)
            
            for c in candidates:
                score = 0.0
                for t_idx in block_triples[c]:
                    if is_uncovered[t_idx]:
                        score += inv_freq[t_idx]
                if score > best_score:
                    best_score = score
                    best_candidate = c
            
            if best_candidate < 0 or best_score <= 0:
                break

            selected_blocks.append(best_candidate)

            for t_idx in block_triples[best_candidate]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    for other_b_idx in triple_blocks[t_idx]:
                        gain[other_b_idx] -= 1

        return selected_blocks

    def greedy_max_gain(seed):
        """Standard max-gain greedy with random tie-breaking."""
        random.seed(seed)
        gain = base_gain.copy()
        is_uncovered = np.ones(num_triples, dtype=np.bool_)

        selected_blocks = []

        while is_uncovered.any():
            if time.time() > deadline - 10:
                break

            max_gain = np.max(gain)
            if max_gain == 0:
                break

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
        """Remove redundant blocks iteratively."""
        if not selected_blocks:
            return []
            
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

    def simulated_annealing(cover, seed, time_budget=10.0):
        """Simulated annealing local search."""
        random.seed(seed)
        if not cover:
            return cover

        cover = reduce_cover(cover)
        
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in cover:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1

        best_cover = list(cover)
        best_len = len(best_cover)
        current_cover = list(cover)
        
        end_time = time.time() + time_budget
        T = 10.0  # Initial temperature
        cooling = 0.999
        
        for _ in range(50000):
            if time.time() > end_time or time.time() > deadline - 3:
                break
            if not current_cover:
                break

            # Pick a random block to try removing
            idx = random.randint(0, len(current_cover) - 1)
            b_remove = current_cover[idx]
            
            # Temporarily remove
            current_cover.pop(idx)
            
            # Update cover counts
            for t_idx in block_triples[b_remove]:
                cover_count[t_idx] -= 1
            
            uncovered = np.where(cover_count == 0)[0]
            
            if len(uncovered) == 0:
                # Coverage is still complete, keep the removal
                best_cover = list(current_cover)
                best_len = len(current_cover)
                T = max(T * cooling, 0.1)
                continue
            else:
                # Coverage is broken, try to add a block that covers all uncovered triples
                uncovered_set = set(uncovered.tolist())
                found_fix = False
                
                # Try to find a block that covers all uncovered triples
                for _ in range(50):
                    b_add = random.randint(0, num_blocks - 1)
                    if b_add in current_cover:
                        continue
                    
                    # Check if b_add covers all uncovered triples
                    remaining = set(uncovered_set)
                    for t_idx in block_triples[b_add]:
                        if t_idx in remaining:
                            remaining.discard(t_idx)
                        if not remaining:
                            break
                    
                    if not remaining:
                        # b_add covers all uncovered triples
                        current_cover.append(b_add)
                        for t_idx in block_triples[b_add]:
                            cover_count[t_idx] += 1
                        found_fix = True
                        
                        # Accept if it doesn't make things worse, or with probability
                        if len(current_cover) < best_len:
                            best_cover = list(current_cover)
                            best_len = len(current_cover)
                        break
                
                if not found_fix:
                    # Restore the removed block with some probability
                    if random.random() < T:
                        current_cover.insert(idx, b_remove)
                        for t_idx in block_triples[b_remove]:
                            cover_count[t_idx] += 1
                    # Otherwise, keep it removed (we'll try to fix later)
                    # But this might leave us in an invalid state, so restore
                    else:
                        current_cover.insert(idx, b_remove)
                        for t_idx in block_triples[b_remove]:
                            cover_count[t_idx] += 1
                
                T = max(T * cooling, 0.1)

        # Final reduction
        best_cover = reduce_cover(best_cover)
        return best_cover

    def two_block_swap(cover, seed, time_budget=5.0):
        """Try swapping one block for another to maintain coverage while reducing size."""
        random.seed(seed)
        if not cover:
            return cover

        cover = reduce_cover(cover)
        
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in cover:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1

        best_cover = list(cover)
        best_len = len(best_cover)
        
        end_time = time.time() + time_budget
        
        for _ in range(20000):
            if time.time() > end_time or time.time() > deadline - 3:
                break
            if not best_cover:
                break

            # Try removing a block and replacing with a different one
            idx = random.randint(0, len(best_cover) - 1)
            b_remove = best_cover[idx]
            
            # Find triples only covered by this block
            unique_triples = []
            for t_idx in block_triples[b_remove]:
                if cover_count[t_idx] == 1:
                    unique_triples.append(t_idx)
            
            if not unique_triples:
                # No unique triples, just remove
                best_cover.pop(idx)
                for t_idx in block_triples[b_remove]:
                    cover_count[t_idx] -= 1
                best_cover = reduce_cover(best_cover)
                best_len = len(best_cover)
                continue
            
            # Try to find a block that covers all unique triples
            unique_set = set(unique_triples)
            found_replacement = False
            
            for _ in range(30):
                b_add = random.randint(0, num_blocks - 1)
                if b_add in best_cover:
                    continue
                
                remaining = set(unique_set)
                for t_idx in block_triples[b_add]:
                    if t_idx in remaining:
                        remaining.discard(t_idx)
                    if not remaining:
                        break
                
                if not remaining:
                    # Replace
                    best_cover[idx] = b_add
                    for t_idx in block_triples[b_remove]:
                        cover_count[t_idx] -= 1
                    for t_idx in block_triples[b_add]:
                        cover_count[t_idx] += 1
                    found_replacement = True
                    break
            
            if not found_replacement:
                # Try multiple removals
                pass

        return best_cover

    best_cover = None
    seed = 0
    
    # Phase 1: Generate initial covers with different strategies
    initial_covers = []
    
    # Weighted greedy
    for s in range(5):
        if time.time() > deadline - 60:
            break
        cover = greedy_weighted(seed + s * 100)
        cover = reduce_cover(cover)
        initial_covers.append(cover)
        if best_cover is None or len(cover) < len(best_cover):
            best_cover = cover
    
    # Max gain greedy
    for s in range(5):
        if time.time() > deadline - 60:
            break
        cover = greedy_max_gain(seed + 500 + s * 100)
        cover = reduce_cover(cover)
        initial_covers.append(cover)
        if best_cover is None or len(cover) < len(best_cover):
            best_cover = cover
    
    seed = 1000
    
    # Phase 2: Local search on each initial cover
    for cover in initial_covers:
        if time.time() > deadline - 30:
            break
        if len(cover) > len(best_cover) + 5:
            continue  # Skip if too far from best
        
        remaining = min(deadline - time.time() - 5, 15.0)
        if remaining < 2:
            break
            
        ls_cover = simulated_annealing(cover, seed, time_budget=remaining)
        if len(ls_cover) < len(best_cover):
            best_cover = ls_cover
        
        seed += 1
    
    # Phase 3: Final polish with two-block swaps
    if best_cover is not None and time.time() < deadline - 10:
        remaining = deadline - time.time() - 5
        if remaining > 3:
            polish_cover = two_block_swap(best_cover, seed + 2000, time_budget=remaining)
            if len(polish_cover) < len(best_cover):
                best_cover = polish_cover
    
    # Final reduction
    if best_cover is not None:
        best_cover = reduce_cover(best_cover)
    
    # Fallback
    if best_cover is None:
        cover = greedy_max_gain(0)
        best_cover = reduce_cover(cover)

    return [list(all_blocks[b_idx]) for b_idx in best_cover]
