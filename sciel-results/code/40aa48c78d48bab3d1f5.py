"""Improved covering design search for C(19,8,5) with deadline guard."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import math


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # All t-subsets of v points
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsubset_index = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Precompute for each t-subset, the list of points that can extend it to a k-block
    # Actually, let's precompute the "extension candidates" for each t-subset
    # For a t-subset S, any point p not in S can be added
    # But more efficiently: precompute for each t-subset, the set of all points not in it
    # This is just range(v) minus S, which is small (v-t = 14 points)
    
    # Precompute all k-blocks and their covered t-subsets for potential use
    # But that's 75582 blocks x 56 t-subsets each = 4.2M entries, might be too much
    # Let's stick with the greedy approach but improve it
    
    def elapsed():
        return time.time() - start_time

    def run_greedy_v3(seed):
        """Improved greedy with better point selection and randomization."""
        rng = random.Random(seed)
        is_uncovered = [True] * n_tsubsets
        uncovered_count = n_tsubsets
        blocks = []
        
        # Track coverage count for each t-subset (for deletion phase)
        cover_count = [0] * n_tsubsets
        
        # Helper to get a random uncovered index
        def get_random_uncovered():
            for _ in range(30):
                idx = rng.randint(0, n_tsubsets - 1)
                if is_uncovered[idx]:
                    return idx
            # If failed, scan
            for i in range(n_tsubsets):
                if is_uncovered[i]:
                    return i
            return -1

        while uncovered_count > 0:
            if elapsed() > deadline - 25:
                break
            
            idx = get_random_uncovered()
            if idx == -1:
                break
                
            core_ts = all_tsubsets[idx]
            block = list(core_ts)
            block_set = set(core_ts)
            
            while len(block) < k:
                best_point = None
                best_gain = -1
                second_best_point = None
                second_best_gain = -1
                
                for p in range(v):
                    if p in block_set:
                        continue
                    
                    # Calculate gain: number of uncovered t-subsets that would be covered
                    # if we add p to the current block
                    gain = 0
                    # For each t-subset containing p and t-1 elements from block_set
                    # that is currently uncovered
                    for rest in itertools.combinations(block_set, t - 1):
                        ts_tuple = tuple(sorted(rest + (p,)))
                        ts_idx = tsubset_index[ts_tuple]
                        if is_uncovered[ts_idx]:
                            gain += 1
                    
                    if gain > best_gain:
                        second_best_point = best_point
                        second_best_gain = best_gain
                        best_point = p
                        best_gain = gain
                    elif gain > second_best_gain:
                        second_best_point = p
                        second_best_gain = gain
                
                if best_point is None:
                    break
                
                # With some probability, use second-best to add diversity
                if second_best_point is not None and rng.random() < 0.15:
                    chosen = second_best_point
                else:
                    chosen = best_point
                    
                block.append(chosen)
                block_set.add(chosen)
            
            block.sort()
            blocks.append(block)
            
            # Update coverage
            for ts in itertools.combinations(block, t):
                ts_idx = tsubset_index[ts]
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    uncovered_count -= 1
                cover_count[ts_idx] += 1
        
        return blocks, cover_count

    # Phase 1: Run greedy with multiple seeds to find a good initial solution
    best_blocks = []
    best_cover_count = []
    best_len = float('inf')
    
    num_seeds = 150
    for i in range(num_seeds):
        if elapsed() > deadline - 30:  # Leave time for local search
            break
        blocks, cover_count = run_greedy_v3(i)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            best_cover_count = cover_count
            
    # Phase 2: Local Search (Deletion)
    current_blocks = list(best_blocks)
    current_cover_count = list(best_cover_count)
    
    # Try removing blocks - multiple passes
    improved = True
    max_passes = 5
    for _pass in range(max_passes):
        if not improved:
            break
        improved = False
        # Shuffle blocks to try different orders
        rng = random.Random(_pass * 1000 + 42)
        indices = list(range(len(current_blocks)))
        rng.shuffle(indices)
        
        for idx in indices:
            if elapsed() > deadline - 5:
                break
            
            # Try removing block at idx
            b = current_blocks[idx]
            
            # Check if removing b leaves any ts with count 0
            can_remove = True
            for ts in itertools.combinations(b, t):
                ts_idx = tsubset_index[ts]
                if current_cover_count[ts_idx] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove block
                current_blocks.pop(idx)
                # Update counts
                for ts in itertools.combinations(b, t):
                    ts_idx = tsubset_index[ts]
                    current_cover_count[ts_idx] -= 1
                improved = True
                break  # Restart loop with new list
                
    if len(current_blocks) < best_len:
        best_len = len(current_blocks)
        best_blocks = current_blocks
        best_cover_count = current_cover_count

    # Phase 3: Try swapping blocks - replace a block with a different one that maintains coverage
    # This is more complex, so let's try a simpler refinement: try to find alternative blocks
    # that cover the same critical t-subsets
    
    # Phase 4: Another round of greedy with the remaining uncovered t-subsets
    # Actually, let's try a different approach: for each block, try to find a smaller
    # or different block that covers the same essential t-subsets
    
    # Let's do one more pass of deletion after potential swaps
    # But first, let's try to improve by replacing blocks
    
    # For each block, identify which t-subsets are only covered by this block
    # Then try to find an alternative block that covers those t-subsets
    # This is complex, so let's just do more deletion passes with different orderings
    
    for seed_offset in range(5):
        if elapsed() > deadline - 3:
            break
        rng = random.Random(10000 + seed_offset)
        indices = list(range(len(best_blocks)))
        rng.shuffle(indices)
        
        for idx in indices:
            if elapsed() > deadline - 2:
                break
            b = best_blocks[idx]
            can_remove = True
            for ts in itertools.combinations(b, t):
                ts_idx = tsubset_index[ts]
                if best_cover_count[ts_idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                best_blocks.pop(idx)
                for ts in itertools.combinations(b, t):
                    ts_idx = tsubset_index[ts]
                    best_cover_count[ts_idx] -= 1

    return best_blocks
# EVOLVE-BLOCK-END
