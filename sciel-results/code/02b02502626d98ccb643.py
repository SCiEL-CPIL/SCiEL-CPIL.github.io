import itertools
import time
import random
import sys
import math
import numpy as np

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def check_time(budget=5.0):
        return time.time() - start_time > deadline - budget

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Build block_to_subsets: list of arrays of subset indices
    # This is the critical precomputation.
    block_to_subsets = []
    for block in all_blocks:
        subs = []
        for s in itertools.combinations(block, t):
            subs.append(tsub_index[s])
        block_to_subsets.append(np.array(subs, dtype=np.int32))

    def run_greedy(random_seed=None):
        """
        Greedy set cover.
        """
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()

        uncovered = np.ones(num_tsubs, dtype=bool)
        gains = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
        
        # Mask out blocks that are already selected? No, we just track uncovered.
        # Gains need to be updated.
        
        selected_blocks = []
        
        # Optimization: Instead of updating all gains every step (O(num_blocks)),
        # we can just recalculate gains for candidate blocks or use a heap.
        # But with NumPy, vectorized update is fast if we do it right.
        # However, updating gains for ALL blocks is O(num_blocks) per step.
        # Steps ~ 300. 300 * 75k = 22.5M ops. This is fine.
        
        # We need to know which blocks cover which uncovered subsets.
        # Let's stick to the previous method but ensure it's vectorized.
        
        # Actually, a faster greedy is to maintain `uncovered` and compute gain for each block as:
        # gain = sum(uncovered[subs])
        # This is O(total_subsets_per_block) per block.
        # Total work per step = sum over blocks of C(k,t) = num_blocks * C(k,t).
        # 75k * 56 = 4.2M operations per step.
        # 300 steps * 4.2M = 1.26 Billion operations. This is too slow in Python/NumPy loop.
        
        # Better: Maintain `gains` array.
        # When we select a block `b`, we cover `subs = block_to_subsets[b]`.
        # `newly_covered = subs[uncovered[subs]]`
        # For each `s` in `newly_covered`, we need to decrement gain of all blocks covering `s`.
        # This requires `subset_to_blocks`.
        
        # Let's build subset_to_blocks as a list of arrays.
        subset_to_blocks = [[] for _ in range(num_tsubs)]
        for b_idx, subs in enumerate(block_to_subsets):
            for s in subs:
                subset_to_blocks[s].append(b_idx)
        subset_to_blocks = [np.array(lst, dtype=np.int32) for lst in subset_to_blocks]

        # Reset gains
        gains = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
        
        while True:
            if check_time(10.0):
                break
            if not np.any(uncovered):
                break

            # Find best block
            # Add noise for diversity
            noise = rng.uniform(0, 0.1, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))

            if gains[best_idx] == 0:
                break

            selected_blocks.append(best_idx)

            subs = block_to_subsets[best_idx]
            if len(subs) == 0:
                continue

            # Find which of these subsets are still uncovered
            newly_covered_mask = uncovered[subs]
            newly_covered = subs[newly_covered_mask]
            
            if len(newly_covered) == 0:
                continue

            uncovered[newly_covered] = False
            
            # Update gains for blocks that cover these newly covered subsets
            # For each newly covered subset s, decrement gain of all blocks in subset_to_blocks[s]
            # We can do this in a loop over newly_covered, but vectorize the inner part?
            # Or just loop, it's usually small.
            for s in newly_covered:
                blocks_covering = subset_to_blocks[s]
                if len(blocks_covering) > 0:
                    gains[blocks_covering] -= 1

        return selected_blocks

    def local_search_deletion(blocks_indices, time_limit, random_seed=None):
        """
        Remove redundant blocks.
        """
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()

        current_blocks = list(blocks_indices)
        if not current_blocks:
            return current_blocks

        # Compute cover counts
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in current_blocks:
            subs = block_to_subsets[b_idx]
            if len(subs) > 0:
                cover_count[subs] += 1

        current_blocks_set = set(current_blocks)
        improved = True
        local_start = time.time()

        while improved and (time.time() - local_start) < time_limit and not check_time(5.0):
            improved = False
            
            # Shuffle to avoid bias
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)

            for i in indices:
                if i >= len(current_blocks):
                    continue
                if (time.time() - local_start) > time_limit or check_time(3.0):
                    break
                
                b_idx = current_blocks[i]
                subs = block_to_subsets[b_idx]
                
                if len(subs) == 0:
                    current_blocks.pop(i)
                    current_blocks_set.discard(b_idx)
                    improved = True
                    break

                # Check if redundant: all subsets in subs have count > 1
                if np.all(cover_count[subs] > 1):
                    # Remove it
                    cover_count[subs] -= 1
                    current_blocks_set.discard(b_idx)
                    current_blocks.pop(i)
                    improved = True
                    break # Restart scan

        return current_blocks

    # Main Execution
    best_blocks = []
    best_len = float('inf')
    
    # 1. Run Greedy with multiple seeds
    num_restarts = 10
    for seed in range(num_restarts):
        if check_time(20.0):
            break
            
        # Greedy
        blocks = run_greedy(random_seed=seed)
        
        if not blocks:
            continue
            
        # Local Search
        blocks = local_search_deletion(blocks, time_limit=5.0, random_seed=seed)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks

    # 2. Try to improve best_blocks with more aggressive local search / random restarts from it
    # We can try to perturb the best solution
    if best_blocks and not check_time(30.0):
        for _ in range(5):
            if check_time(10.0):
                break
            
            # Create a copy
            current = list(best_blocks)
            
            # Perturb: Add 5 random blocks not in current
            current_set = set(current)
            all_block_indices = set(range(num_blocks))
            candidates = list(all_block_indices - current_set)
            
            if candidates:
                perturbations = random.sample(candidates, min(5, len(candidates)))
                current.extend(perturbations)
                
            # Local Search
            current = local_search_deletion(current, time_limit=10.0, random_seed=42)
            
            if len(current) < best_len:
                best_len = len(current)
                best_blocks = current

    # 3. Final cleanup
    if best_blocks and not check_time(5.0):
        best_blocks = local_search_deletion(best_blocks, time_limit=5.0, random_seed=999)

    if not best_blocks:
        # Fallback: Return a valid covering if possible, or empty
        # If greedy failed completely, we might need a brute force fallback for small v, but v=19 is large.
        # However, greedy should always find a covering.
        return []

    return [list(all_blocks[i]) for i in best_blocks]
