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
    deadline = start_time + 115.0  # Safety margin for 120s limit

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
    block_to_subsets = []
    for block in all_blocks:
        subs = []
        for s in itertools.combinations(block, t):
            subs.append(tsub_index[s])
        block_to_subsets.append(np.array(subs, dtype=np.int32))

    # Build subset_to_blocks: for each t-subset, which blocks cover it
    subset_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, subs in enumerate(block_to_subsets):
        for s in subs:
            subset_to_blocks[s].append(b_idx)
    subset_to_blocks = [np.array(lst, dtype=np.int32) for lst in subset_to_blocks]

    def greedy_cover(seed=0, noise_scale=0.0):
        """
        Greedy set cover with optional randomized tie-breaking.
        Returns list of block indices.
        """
        rng = np.random.default_rng(seed)
        uncovered = np.ones(num_tsubs, dtype=bool)
        
        # Initial gains
        gains = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
        
        selected = []
        
        # We need to efficiently update gains.
        # When we pick block b, we cover subs = block_to_subsets[b].
        # Newly covered: subs[uncovered[subs]]
        # For each newly covered s, decrement gain of all blocks in subset_to_blocks[s].
        
        while True:
            if check_time(10.0):
                break
            if not np.any(uncovered):
                break
            
            # Find best block
            scores = gains.astype(np.float64)
            if noise_scale > 0:
                scores += rng.uniform(0, noise_scale, size=num_blocks)
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected.append(best_idx)
            
            subs = block_to_subsets[best_idx]
            newly_covered_mask = uncovered[subs]
            newly_covered = subs[newly_covered_mask]
            
            if len(newly_covered) == 0:
                continue
                
            uncovered[newly_covered] = False
            
            # Update gains
            # We can batch this if we use a trick, but looping over newly_covered is O(k) avg.
            # Since C(8,5)=56, and we have ~300 blocks, total ops ~ 300*56*avg_blocks_per_subset.
            # This is acceptable.
            for s in newly_covered:
                blocks_covering = subset_to_blocks[s]
                if len(blocks_covering) > 0:
                    gains[blocks_covering] -= 1

        return selected

    def local_search_deletion(blocks_indices, time_limit=5.0, seed=0):
        """
        Remove redundant blocks.
        """
        rng = random.Random(seed)
        current_blocks = list(blocks_indices)
        if not current_blocks:
            return current_blocks

        # Compute cover counts
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in current_blocks:
            subs = block_to_subsets[b_idx]
            if len(subs) > 0:
                cover_count[subs] += 1

        improved = True
        local_start = time.time()

        while improved and (time.time() - local_start) < time_limit and not check_time(5.0):
            improved = False
            
            # Shuffle indices to avoid bias
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
                    improved = True
                    break

                # Check if redundant: all subsets in subs have count > 1
                if np.all(cover_count[subs] > 1):
                    cover_count[subs] -= 1
                    current_blocks.pop(i)
                    improved = True
                    break

        return current_blocks

    def swap_improve(blocks_indices, time_limit=10.0, seed=0):
        """
        Try to swap a block for two others to reduce size? No, that increases size.
        Try to remove one and add one? No.
        Try to find a block that can be removed, and if not, try to replace a pair with a single block?
        Actually, the standard local search for set cover minimization is deletion.
        A more powerful move is "exchange": remove one block, and if coverage is lost,
        try to add a different block that covers more of the lost coverage?
        
        Let's implement a simple "2-for-1" or "1-for-1" exchange search.
        Given the time, deletion is the most effective.
        However, we can try to *add* a block that allows removing two others.
        
        Let's stick to deletion + random restarts for now, as it's robust.
        But let's add a "perturbation" step: add random blocks, then delete.
        """
        return blocks_indices

    # Main Execution
    best_blocks = []
    best_len = float('inf')
    
    # 1. Run Greedy with multiple seeds and different noise levels
    # Noise helps escape deterministic traps
    seeds_to_try = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for seed in seeds_to_try:
        if check_time(15.0):
            break
            
        # Vary noise scale
        noise = 0.0 if seed % 2 == 0 else 0.5
        
        blocks = greedy_cover(seed=seed, noise_scale=noise)
        
        if not blocks:
            continue
            
        # Local Search (Deletion)
        blocks = local_search_deletion(blocks, time_limit=5.0, random_seed=seed)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
            
        # 2. Iterated Local Search: Perturb best solution
        # Add a few random blocks, then delete again
        if not check_time(10.0):
            current = list(best_blocks)
            current_set = set(current)
            
            # Add 3 random blocks not in current
            all_block_indices = list(range(num_blocks))
            candidates = [b for b in all_block_indices if b not in current_set]
            
            if candidates:
                # Use a dedicated RNG for perturbation
                perm = np.random.default_rng(seed + 100).permutation(len(candidates))
                add_count = min(3, len(candidates))
                to_add = [candidates[int(perm[i])] for i in range(add_count)]
                current.extend(to_add)
                
            # Local Search
            current = local_search_deletion(current, time_limit=5.0, random_seed=seed + 200)
            
            if len(current) < best_len:
                best_len = len(current)
                best_blocks = list(current)

    # 3. Final polish
    if best_blocks and not check_time(5.0):
        best_blocks = local_search_deletion(best_blocks, time_limit=5.0, random_seed=999)

    if not best_blocks:
        # Fallback
        # If we didn't find anything, try one more greedy without time pressure check inside?
        # Or just return empty (invalid). But greedy should always work.
        # Let's try a very fast greedy just in case.
        blocks = greedy_cover(seed=42, noise_scale=0.0)
        if blocks:
            return [list(all_blocks[i]) for i in blocks]
        return []

    return [list(all_blocks[i]) for i in best_blocks]
