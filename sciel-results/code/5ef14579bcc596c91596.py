import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    # Precompute all t-subsets and map to IDs
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}

    # Precompute all k-subsets (blocks) as tuples
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Build bitmask for each block: bit i set if block covers triplet i
    block_masks = [0] * n_blocks
    for idx, block in enumerate(all_blocks):
        mask = 0
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
        block_masks[idx] = mask

    # ---------- Fast Greedy Construction ----------
    def greedy_fast(seed=0):
        rng = random.Random(seed)
        is_uncovered = np.ones(n_triplets, dtype=bool)
        uncovered_list = list(range(n_triplets))
        pos = np.arange(n_triplets, dtype=np.int32)
        blocks = []

        while uncovered_list:
            if time.time() > deadline - 20:
                break

            # Pick a random uncovered triplet
            idx = rng.randrange(len(uncovered_list))
            start_tid = uncovered_list[idx]
            start_triplet = all_triplets[start_tid]

            block = list(start_triplet)
            block_set = set(block)

            # Greedily add points to maximize coverage of uncovered triplets
            while len(block) < k:
                best_point = -1
                max_gain = -1
                candidates = [p for p in range(v) if p not in block_set]

                # Precompute (t-1)-subsets of current block
                pairs_in_block = list(itertools.combinations(block, t - 1))

                for p in candidates:
                    gain = 0
                    for pair in pairs_in_block:
                        triplet = tuple(sorted(pair + (p,)))
                        tid = triplet_to_id[triplet]
                        if is_uncovered[tid]:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_point = p
                        if max_gain == len(pairs_in_block):
                            break

                if max_gain <= 0:
                    best_point = rng.choice(candidates)

                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            blocks.append(tuple(block))

            # Update uncovered
            for triplet in itertools.combinations(block, t):
                tid = triplet_to_id[triplet]
                if is_uncovered[tid]:
                    is_uncovered[tid] = False
                    cur_idx = pos[tid]
                    last_idx = len(uncovered_list) - 1
                    if cur_idx != last_idx:
                        last_tid = uncovered_list[last_idx]
                        uncovered_list[cur_idx] = last_tid
                        pos[last_tid] = cur_idx
                    uncovered_list.pop()

        return blocks

    # ---------- Local Search ----------
    def local_search(blocks, time_limit=10):
        """Perform redundancy removal and 1-for-1 swaps."""
        ls_start = time.time()
        ls_deadline = ls_start + time_limit

        # Build count array
        count = np.zeros(n_triplets, dtype=np.int32)

        def update_counts(block_list, delta):
            for b in block_list:
                for triplet in itertools.combinations(b, t):
                    tid = triplet_to_id[triplet]
                    count[tid] += delta

        # Initialize counts
        update_counts(blocks, 1)

        progress = True
        while progress and time.time() < ls_deadline:
            progress = False

            # Pass 1: Remove redundant blocks
            new_blocks = []
            for b in blocks:
                redundant = True
                for triplet in itertools.combinations(b, t):
                    tid = triplet_to_id[triplet]
                    if count[tid] == 1:
                        redundant = False
                        break
                if redundant:
                    update_counts([b], -1)
                    progress = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks

            # Pass 2: 1-for-1 swap
            n = len(blocks)
            if n >= 1 and time.time() < ls_deadline:
                # Precompute unique triplets for each block
                unique_triplets_per_block = []
                for b in blocks:
                    uniques = []
                    for triplet in itertools.combinations(b, t):
                        tid = triplet_to_id[triplet]
                        if count[tid] == 1:
                            uniques.append(tid)
                    unique_triplets_per_block.append(uniques)
                
                swapped = False
                for i in range(n):
                    if time.time() > ls_deadline:
                        break
                    if not unique_triplets_per_block[i]:
                        continue
                    
                    # Build mask of required triplets
                    req_mask = 0
                    for tid in unique_triplets_per_block[i]:
                        req_mask |= (1 << tid)
                    
                    # Find a candidate block that covers req_mask
                    found_idx = -1
                    for bi, mask in enumerate(block_masks):
                        if (mask & req_mask) == req_mask:
                            # Check if this block is already the one we are replacing
                            if all_blocks[bi] != blocks[i]:
                                found_idx = bi
                                break
                                
                    if found_idx != -1:
                        old_block = blocks[i]
                        new_block = all_blocks[found_idx]
                        
                        # Verify coverage (safety check)
                        valid = True
                        for tid in unique_triplets_per_block[i]:
                            triplet = all_triplets[tid]
                            if not all(p in new_block for p in triplet):
                                valid = False
                                break
                            
                        if valid:
                            update_counts([old_block], -1)
                            update_counts([new_block], 1)
                            blocks[i] = new_block
                            progress = True
                            swapped = True
                            break
                if swapped:
                    continue

        return blocks

    # ---------- Main ----------
    best_blocks = None
    best_count = float('inf')

    # Run greedy with a few seeds
    for seed in range(5):
        if time.time() > deadline - 30:
            break
        blocks = greedy_fast(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    # Local search on best
    if best_blocks is not None and time.time() < deadline - 10:
        best_blocks = local_search(best_blocks, time_limit=min(20, deadline - time.time() - 5))

    return best_blocks if best_blocks is not None else []

if __name__ == "__main__":
    pass
