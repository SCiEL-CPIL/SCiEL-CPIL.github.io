import itertools
import time
import random
import numpy as np
from collections import defaultdict
from itertools import combinations

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # Precomputation
    all_triplets = list(combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    id_to_triplet = [None] * n_triplets
    for i, trip in enumerate(all_triplets):
        id_to_triplet[i] = trip

    # Precompute triplet ids for each block (cached)
    block_triplet_cache = {}
    def get_block_triplet_ids(block):
        key = tuple(block)
        if key in block_triplet_cache:
            return block_triplet_cache[key]
        ids = frozenset(triplet_to_id[triplet] for triplet in combinations(block, t))
        block_triplet_cache[key] = ids
        return ids

    # Precompute mask for a block
    def get_block_mask(block):
        mask = 0
        for triplet in combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
        return mask

    # Enumerate blocks containing a set of points (limited)
    def blocks_containing(points, max_blocks=None):
        points = frozenset(points)
        if len(points) > k:
            return
        remaining = [p for p in range(v) if p not in points]
        need = k - len(points)
        if need < 0:
            return
        count = 0
        for comb in combinations(remaining, need):
            yield tuple(sorted(points | set(comb)))
            count += 1
            if max_blocks and count >= max_blocks:
                return

    # Greedy Construction
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = (1 << n_triplets) - 1
        blocks = []

        while uncovered:
            # Pick a random uncovered triplet
            idx = -1
            for _ in range(15):
                cand = random.randint(0, n_triplets - 1)
                if (uncovered >> cand) & 1:
                    idx = cand
                    break
            if idx == -1:
                lsb = uncovered & -uncovered
                idx = lsb.bit_length() - 1

            start_triplet = id_to_triplet[idx]

            # Find best block containing start_triplet
            best_block = None
            best_ratio = -1
            best_mask = 0
            best_gain = 0

            for block in blocks_containing(start_triplet, max_blocks=5000):
                b_mask = get_block_mask(block)
                gain = (b_mask & uncovered).bit_count()
                cost = b_mask.bit_count()
                ratio = gain / max(cost, 1)

                if ratio > best_ratio or (ratio == best_ratio and gain > best_gain):
                    best_ratio = ratio
                    best_block = block
                    best_mask = b_mask
                    best_gain = gain

            blocks.append(best_block)
            uncovered &= ~best_mask

        return blocks

    # Local Search Optimizer
    def optimize(blocks, time_limit):
        # Maintain counts
        counts = np.zeros(n_triplets, dtype=np.int32)

        def recalc_counts(b_list):
            counts[:] = 0
            for b in b_list:
                for triplet in combinations(b, t):
                    counts[triplet_to_id[triplet]] += 1

        recalc_counts(blocks)

        # Helper to check if a block is redundant
        def is_redundant(block):
            for triplet in combinations(block, t):
                if counts[triplet_to_id[triplet]] == 1:
                    return False
            return True

        no_improve_count = 0
        max_no_improve = 80

        while time.time() < time_limit:
            improved = False
            n = len(blocks)
            if n == 0:
                break

            # 1. Try Remove Redundant
            for i in range(n):
                b = blocks[i]
                if is_redundant(b):
                    for triplet in combinations(b, t):
                        counts[triplet_to_id[triplet]] -= 1
                    blocks.pop(i)
                    improved = True
                    break

            # 2. Try 2-for-1 with limited search
            if not improved:
                n = len(blocks)
                unique_loads = []
                for i in range(n):
                    b = blocks[i]
                    load = 0
                    for triplet in combinations(b, t):
                        if counts[triplet_to_id[triplet]] == 1:
                            load += 1
                    unique_loads.append((load, i))

                unique_loads.sort(reverse=True)
                tried_pairs = set()
                max_pairs_to_try = 15

                for load1, i in unique_loads[:10]:
                    if load1 == 0:
                        continue
                    for load2, j in unique_loads[:10]:
                        if i == j:
                            continue
                        pair_key = (min(i, j), max(i, j))
                        if pair_key in tried_pairs:
                            continue
                        tried_pairs.add(pair_key)
                        if len(tried_pairs) > max_pairs_to_try:
                            break

                        b1 = blocks[i]
                        b2 = blocks[j]

                        ids_b1 = get_block_triplet_ids(b1)
                        ids_b2 = get_block_triplet_ids(b2)
                        all_involved = ids_b1 | ids_b2

                        needed_triplets = []
                        points_needed = set()
                        for tid in all_involved:
                            if counts[tid] == 1:
                                needed_triplets.append(tid)
                                points_needed.update(id_to_triplet[tid])

                        if not needed_triplets:
                            # Both redundant
                            for triplet in combinations(b2, t):
                                counts[triplet_to_id[triplet]] -= 1
                            blocks.pop(j)
                            for triplet in combinations(b1, t):
                                counts[triplet_to_id[triplet]] -= 1
                            blocks.pop(i)
                            improved = True
                            break

                        if len(points_needed) > k:
                            continue

                        found_replacement = False
                        for new_block in blocks_containing(points_needed, max_blocks=2000):
                            if set(new_block) == set(b1) or set(new_block) == set(b2):
                                continue

                            new_ids = get_block_triplet_ids(new_block)
                            covers = all(tid in new_ids for tid in needed_triplets)

                            if covers:
                                for triplet in combinations(b2, t):
                                    counts[triplet_to_id[triplet]] -= 1
                                blocks.pop(j)
                                for triplet in combinations(b1, t):
                                    counts[triplet_to_id[triplet]] -= 1
                                blocks.pop(i)

                                blocks.append(new_block)
                                for triplet in combinations(new_block, t):
                                    counts[triplet_to_id[triplet]] += 1

                                improved = True
                                found_replacement = True
                                break

                        if improved:
                            break
                        if time.time() > time_limit - 2:
                            break

            # 3. Try 1-for-1 replacement (Shift)
            if not improved:
                n = len(blocks)
                if n == 0:
                    break

                best_i = -1
                best_load = -1
                for i in range(n):
                    b = blocks[i]
                    load = 0
                    for triplet in combinations(b, t):
                        if counts[triplet_to_id[triplet]] == 1:
                            load += 1
                    if load > best_load:
                        best_load = load
                        best_i = i

                if best_i == -1 or best_load == 0:
                    continue

                i = best_i
                b1 = blocks[i]
                ids_b1 = get_block_triplet_ids(b1)

                needed_triplets = []
                points_needed = set()
                for tid in ids_b1:
                    if counts[tid] == 1:
                        needed_triplets.append(tid)
                        points_needed.update(id_to_triplet[tid])

                if not needed_triplets:
                    continue

                if len(points_needed) > k:
                    continue

                for new_block in blocks_containing(points_needed, max_blocks=3000):
                    if set(new_block) == set(b1):
                        continue
                    new_ids = get_block_triplet_ids(new_block)
                    covers = all(tid in new_ids for tid in needed_triplets)
                    if covers:
                        for triplet in combinations(b1, t):
                            counts[triplet_to_id[triplet]] -= 1
                        blocks[i] = new_block
                        for triplet in combinations(new_block, t):
                            counts[triplet_to_id[triplet]] += 1
                        improved = True
                        break

            # 4. Try multi-point swap moves
            if not improved:
                n = len(blocks)
                if n >= 2:
                    # Try random 2-point swaps between two blocks
                    for _ in range(3):
                        i, j = random.sample(range(n), 2)
                        b1 = list(blocks[i])
                        b2 = list(blocks[j])
                        
                        # Try swapping 2 points from b1 with 2 points from b2
                        for p1a, p1b in combinations(b1, 2):
                            for p2a, p2b in combinations(b2, 2):
                                if p1a == p2a or p1a == p2b or p1b == p2a or p1b == p2b:
                                    continue
                                new_b1 = [x if x not in (p1a, p1b) else (p2a if x == p1a else p2b) for x in b1]
                                new_b2 = [x if x not in (p2a, p2b) else (p1a if x == p2a else p1b) for x in b2]
                                
                                if len(set(new_b1)) < k or len(set(new_b2)) < k:
                                    continue
                                
                                new_b1.sort()
                                new_b2.sort()
                                
                                old_ids_b1 = get_block_triplet_ids(b1)
                                old_ids_b2 = get_block_triplet_ids(b2)
                                new_ids_b1 = get_block_triplet_ids(tuple(new_b1))
                                new_ids_b2 = get_block_triplet_ids(tuple(new_b2))
                                
                                unique_b1 = [tid for tid in old_ids_b1 if counts[tid] == 1]
                                unique_b2 = [tid for tid in old_ids_b2 if counts[tid] == 1]
                                
                                if all(tid in new_ids_b1 or tid in new_ids_b2 for tid in unique_b1) and \
                                   all(tid in new_ids_b2 or tid in new_ids_b1 for tid in unique_b2):
                                    for triplet in combinations(b1, t):
                                        counts[triplet_to_id[triplet]] -= 1
                                    for triplet in combinations(b2, t):
                                        counts[triplet_to_id[triplet]] -= 1
                                    blocks[i] = new_b1
                                    blocks[j] = new_b2
                                    for triplet in combinations(new_b1, t):
                                        counts[triplet_to_id[triplet]] += 1
                                    for triplet in combinations(new_b2, t):
                                        counts[triplet_to_id[triplet]] += 1
                                    improved = True
                                    break
                        if improved:
                            break
                        if time.time() > time_limit - 1:
                            break

            if not improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
            else:
                no_improve_count = 0

        return blocks

    best_blocks = None
    best_len = float('inf')

    n_restarts = 12

    for seed in range(n_restarts):
        if time.time() > deadline - 8:
            break

        init_blocks = greedy_construct(seed)
        opt_blocks = optimize(init_blocks, deadline - 8)

        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in opt_blocks:
            for triplet in combinations(b, t):
                counts[triplet_to_id[triplet]] += 1

        if np.min(counts) > 0:
            if len(opt_blocks) < best_len:
                best_len = len(opt_blocks)
                best_blocks = opt_blocks
                # If we've found 76 or better, do one more polish pass
                if best_len <= 76:
                    opt_blocks = optimize(opt_blocks, deadline - 3)
                    counts2 = np.zeros(n_triplets, dtype=np.int32)
                    for b in opt_blocks:
                        for triplet in combinations(b, t):
                            counts2[triplet_to_id[triplet]] += 1
                    if np.min(counts2) > 0 and len(opt_blocks) < best_len:
                        best_len = len(opt_blocks)
                        best_blocks = opt_blocks

    if best_blocks is None:
        best_blocks = greedy_construct(0)

    return best_blocks

if __name__ == "__main__":
    pass
