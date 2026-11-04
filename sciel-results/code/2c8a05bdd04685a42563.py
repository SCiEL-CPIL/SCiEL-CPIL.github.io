import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Precompute all t-subsets (triplets)
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}

    def get_id(triplet_tuple):
        return triplet_to_id[triplet_tuple]

    # Precompute the list of triplets for each point
    # point_to_triplets[p] = list of triplet IDs that contain point p
    point_to_triplets = [[] for _ in range(v)]
    for tid, triplet in enumerate(all_triplets):
        for p in triplet:
            point_to_triplets[p].append(tid)

    # Precompute for each pair of points, the list of triplets containing both
    # This is useful for gain calculation: gain of adding p to block B is
    # sum over pairs (a,b) in B of |uncovered triplets containing {a,b,p}|
    # But actually, gain = number of uncovered triplets {x,y,p} where {x,y} subset of B
    # = sum over (t-1)-subsets S of B of |uncovered triplets containing S ∪ {p}|
    # For t=3, (t-1)=2, so pairs.
    # Let's precompute pair_to_triplets: for each pair (a,b), list of triplet IDs containing {a,b}
    pair_to_triplets = {}
    for tid, triplet in enumerate(all_triplets):
        for pair in itertools.combinations(triplet, t - 1):
            pair_to_triplets.setdefault(pair, []).append(tid)

    def greedy_construct(v, k, t, seed=0):
        rng = random.Random(seed)
        uncovered = set(range(n_triplets))
        blocks = []

        while uncovered:
            # Pick an uncovered triplet to start the block
            # Choose one that is "hard" to cover: e.g., one with few points that have high uncovered degree
            # Or just random
            start_triplet_id = rng.choice(list(uncovered))
            start_triplet = all_triplets[start_triplet_id]

            block = list(start_triplet)
            block_set = set(block)

            while len(block) < k:
                best_point = -1
                max_gain = -1
                candidates = [p for p in range(v) if p not in block_set]

                # Compute gain for each candidate
                # For t=3: gain(p) = sum over pairs (a,b) in block of |uncovered triplets containing {a,b,p}|
                # Precompute pairs in block
                pairs_in_block = list(itertools.combinations(block, t - 1))

                for p in candidates:
                    gain = 0
                    for pair in pairs_in_block:
                        # Triplets containing pair and p
                        # The triplet is sorted(pair + (p,))
                        # Look up in pair_to_triplets? No, pair_to_triplets maps pair -> triplets containing that pair.
                        # But we need triplets containing pair AND p.
                        # Actually, for t=3, a triplet is determined by 3 points.
                        # If we have pair (a,b) and point p, the triplet is {a,b,p}.
                        # We just need to check if that specific triplet is uncovered.
                        # So gain is just counting how many pairs (a,b) in block such that {a,b,p} is uncovered.
                        triplet = tuple(sorted(pair + (p,)))
                        tid = get_id(triplet)
                        if tid in uncovered:
                            gain += 1

                    if gain > max_gain:
                        max_gain = gain
                        best_point = p

                if max_gain == 0:
                    best_point = rng.choice(candidates)

                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            blocks.append(block)

            # Update uncovered
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                uncovered.discard(tid)

        return blocks

    # 1. Initial Construction
    best_blocks = None
    best_count = float('inf')

    # Try multiple seeds to find a good starting point
    num_seeds = 20
    for seed in range(num_seeds):
        if time.time() > deadline - 10:  # Leave time for optimization
            break
        blocks = greedy_construct(v, k, t, seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_construct(v, k, t, 0)

    # 2. Local Search Optimization
    count = np.zeros(n_triplets, dtype=np.int32)

    def update_counts(blocks, delta=1):
        for block in blocks:
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                count[tid] += delta

    update_counts(best_blocks, 1)
    current_blocks = list(best_blocks)

    def is_redundant(block):
        for triplet in itertools.combinations(block, t):
            if count[get_id(triplet)] == 1:
                return False
        return True

    def try_remove_redundant(blocks):
        removed = False
        new_blocks = []
        for b in blocks:
            if is_redundant(b):
                update_counts([b], -1)
                removed = True
            else:
                new_blocks.append(b)
        return new_blocks, removed

    def try_swap_2_for_1(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False

        for i in range(n):
            if time.time() > deadline:
                return blocks, False
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return blocks, False

                b1 = blocks[i]
                b2 = blocks[j]

                # Precompute set of triplet IDs for b1 and b2
                ids_b1 = set()
                for triplet in itertools.combinations(b1, t):
                    ids_b1.add(get_id(triplet))

                ids_b2 = set()
                for triplet in itertools.combinations(b2, t):
                    ids_b2.add(get_id(triplet))

                check_ids = ids_b1 | ids_b2

                u_need = []
                for tid in check_ids:
                    c = count[tid]
                    covered_by_b1 = 1 if tid in ids_b1 else 0
                    covered_by_b2 = 1 if tid in ids_b2 else 0
                    if c == covered_by_b1 + covered_by_b2:
                        u_need.append(all_triplets[tid])

                if not u_need:
                    update_counts([b1, b2], -1)
                    return [b for idx, b in enumerate(blocks) if idx != i and idx != j], True

                # Check if a single block can cover u_need
                p_need = set()
                for triplet in u_need:
                    p_need.update(triplet)

                if len(p_need) <= k:
                    new_block = list(p_need)
                    while len(new_block) < k:
                        p = random.randint(0, v - 1)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()

                    update_counts([b1, b2], -1)
                    update_counts([new_block], 1)

                    new_blocks = []
                    for idx, b in enumerate(blocks):
                        if idx != i and idx != j:
                            new_blocks.append(b)
                    new_blocks.append(new_block)

                    return new_blocks, True

        return blocks, False

    # Try 1-for-1 swap: replace block i with a new block that covers all triplets uniquely covered by block i
    def try_swap_1_for_1(blocks):
        n = len(blocks)
        if n < 1:
            return blocks, False

        for i in range(n):
            if time.time() > deadline:
                return blocks, False

            b1 = blocks[i]
            ids_b1 = set()
            for triplet in itertools.combinations(b1, t):
                ids_b1.add(get_id(triplet))

            u_need = []
            for tid in ids_b1:
                if count[tid] == 1:
                    u_need.append(all_triplets[tid])

            if not u_need:
                # Block is redundant, handled by remove
                continue

            p_need = set()
            for triplet in u_need:
                p_need.update(triplet)

            if len(p_need) <= k:
                # Can we form a block that covers u_need?
                # The new block must contain p_need.
                # It might cover additional triplets, which is fine.
                new_block = list(p_need)
                while len(new_block) < k:
                    p = random.randint(0, v - 1)
                    if p not in new_block:
                        new_block.append(p)
                new_block.sort()

                # Check if new_block is different from b1
                if set(new_block) == set(b1):
                    continue

                # Check if new_block covers all of u_need
                # It does by construction (contains p_need)
                # But we need to make sure it doesn't uncover anything else
                # Since we are replacing b1 with new_block, and new_block covers u_need,
                # and u_need are the only triplets uniquely covered by b1,
                # the rest are covered by other blocks.
                # So the swap is valid.

                update_counts([b1], -1)
                update_counts([new_block], 1)

                new_blocks = []
                for idx, b in enumerate(blocks):
                    if idx != i:
                        new_blocks.append(b)
                new_blocks.append(new_block)

                return new_blocks, True

        return blocks, False

    # Main Optimization Loop
    progress_made = True
    while progress_made and time.time() < deadline - 2:
        progress_made = False

        # Try removing redundants
        current_blocks, removed = try_remove_redundant(current_blocks)
        if removed:
            progress_made = True

        # Try swapping 2 for 1
        if not progress_made:
            current_blocks, swapped = try_swap_2_for_1(current_blocks)
            if swapped:
                progress_made = True

        # Try swapping 1 for 1
        if not progress_made:
            current_blocks, swapped = try_swap_1_for_1(current_blocks)
            if swapped:
                progress_made = True

    return current_blocks

# Test/Entry
if __name__ == "__main__":
    pass
