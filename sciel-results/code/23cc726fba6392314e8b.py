import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 132.0  # safety margin under the 150s kill

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}

    def get_id(triplet_tuple):
        return triplet_to_id[triplet_tuple]

    # Precompute for each (t-1)-subset the list of t-subsets (triplet ids) containing it.
    # For t=3 this maps each pair to the triplets containing that pair.
    sub_to_triplets = {}
    for tid, triplet in enumerate(all_triplets):
        for sub in itertools.combinations(triplet, t - 1):
            sub_to_triplets.setdefault(sub, []).append(tid)

    # Precompute the list of t-subsets containing each point (for 1-for-1 swap).
    point_to_triplets = [[] for _ in range(v)]
    for tid, triplet in enumerate(all_triplets):
        for p in triplet:
            point_to_triplets[p].append(tid)

    def greedy_construct(seed=0, bias=0.0):
        rng = random.Random(seed)
        uncovered = np.ones(n_triplets, dtype=bool)
        blocks = []

        while uncovered.any():
            start_tid = int(rng.choice(np.flatnonzero(uncovered)))
            block = list(all_triplets[start_tid])
            block_set = set(block)

            while len(block) < k:
                # (t-1)-subsets of the current block
                subs_in_block = list(itertools.combinations(block, t - 1))
                best_point = -1
                max_gain = -1
                second_best_point = -1
                second_best_gain = -1

                for p in range(v):
                    if p in block_set:
                        continue
                    gain = 0
                    for sub in subs_in_block:
                        for tid in sub_to_triplets[sub]:
                            if uncovered[tid]:
                                gain += 1
                    if gain > max_gain:
                        second_best_gain = max_gain
                        second_best_point = best_point
                        max_gain = gain
                        best_point = p
                    elif gain > second_best_gain:
                        second_best_gain = gain
                        second_best_point = p

                if max_gain == 0:
                    cands = [p for p in range(v) if p not in block_set]
                    best_point = rng.choice(cands)
                else:
                    r = rng.random()
                    if r < (1.0 - bias):
                        pass
                    elif second_best_point != -1 and r < (1.0 - bias) + bias * 0.5:
                        best_point = second_best_point
                    else:
                        cands = [p for p in range(v) if p not in block_set]
                        best_point = rng.choice(cands)

                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            blocks.append(block)
            for triplet in itertools.combinations(block, t):
                uncovered[get_id(triplet)] = False

        return blocks

    # 1. Initial construction with diverse seeds/biases
    best_blocks = None
    best_count = float('inf')

    for seed in range(24):
        if time.time() > deadline - 45.0:
            break
        bias = 0.0 if seed < 12 else 0.15 + 0.02 * (seed - 12)
        blocks = greedy_construct(seed, bias)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_construct(0, 0.0)

    # 2. Local search
    count = np.zeros(n_triplets, dtype=np.int32)

    def update_counts(blocks, delta=1):
        for block in blocks:
            for triplet in itertools.combinations(block, t):
                count[get_id(triplet)] += delta

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

    def try_swap_1_for_1(blocks):
        n = len(blocks)
        for i in range(n):
            if time.time() > deadline - 2.0:
                return blocks, False
            b = blocks[i]
            # Triplets covered only by b (count == 1)
            critical_tids = [tid for tid in point_to_triplets[b[0]] if count[tid] == 1]
            # A point must be kept in the new block iff removing it would leave some
            # critical triplet with no other cover.
            must_keep = set()
            for p in b:
                other = [q for q in b if q != p]
                for tid in critical_tids:
                    if all(q in other for q in all_triplets[tid]):
                        must_keep.add(p)
                        break
            if len(must_keep) >= k:
                continue
            new_block = list(must_keep)
            while len(new_block) < k:
                p = random.randint(0, v - 1)
                if p not in new_block:
                    new_block.append(p)
            new_block.sort()
            if new_block == b:
                continue
            update_counts([b], -1)
            update_counts([new_block], 1)
            blocks[i] = new_block
            return blocks, True
        return blocks, False

    def try_swap_2_for_1(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False
        for i in range(n):
            if time.time() > deadline - 2.0:
                return blocks, False
            b1 = blocks[i]
            ids_b1 = set(get_id(tr) for tr in itertools.combinations(b1, t))
            for j in range(i + 1, n):
                if time.time() > deadline - 2.0:
                    return blocks, False
                b2 = blocks[j]
                ids_b2 = set(get_id(tr) for tr in itertools.combinations(b2, t))
                critical_points = set()
                for tid in ids_b1:
                    covered = (1 if tid in ids_b1 else 0) + (1 if tid in ids_b2 else 0)
                    if count[tid] == covered:
                        critical_points.update(all_triplets[tid])
                for tid in ids_b2:
                    if tid in ids_b1:
                        continue
                    if count[tid] == 1:
                        critical_points.update(all_triplets[tid])
                if not critical_points:
                    update_counts([b1, b2], -1)
                    return [x for idx, x in enumerate(blocks) if idx != i and idx != j], True
                if len(critical_points) <= k:
                    new_block = list(critical_points)
                    while len(new_block) < k:
                        p = random.randint(0, v - 1)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()
                    update_counts([b1, b2], -1)
                    update_counts([new_block], 1)
                    new_blocks = [x for idx, x in enumerate(blocks) if idx != i and idx != j]
                    new_blocks.append(new_block)
                    return new_blocks, True
        return blocks, False

    update_counts(best_blocks, 1)
    current_blocks = list(best_blocks)

    progress = True
    while progress and time.time() < deadline - 2.0:
        progress = False
        current_blocks, r1 = try_remove_redundant(current_blocks)
        if r1:
            progress = True
        if not progress:
            current_blocks, r2 = try_swap_1_for_1(current_blocks)
            if r2:
                progress = True
        if not progress:
            current_blocks, r3 = try_swap_2_for_1(current_blocks)
            if r3:
                progress = True

    current_blocks, _ = try_remove_redundant(current_blocks)

    # Safety: if somehow empty/invalid, fall back to a fresh greedy.
    if not current_blocks:
        current_blocks = greedy_construct(0, 0.0)

    return current_blocks

if __name__ == "__main__":
    pass
