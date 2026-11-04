"""Strong-greedy with bitset acceleration + enhanced local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    # Precompute all t-subsets
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_id = {ts: i for i, ts in enumerate(all_tsets)}

    def get_id(ts_tuple):
        return tset_to_id[ts_tuple]

    # For each pair (t-1 subset), store list of t-set ids containing it
    t1_to_tsets = {}
    for tid, ts in enumerate(all_tsets):
        for sub in itertools.combinations(ts, t - 1):
            t1_to_tsets.setdefault(sub, []).append(tid)

    # Bitset representation for uncovered sets
    # Each t-set is a bit in a numpy uint64 array
    n_words = (n_tsets + 63) // 64
    uncovered_bits = np.ones(n_words, dtype=np.uint64)

    def set_bit(arr, idx):
        arr[idx >> 6] |= (1 << (idx & 63))

    def clear_bit(arr, idx):
        arr[idx >> 6] &= ~(1 << (idx & 63))

    def is_set(arr, idx):
        return (arr[idx >> 6] >> (idx & 63)) & 1

    def count_bits(arr):
        return int(np.unpackbits(arr.view(np.uint8), axis=1).sum())

    def bitset_popcount(arr):
        return int(np.unpackbits(arr.view(np.uint8), axis=1).sum())

    def greedy_construct(seed=0, strategy='strong'):
        """Greedy construction.
        strategy: 'strong' - pick block covering most uncovered triples
                  'first' - pick first uncovered triple, build around it
                  'random' - randomized
        """
        rng = random.Random(seed)
        uncov = uncovered_bits.copy()
        blocks = []
        n_uncov = n_tsets

        while n_uncov > 0:
            if time.time() > deadline - 20:
                break

            if strategy == 'strong':
                # Pick the block covering the most uncovered t-sets
                # Sample a subset of candidate blocks for speed
                best_block = None
                best_gain = -1
                best_gain_bits = None

                # Generate candidate blocks: sample random k-subsets
                # But focus on blocks that contain some uncovered t-set
                # Strategy: pick a random uncovered t-set, then try to build optimal block around it
                # Also try some fully random blocks
                n_candidates = 30

                # Find some uncovered t-sets to anchor on
                uncov_ids = []
                for w in range(n_words):
                    word = uncov[w]
                    while word:
                        bit = (word & -word).bit_length() - 1
                        tid = w * 64 + bit
                        if tid < n_tsets:
                            uncov_ids.append(tid)
                        word &= word - 1
                        if len(uncov_ids) >= 20:
                            break
                    if len(uncov_ids) >= 20:
                        break

                if not uncov_ids:
                    break

                # Try blocks anchored on uncovered t-sets
                anchors = rng.sample(uncov_ids, min(len(uncov_ids), 10))
                for anchor_tid in anchors:
                    anchor_ts = all_tsets[anchor_tid]
                    block = list(anchor_ts)
                    block_set = set(block)

                    while len(block) < k:
                        best_point = -1
                        best_gain = -1
                        candidates = [p for p in range(v) if p not in block_set]
                        subs_in_block = list(itertools.combinations(block, t - 1))

                        for p in candidates:
                            gain = 0
                            for sub in subs_in_block:
                                ts = tuple(sorted(sub + (p,)))
                                tid = get_id(ts)
                                if is_set(uncov, tid):
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_point = p

                        if best_point == -1:
                            best_point = rng.choice(candidates)
                        block.append(best_point)
                        block_set.add(best_point)

                    block.sort()
                    # Count gain of this block
                    gain = 0
                    for ts in itertools.combinations(block, t):
                        tid = get_id(ts)
                        if is_set(uncov, tid):
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block = block
                        best_gain_bits = None

                # Also try a few fully random blocks
                for _ in range(5):
                    block = sorted(rng.sample(range(v), k))
                    gain = 0
                    for ts in itertools.combinations(block, t):
                        tid = get_id(ts)
                        if is_set(uncov, tid):
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block = block

                if best_block is None:
                    # Fallback: pick first uncovered
                    first_tid = uncov_ids[0]
                    block = list(all_tsets[first_tid])
                    block_set = set(block)
                    while len(block) < k:
                        block.append(rng.choice([p for p in range(v) if p not in block_set]))
                        block_set.add(block[-1])
                    best_block = sorted(block)

                blocks.append(best_block)

                # Update uncovered
                for ts in itertools.combinations(best_block, t):
                    tid = get_id(ts)
                    if is_set(uncov, tid):
                        clear_bit(uncov, tid)
                        n_uncov -= 1

            elif strategy == 'first':
                # Pick first uncovered t-set
                first_tid = -1
                for w in range(n_words):
                    word = uncov[w]
                    if word:
                        bit = (word & -word).bit_length() - 1
                        first_tid = w * 64 + bit
                        break
                if first_tid < 0:
                    break

                block = list(all_tsets[first_tid])
                block_set = set(block)

                while len(block) < k:
                    best_point = -1
                    best_gain = -1
                    candidates = [p for p in range(v) if p not in block_set]
                    subs_in_block = list(itertools.combinations(block, t - 1))

                    for p in candidates:
                        gain = 0
                        for sub in subs_in_block:
                            ts = tuple(sorted(sub + (p,)))
                            tid = get_id(ts)
                            if is_set(uncov, tid):
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_point = p

                    if best_point == -1:
                        best_point = rng.choice(candidates)
                    block.append(best_point)
                    block_set.add(best_point)

                block.sort()
                blocks.append(block)

                for ts in itertools.combinations(block, t):
                    tid = get_id(ts)
                    if is_set(uncov, tid):
                        clear_bit(uncov, tid)
                        n_uncov -= 1

            else:  # random
                block = sorted(rng.sample(range(v), k))
                blocks.append(block)
                for ts in itertools.combinations(block, t):
                    tid = get_id(ts)
                    if is_set(uncov, tid):
                        clear_bit(uncov, tid)
                        n_uncov -= 1

        return blocks

    # Phase 1: Multiple greedy restarts
    best_blocks = None
    best_count = float('inf')

    strategies = ['strong', 'strong', 'first', 'strong', 'first', 'strong', 'random', 'strong']
    num_seeds = 30
    for seed in range(num_seeds):
        if time.time() > deadline - 30:
            break
        strategy = strategies[seed % len(strategies)]
        blocks = greedy_construct(seed, strategy)

        # Verify completeness
        if len(blocks) > 0:
            covered = np.zeros(n_tsets, dtype=np.int32)
            for block in blocks:
                for ts in itertools.combinations(block, t):
                    covered[get_id(ts)] += 1
            if (covered > 0).all():
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_construct(0, 'first')

    # Phase 2: Local search
    count = np.zeros(n_tsets, dtype=np.int32)
    for block in best_blocks:
        for ts in itertools.combinations(block, t):
            count[get_id(ts)] += 1

    current_blocks = list(best_blocks)

    def is_redundant(block):
        for ts in itertools.combinations(block, t):
            if count[get_id(ts)] == 1:
                return False
        return True

    def try_remove_redundant(blocks):
        removed = False
        new_blocks = []
        for b in blocks:
            if is_redundant(b):
                for ts in itertools.combinations(b, t):
                    count[get_id(ts)] -= 1
                removed = True
            else:
                new_blocks.append(b)
        return new_blocks, removed

    def try_swap_2_for_1(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False

        # Precompute block t-set IDs
        block_ids = []
        for b in blocks:
            ids = set()
            for ts in itertools.combinations(b, t):
                ids.add(get_id(ts))
            block_ids.append(ids)

        # Try pairs with high overlap
        pair_scores = []
        max_pairs = min(n * (n - 1) // 2, 2000)
        for i in range(n):
            for j in range(i + 1, n):
                overlap = len(block_ids[i] & block_ids[j])
                total = len(block_ids[i] | block_ids[j])
                if total > 0:
                    score = overlap / total
                    unique_i = len(block_ids[i] - block_ids[j])
                    unique_j = len(block_ids[j] - block_ids[i])
                    if unique_i < 5 or unique_j < 5:
                        score += 0.5
                    pair_scores.append((score, i, j))

        pair_scores.sort(reverse=True)
        max_to_try = min(len(pair_scores), 300)

        for score, i, j in pair_scores[:max_to_try]:
            if time.time() > deadline - 10:
                break

            b1 = blocks[i]
            b2 = blocks[j]
            ids_b1 = block_ids[i]
            ids_b2 = block_ids[j]
            check_ids = ids_b1 | ids_b2

            u_need = []
            for tid in check_ids:
                c = count[tid]
                cov = (1 if tid in ids_b1 else 0) + (1 if tid in ids_b2 else 0)
                if c == cov:
                    u_need.append(tid)

            if not u_need:
                for ts in itertools.combinations(b1, t):
                    count[get_id(ts)] -= 1
                for ts in itertools.combinations(b2, t):
                    count[get_id(ts)] -= 1
                return [b for idx, b in enumerate(blocks) if idx != i and idx != j], True

            p_need = set()
            for tid in u_need:
                p_need.update(all_tsets[tid])

            if len(p_need) > k:
                continue

            new_block = list(p_need)
            new_block_set = set(new_block)
            rng = random.Random(i * 1000 + j)
            candidates = [p for p in range(v) if p not in new_block_set]

            for _ in range(k - len(new_block)):
                if not candidates:
                    break
                best_p = -1
                best_gain = -1
                subs_in_block = list(itertools.combinations(new_block, t - 1))

                for p in candidates:
                    gain = 0
                    for sub in subs_in_block:
                        ts = tuple(sorted(sub + (p,)))
                        tid = get_id(ts)
                        if tid not in ids_b1 and tid not in ids_b2:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p

                if best_p == -1 or best_gain == 0:
                    best_p = rng.choice(candidates)

                new_block.append(best_p)
                new_block_set.add(best_p)
                candidates.remove(best_p)

            new_block.sort()

            new_ids = set()
            for ts in itertools.combinations(new_block, t):
                new_ids.add(get_id(ts))

            if all(tid in new_ids for tid in u_need):
                for ts in itertools.combinations(b1, t):
                    count[get_id(ts)] -= 1
                for ts in itertools.combinations(b2, t):
                    count[get_id(ts)] -= 1
                for ts in itertools.combinations(new_block, t):
                    count[get_id(ts)] += 1
                new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                new_blocks.append(new_block)
                return new_blocks, True

        return blocks, False

    def try_swap_1_for_1(blocks):
        n = len(blocks)
        if n < 1:
            return blocks, False

        block_unique_counts = []
        for i in range(n):
            b = blocks[i]
            ids_b = set()
            for ts in itertools.combinations(b, t):
                ids_b.add(get_id(ts))
            unique = sum(1 for tid in ids_b if count[tid] == 1)
            block_unique_counts.append((unique, i))

        block_unique_counts.sort()
        max_to_try = min(n, 80)

        for unique_count, i in block_unique_counts[:max_to_try]:
            if time.time() > deadline - 10:
                break

            b = blocks[i]
            ids_b = set()
            for ts in itertools.combinations(b, t):
                ids_b.add(get_id(ts))

            u_need = [tid for tid in ids_b if count[tid] == 1]

            if not u_need:
                for ts in itertools.combinations(b, t):
                    count[get_id(ts)] -= 1
                return [x for idx, x in enumerate(blocks) if idx != i], True

            p_need = set()
            for tid in u_need:
                p_need.update(all_tsets[tid])

            if len(p_need) > k:
                continue

            new_block = list(p_need)
            new_block_set = set(new_block)
            rng = random.Random(i * 7919)
            candidates = [p for p in range(v) if p not in new_block_set]

            for _ in range(k - len(new_block)):
                if not candidates:
                    break
                best_p = -1
                best_gain = -1
                subs_in_block = list(itertools.combinations(new_block, t - 1))

                for p in candidates:
                    gain = 0
                    for sub in subs_in_block:
                        ts = tuple(sorted(sub + (p,)))
                        tid = get_id(ts)
                        if tid not in ids_b:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p

                if best_p == -1 or best_gain == 0:
                    best_p = rng.choice(candidates)

                new_block.append(best_p)
                new_block_set.add(best_p)
                candidates.remove(best_p)

            new_block.sort()

            if new_block != b:
                new_ids = set()
                for ts in itertools.combinations(new_block, t):
                    new_ids.add(get_id(ts))

                if all(tid in new_ids for tid in u_need):
                    for ts in itertools.combinations(b, t):
                        count[get_id(ts)] -= 1
                    for ts in itertools.combinations(new_block, t):
                        count[get_id(ts)] += 1
                    new_blocks = [x for idx, x in enumerate(blocks) if idx != i]
                    new_blocks.append(new_block)
                    return new_blocks, True

        return blocks, False

    def try_perturbation(blocks):
        n = len(blocks)
        if n < 1:
            return blocks, False

        rng = random.Random(int(time.time() * 1000) % 100000)

        for _attempt in range(min(3, n)):
            if time.time() > deadline - 10:
                break

            i = rng.randrange(n)
            b = blocks[i]

            ids_b = set()
            for ts in itertools.combinations(b, t):
                ids_b.add(get_id(ts))

            u_need = [tid for tid in ids_b if count[tid] == 1]

            num_replace = rng.choice([1, 2, 3])
            points_to_replace = rng.sample(b, min(num_replace, len(b)))
            replace_set = set(points_to_replace)

            kept = [p for p in b if p not in replace_set]
            candidates = [p for p in range(v) if p not in set(kept)]

            if len(candidates) < num_replace:
                continue

            new_points = rng.sample(candidates, num_replace)
            new_block = sorted(kept + new_points)

            if new_block == b:
                continue

            new_ids = set()
            for ts in itertools.combinations(new_block, t):
                new_ids.add(get_id(ts))

            if all(tid in new_ids for tid in u_need):
                for ts in itertools.combinations(b, t):
                    count[get_id(ts)] -= 1
                for ts in itertools.combinations(new_block, t):
                    count[get_id(ts)] += 1
                new_blocks = [x for idx, x in enumerate(blocks) if idx != i]
                new_blocks.append(new_block)
                return new_blocks, True

        return blocks, False

    # Iterative local search
    improved = True
    max_rounds = 200
    stall_count = 0
    for _ in range(max_rounds):
        if time.time() > deadline - 15:
            break
        if not improved:
            stall_count += 1
            if stall_count >= 3:
                if time.time() < deadline - 20:
                    new_blocks, pert = try_perturbation(current_blocks)
                    if pert:
                        current_blocks = new_blocks
                        improved = True
                        stall_count = 0
                        for _pass in range(3):
                            if time.time() > deadline - 15:
                                break
                            new_blocks, rem = try_remove_redundant(current_blocks)
                            if rem:
                                current_blocks = new_blocks
                                improved = True
                            else:
                                break
                if not improved:
                    break
                continue
            else:
                break
        else:
            stall_count = 0
        improved = False

        for _pass in range(3):
            if time.time() > deadline - 15:
                break
            new_blocks, rem = try_remove_redundant(current_blocks)
            if rem:
                current_blocks = new_blocks
                improved = True
            else:
                break

        new_blocks, swp = try_swap_2_for_1(current_blocks)
        if swp:
            current_blocks = new_blocks
            improved = True
            for _pass in range(2):
                if time.time() > deadline - 15:
                    break
                new_blocks, rem = try_remove_redundant(current_blocks)
                if rem:
                    current_blocks = new_blocks
                    improved = True
                else:
                    break

        new_blocks, swp = try_swap_1_for_1(current_blocks)
        if swp:
            current_blocks = new_blocks
            improved = True
            for _pass in range(2):
                if time.time() > deadline - 15:
                    break
                new_blocks, rem = try_remove_redundant(current_blocks)
                if rem:
                    current_blocks = new_blocks
                    improved = True
                else:
                    break

    # Final verification
    covered = np.zeros(n_tsets, dtype=np.int32)
    for block in current_blocks:
        for ts in itertools.combinations(block, t):
            covered[get_id(ts)] += 1

    if (covered > 0).all():
        # Final polish
        if time.time() < deadline - 5:
            for _pass in range(5):
                if time.time() > deadline - 3:
                    break
                new_blocks, rem = try_remove_redundant(current_blocks)
                if rem:
                    current_blocks = new_blocks
                else:
                    break
        return current_blocks
    else:
        return best_blocks
# EVOLVE-BLOCK-END
