"""Combined program: fast greedy construction + enhanced local search with block perturbation."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Precompute all t-subsets
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_id = {ts: i for i, ts in enumerate(all_tsets)}

    # Precompute (t-1)-subset to t-subsets mapping
    # For t=3: pair -> list of triplets containing that pair
    t1_to_tsets = {}
    for tid, ts in enumerate(all_tsets):
        for sub in itertools.combinations(ts, t - 1):
            t1_to_tsets.setdefault(sub, []).append(tid)

    def get_id(ts_tuple):
        return tset_to_id[ts_tuple]

    def greedy_construct(seed=0, strategy='max_gain'):
        """Greedy construction with different strategies.
        strategy: 'max_gain' (absolute), 'ratio' (gain/1, same for fixed k),
                  'ratio2' (gain / overlap_penalty), 'random' (randomized)
        """
        rng = random.Random(seed)
        uncovered = set(range(n_tsets))
        blocks = []

        while uncovered:
            # Pick a random uncovered t-set to start
            start_id = rng.choice(list(uncovered))
            start_ts = all_tsets[start_id]
            block = list(start_ts)
            block_set = set(block)

            while len(block) < k:
                best_point = -1
                best_score = -1
                candidates = [p for p in range(v) if p not in block_set]

                subs_in_block = list(itertools.combinations(block, t - 1))

                for p in candidates:
                    gain = 0
                    for sub in subs_in_block:
                        ts = tuple(sorted(sub + (p,)))
                        tid = get_id(ts)
                        if tid in uncovered:
                            gain += 1

                    if strategy == 'max_gain':
                        score = float(gain)
                    elif strategy == 'ratio':
                        # Penalize points that add little
                        score = gain / (1 + 0.1 * (k - len(block) - 1))
                    elif strategy == 'ratio2':
                        # Stronger penalty for late additions
                        progress = len(block) / k
                        score = gain / (1 + progress)
                    else:  # random
                        score = rng.random() * gain if gain > 0 else rng.random() * 0.1

                    # Add small random tie-breaker for diversity
                    score += rng.random() * 0.01

                    if score > best_score:
                        best_score = score
                        best_point = p

                if best_score <= 0.01:
                    best_point = rng.choice(candidates)

                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            blocks.append(block)

            # Update uncovered
            for ts in itertools.combinations(block, t):
                tid = get_id(ts)
                uncovered.discard(tid)

        return blocks

    # Phase 1: Multiple greedy restarts with varied strategies
    best_blocks = None
    best_count = float('inf')

    strategies = ['max_gain', 'max_gain', 'ratio', 'ratio2', 'random']
    num_seeds = 40
    for seed in range(num_seeds):
        if time.time() > deadline - 25:
            break
        strategy = strategies[seed % len(strategies)]
        blocks = greedy_construct(seed, strategy)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_construct(0, 'max_gain')

    # Phase 1.5: Try to improve best greedy with a second pass of targeted greedy
    # Fill remaining gaps with additional greedy blocks if needed
    if time.time() < deadline - 30:
        count_check = np.zeros(n_tsets, dtype=np.int32)
        for block in best_blocks:
            for ts in itertools.combinations(block, t):
                count_check[get_id(ts)] += 1
        uncovered_ids = np.where(count_check == 0)[0]
        if len(uncovered_ids) > 0 and len(uncovered_ids) < n_tsets * 0.1:
            # Add a few more greedy blocks to cover remaining gaps
            remaining_uncovered = set(uncovered_ids.tolist())
            extra_blocks = []
            while remaining_uncovered and time.time() < deadline - 25:
                start_id = random.Random(len(extra_blocks)).choice(list(remaining_uncovered))
                start_ts = all_tsets[start_id]
                block = list(start_ts)
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
                            if tid in remaining_uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_point = p
                    if best_point == -1:
                        best_point = random.Random(len(extra_blocks) + 1).choice(candidates)
                    block.append(best_point)
                    block_set.add(best_point)
                block.sort()
                extra_blocks.append(block)
                for ts in itertools.combinations(block, t):
                    tid = get_id(ts)
                    remaining_uncovered.discard(tid)
            if len(extra_blocks) < 10:  # Only add if it's not too many
                candidate = best_blocks + extra_blocks
                if len(candidate) < best_count:
                    best_count = len(candidate)
                    best_blocks = candidate

    # Phase 2: Local search
    count = np.zeros(n_tsets, dtype=np.int32)

    def update_counts(blocks, delta=1):
        for block in blocks:
            for ts in itertools.combinations(block, t):
                tid = get_id(ts)
                count[tid] += delta

    update_counts(best_blocks, 1)
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
                update_counts([b], -1)
                removed = True
            else:
                new_blocks.append(b)
        return new_blocks, removed

    def try_swap_2_for_1(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False

        # Precompute block t-set IDs for all blocks
        block_ids = []
        for b in blocks:
            ids = set()
            for ts in itertools.combinations(b, t):
                ids.add(get_id(ts))
            block_ids.append(ids)

        # Score each pair by how "redundant" they are (lower score = more overlap potential)
        # Use a heuristic: prioritize pairs with high overlap in their covered t-sets
        pair_scores = []
        for i in range(n):
            if time.time() > deadline - 10:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 10:
                    break
                # Quick overlap estimate
                overlap = len(block_ids[i] & block_ids[j])
                total = len(block_ids[i] | block_ids[j])
                if total > 0:
                    # Higher overlap ratio means more likely to be replaceable
                    score = overlap / total
                    # Also consider if either block has low unique coverage
                    unique_i = len(block_ids[i] - block_ids[j])
                    unique_j = len(block_ids[j] - block_ids[i])
                    low_unique = (unique_i < 5 or unique_j < 5)
                    if low_unique:
                        score += 0.5
                    pair_scores.append((score, i, j))

        # Sort by score descending and try top candidates first
        pair_scores.sort(reverse=True)
        max_pairs_to_try = min(len(pair_scores), 500)  # Limit for speed

        for score, i, j in pair_scores[:max_pairs_to_try]:
            if time.time() > deadline - 10:
                break

            b1 = blocks[i]
            b2 = blocks[j]
            ids_b1 = block_ids[i]
            ids_b2 = block_ids[j]

            check_ids = ids_b1 | ids_b2

            # Find t-sets uniquely covered by b1 or b2
            u_need = []
            for tid in check_ids:
                c = count[tid]
                cov_b1 = 1 if tid in ids_b1 else 0
                cov_b2 = 1 if tid in ids_b2 else 0
                if c == cov_b1 + cov_b2:
                    u_need.append(tid)

            if not u_need:
                # Both blocks are redundant together
                update_counts([b1, b2], -1)
                return [b for idx, b in enumerate(blocks) if idx != i and idx != j], True

            # Find required points
            p_need = set()
            for tid in u_need:
                p_need.update(all_tsets[tid])

            if len(p_need) > k:
                continue

            # Build a new block that covers u_need, optimizing for additional coverage
            new_block = list(p_need)
            new_block_set = set(new_block)
            rng = random.Random(i * 1000 + j)

            # Try to fill remaining slots with points that maximize additional coverage
            remaining_slots = k - len(new_block)
            candidates = [p for p in range(v) if p not in new_block_set]

            # Score candidates by how many additional t-sets they cover
            for _ in range(remaining_slots):
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

            # Verify the new block actually covers all u_need
            new_ids = set()
            for ts in itertools.combinations(new_block, t):
                new_ids.add(get_id(ts))

            if all(tid in new_ids for tid in u_need):
                update_counts([b1, b2], -1)
                update_counts([new_block], 1)

                new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                new_blocks.append(new_block)
                return new_blocks, True

        return blocks, False

    def try_swap_1_for_1(blocks):
        n = len(blocks)
        if n < 1:
            return blocks, False

        # Prioritize blocks with fewer unique t-sets (more likely to be replaceable)
        block_unique_counts = []
        for i in range(n):
            b = blocks[i]
            ids_b = set()
            for ts in itertools.combinations(b, t):
                ids_b.add(get_id(ts))
            unique = sum(1 for tid in ids_b if count[tid] == 1)
            block_unique_counts.append((unique, i))

        # Sort by unique count ascending
        block_unique_counts.sort()
        max_blocks_to_try = min(n, 100)

        for unique_count, i in block_unique_counts[:max_blocks_to_try]:
            if time.time() > deadline - 10:
                break

            b = blocks[i]

            # Get t-set IDs covered by b
            ids_b = set()
            for ts in itertools.combinations(b, t):
                ids_b.add(get_id(ts))

            # Find t-sets uniquely covered by b
            u_need = []
            for tid in ids_b:
                if count[tid] == 1:
                    u_need.append(tid)

            if not u_need:
                # Block is redundant
                update_counts([b], -1)
                return [x for idx, x in enumerate(blocks) if idx != i], True

            # Find required points
            p_need = set()
            for tid in u_need:
                p_need.update(all_tsets[tid])

            if len(p_need) > k:
                continue

            # Build new block optimizing for additional coverage
            new_block = list(p_need)
            new_block_set = set(new_block)
            rng = random.Random(i * 7919)
            candidates = [p for p in range(v) if p not in new_block_set]

            remaining_slots = k - len(new_block)
            for _ in range(remaining_slots):
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

            # Verify coverage and ensure different from old
            if new_block != b:
                new_ids = set()
                for ts in itertools.combinations(new_block, t):
                    new_ids.add(get_id(ts))

                if all(tid in new_ids for tid in u_need):
                    update_counts([b], -1)
                    update_counts([new_block], 1)
                    new_blocks = [x for idx, x in enumerate(blocks) if idx != i]
                    new_blocks.append(new_block)
                    return new_blocks, True

        return blocks, False

    def try_block_perturbation(blocks):
        """Perturb a block by replacing 1-3 points with different ones, then try to recover coverage."""
        n = len(blocks)
        if n < 1:
            return blocks, False

        rng = random.Random(int(time.time() * 1000) % 100000)

        # Try perturbing a few random blocks
        for _attempt in range(min(5, n)):
            if time.time() > deadline - 10:
                break

            i = rng.randrange(n)
            b = blocks[i]

            # Get t-set IDs covered by b
            ids_b = set()
            for ts in itertools.combinations(b, t):
                ids_b.add(get_id(ts))

            # Find t-sets uniquely covered by b
            u_need = [tid for tid in ids_b if count[tid] == 1]

            # Pick 1-3 points to replace
            num_replace = rng.choice([1, 2, 3])
            points_to_replace = rng.sample(b, min(num_replace, len(b)))
            replace_set = set(points_to_replace)

            # New block: keep points not in replace_set, add new random points
            kept = [p for p in b if p not in replace_set]
            candidates = [p for p in range(v) if p not in set(kept)]

            if len(candidates) < num_replace:
                continue

            new_points = rng.sample(candidates, num_replace)
            new_block = sorted(kept + new_points)

            if new_block == b:
                continue

            # Check if new block covers all u_need
            new_ids = set()
            for ts in itertools.combinations(new_block, t):
                new_ids.add(get_id(ts))

            if all(tid in new_ids for tid in u_need):
                # Check if this creates any new redundancy or helps
                # Apply the change
                update_counts([b], -1)
                update_counts([new_block], 1)
                new_blocks = [x for idx, x in enumerate(blocks) if idx != i]
                new_blocks.append(new_block)
                return new_blocks, True

        return blocks, False

    # Iteratively apply local search moves with multiple passes per round
    improved = True
    max_rounds = 150
    stall_count = 0
    for _ in range(max_rounds):
        if time.time() > deadline - 15:
            break
        if not improved:
            stall_count += 1
            if stall_count >= 3:
                # Try perturbation to escape local optimum
                if time.time() < deadline - 20:
                    new_blocks, pert = try_block_perturbation(current_blocks)
                    if pert:
                        current_blocks = new_blocks
                        improved = True
                        stall_count = 0
                        # After perturbation, try removals
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

        # Try removing redundant blocks (multiple passes)
        for _pass in range(3):
            if time.time() > deadline - 15:
                break
            new_blocks, rem = try_remove_redundant(current_blocks)
            if rem:
                current_blocks = new_blocks
                improved = True
            else:
                break

        # Try 2-for-1 swaps
        new_blocks, swp = try_swap_2_for_1(current_blocks)
        if swp:
            current_blocks = new_blocks
            improved = True
            # After a swap, try more removals
            for _pass in range(2):
                if time.time() > deadline - 15:
                    break
                new_blocks, rem = try_remove_redundant(current_blocks)
                if rem:
                    current_blocks = new_blocks
                    improved = True
                else:
                    break

        # Try 1-for-1 swaps
        new_blocks, swp = try_swap_1_for_1(current_blocks)
        if swp:
            current_blocks = new_blocks
            improved = True
            # After a swap, try more removals
            for _pass in range(2):
                if time.time() > deadline - 15:
                    break
                new_blocks, rem = try_remove_redundant(current_blocks)
                if rem:
                    current_blocks = new_blocks
                    improved = True
                else:
                    break

    # Final verification: ensure all t-sets are covered
    # (Should be valid, but just in case)
    covered = set()
    for block in current_blocks:
        for ts in itertools.combinations(block, t):
            covered.add(get_id(ts))

    if len(covered) < n_tsets:
        # Fallback: use the best greedy result
        return best_blocks

    # Final polish: try one more round of removals
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
# EVOLVE-BLOCK-END
