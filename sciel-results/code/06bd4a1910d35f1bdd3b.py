import itertools
import time
import random
from collections import defaultdict
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0

    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    best_blocks = None
    best_len = float('inf')

    # Strategy 1: Vectorized greedy with multiple seeds
    for trial in range(12):
        if time.time() > deadline:
            break
        seed = trial * 7919 + 42
        blocks = _greedy_cover_vectorized(v, k, t, all_tsubsets, tsub_to_idx, deadline, seed)
        if blocks is not None:
            blocks = _remove_redundant(blocks, v, k, t, deadline)
            blocks = _local_search(blocks, v, k, t, all_tsubsets, tsub_to_idx, deadline, seed)
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
                if best_len <= 76:
                    break

    # Strategy 2: Greedy with rarity-weighted scoring
    for trial in range(6):
        if time.time() > deadline:
            break
        seed = trial * 104729 + 13
        blocks = _greedy_cover_rarity(v, k, t, all_tsubsets, tsub_to_idx, deadline, seed)
        if blocks is not None:
            blocks = _remove_redundant(blocks, v, k, t, deadline)
            blocks = _local_search(blocks, v, k, t, all_tsubsets, tsub_to_idx, deadline, seed)
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
                if best_len <= 76:
                    break

    # Strategy 3: WalkSAT-style minimization on the best solution
    if best_blocks is not None and time.time() < deadline - 15:
        ws_blocks = _walksat_minimize(best_blocks, v, k, t, all_tsubsets, tsub_to_idx, deadline)
        if len(ws_blocks) < best_len:
            best_len = len(ws_blocks)
            best_blocks = ws_blocks

    # Final deep local search
    if best_blocks is not None and time.time() < deadline - 10:
        polished = _deep_local_search(best_blocks, v, k, t, all_tsubsets, tsub_to_idx, deadline)
        if len(polished) < best_len:
            best_len = len(polished)
            best_blocks = polished

    if best_blocks is None:
        blocks = _greedy_cover_vectorized(v, k, t, all_tsubsets, tsub_to_idx, deadline, -1)
        if blocks is not None:
            best_blocks = _remove_redundant(blocks, v, k, t, deadline)
        else:
            best_blocks = []

    return [list(b) for b in best_blocks]


def _greedy_cover_vectorized(v, k, t, all_tsubsets, tsub_to_idx, deadline, seed_val):
    """
    Vectorized greedy covering: at each step, pick the block that covers the most
    uncovered t-subsets. Uses numpy for fast gain computation.
    """
    rng = random.Random(seed_val if seed_val >= 0 else int(time.time()))
    uncovered = np.ones(len(all_tsubsets), dtype=np.int8)
    blocks = []
    points = list(range(v))

    while uncovered.sum() > 0:
        if time.time() > deadline:
            return None

        # Pick a random uncovered t-subset as seed
        uncovered_indices = np.where(uncovered == 1)[0]
        if len(uncovered_indices) == 0:
            break
        seed_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
        seed = all_tsubsets[seed_idx]
        block = list(seed)
        block_set = set(block)

        while len(block) < k:
            if time.time() > deadline:
                return None

            # Compute gain for each candidate point
            t_minus_1_subsets = list(itertools.combinations(block, t - 1))
            # For each candidate x, gain = number of t-subsets {x} ∪ p for p in t_minus_1_subsets that are uncovered
            best_x = -1
            best_score = -1
            candidates = []

            for x in points:
                if x in block_set:
                    continue

                gain = 0
                for p in t_minus_1_subsets:
                    s = tuple(sorted(p + (x,)))
                    idx = tsub_to_idx.get(s)
                    if idx is not None and uncovered[idx] == 1:
                        gain += 1

                if gain == 0:
                    continue

                score = gain + rng.random() * 0.1
                candidates.append((score, x))

            if not candidates:
                remaining = [x for x in points if x not in block_set]
                if not remaining:
                    return None
                chosen_x = rng.choice(remaining)
            else:
                candidates.sort(key=lambda c: -c[0])
                top_n = min(5, len(candidates))
                chosen_x = candidates[rng.randint(0, top_n - 1)][1]

            block.append(chosen_x)
            block_set.add(chosen_x)

        block.sort()
        blocks.append(tuple(block))

        for s in itertools.combinations(block, t):
            idx = tsub_to_idx.get(s)
            if idx is not None:
                uncovered[idx] = 0

    return blocks


def _greedy_cover_rarity(v, k, t, all_tsubsets, tsub_to_idx, deadline, seed_val):
    """
    Greedy covering with rarity-weighted scoring.
    """
    rng = random.Random(seed_val)
    uncovered = set(all_tsubsets)
    blocks = []
    points = list(range(v))

    while uncovered:
        if time.time() > deadline:
            return None

        seed = tuple(rng.choice(list(uncovered)))
        block = list(seed)
        block_set = set(block)

        while len(block) < k:
            if time.time() > deadline:
                return None

            t_minus_1_subsets = list(itertools.combinations(block, t - 1))
            candidates = []

            for x in points:
                if x in block_set:
                    continue

                gain = 0
                for p in t_minus_1_subsets:
                    s = tuple(sorted(p + (x,)))
                    if s in uncovered:
                        gain += 1

                if gain == 0:
                    continue

                score = gain + rng.random() * 0.05
                candidates.append((score, x))

            if not candidates:
                remaining = [x for x in points if x not in block_set]
                if not remaining:
                    return None
                chosen_x = rng.choice(remaining)
            else:
                candidates.sort(key=lambda c: -c[0])
                top_n = min(5, len(candidates))
                chosen_x = candidates[rng.randint(0, top_n - 1)][1]

            block.append(chosen_x)
            block_set.add(chosen_x)

        block.sort()
        blocks.append(tuple(block))

        for s in itertools.combinations(block, t):
            uncovered.discard(s)

    return blocks


def _remove_redundant(blocks, v, k, t, deadline):
    """Remove redundant blocks."""
    if not blocks:
        return blocks

    cover_count = defaultdict(int)
    for block in blocks:
        for s in itertools.combinations(block, t):
            cover_count[s] += 1

    improved = list(blocks)

    for i in range(len(improved) - 1, -1, -1):
        if time.time() > deadline:
            break

        block = improved[i]
        redundant = True
        for s in itertools.combinations(block, t):
            if cover_count[s] <= 1:
                redundant = False
                break

        if redundant:
            for s in itertools.combinations(block, t):
                cover_count[s] -= 1
            improved.pop(i)

    return improved


def _local_search(blocks, v, k, t, all_tsubsets, tsub_to_idx, deadline, seed_val):
    """Local search to reduce block count."""
    if not blocks:
        return blocks

    rng = random.Random(seed_val)
    improved = list(blocks)

    cover_count = defaultdict(int)
    for block in improved:
        for s in itertools.combinations(block, t):
            cover_count[s] += 1

    n_blocks = len(improved)
    max_iterations = 200

    for iteration in range(max_iterations):
        if time.time() > deadline:
            break

        any_change = False

        for i in range(n_blocks):
            if time.time() > deadline:
                break

            old_block = improved[i]
            old_tsubs = list(itertools.combinations(old_block, t))
            old_tsub_set = set(old_tsubs)

            unique_subs = [s for s in old_tsubs if cover_count[s] == 1]

            if not unique_subs:
                for s in old_tsubs:
                    cover_count[s] -= 1
                    if cover_count[s] == 0:
                        del cover_count[s]
                improved.pop(i)
                n_blocks -= 1
                any_change = True
                break

            best_new_block = None
            best_score = -1

            for _ in range(30):
                if time.time() > deadline:
                    break

                new_block = list(old_block)
                remove_count = rng.randint(1, min(2, k))
                to_remove = rng.sample(new_block, remove_count)
                for x in to_remove:
                    new_block.remove(x)
                while len(new_block) < k:
                    x = rng.randint(0, v - 1)
                    if x not in new_block:
                        new_block.append(x)
                cand = tuple(sorted(new_block))

                if cand == old_block:
                    continue

                cand_tsubs = set(itertools.combinations(cand, t))

                if not all(s in cand_tsubs for s in unique_subs):
                    continue

                new_subs = cand_tsubs - old_tsub_set
                gain = len(new_subs)

                redundancy_bonus = 0
                for s in new_subs:
                    if cover_count.get(s, 0) >= 1:
                        redundancy_bonus += 1

                score = gain + redundancy_bonus * 0.5

                if score > best_score:
                    best_score = score
                    best_new_block = cand

            if best_new_block is not None and best_score > 0:
                for s in old_tsubs:
                    cover_count[s] -= 1
                    if cover_count[s] == 0:
                        del cover_count[s]
                for s in itertools.combinations(best_new_block, t):
                    cover_count[s] += 1
                improved[i] = best_new_block
                n_blocks -= 1
                any_change = True
                break

        if not any_change:
            break

    improved = _remove_redundant(improved, v, k, t, deadline)
    return improved


def _deep_local_search(blocks, v, k, t, all_tsubsets, tsub_to_idx, deadline):
    """More intensive local search."""
    if not blocks:
        return blocks

    improved = list(blocks)
    best = improved
    best_len = len(improved)

    for trial in range(10):
        if time.time() > deadline:
            break

        seed = trial * 104729 + 7
        candidate = _local_search(improved, v, k, t, all_tsubsets, tsub_to_idx, deadline, seed)
        candidate = _remove_redundant(candidate, v, k, t, deadline)

        if len(candidate) < best_len:
            best = candidate
            best_len = len(candidate)
            improved = candidate

    return best


def _walksat_minimize(blocks, v, k, t, all_tsubsets, tsub_to_idx, deadline):
    """WalkSAT-style stochastic local search to minimize block count."""
    if not blocks:
        return blocks

    rng = random.Random(int(time.time() * 1000) % 100000)

    n_blocks = len(blocks)
    n_tsubs = len(all_tsubsets)

    block_covers = []
    tsub_coverers = [[] for _ in range(n_tsubs)]

    for i, block in enumerate(blocks):
        covered = set()
        for s in itertools.combinations(block, t):
            idx = tsub_to_idx.get(s)
            if idx is not None:
                covered.add(idx)
                tsub_coverers[idx].append(i)
        block_covers.append(covered)

    selected = [True] * n_blocks
    cover_count = np.zeros(n_tsubs, dtype=np.int32)
    for i in range(n_blocks):
        for idx in block_covers[i]:
            cover_count[idx] += 1

    num_uncovered = np.sum(cover_count == 0)
    num_selected = n_blocks
    best_score = num_selected + 10 * num_uncovered
    best_selected = selected[:]
    best_num_selected = num_selected

    lambda_penalty = 10
    p_flip = 0.3
    max_steps = 5000
    patience = 200

    current_score = num_selected + lambda_penalty * num_uncovered
    steps_since_improve = 0

    for step in range(max_steps):
        if time.time() > deadline:
            break

        if rng.random() < p_flip:
            i = rng.randint(0, n_blocks - 1)
        else:
            best_i = -1
            best_delta = 0

            n_samples = min(20, n_blocks)
            sampled = rng.sample(range(n_blocks), n_samples)

            for i in sampled:
                if selected[i]:
                    delta_selected = -1
                    newly_uncovered = 0
                    for idx in block_covers[i]:
                        if cover_count[idx] == 1:
                            newly_uncovered += 1
                    delta_uncovered = newly_uncovered
                    delta = delta_selected + lambda_penalty * delta_uncovered
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                else:
                    delta_selected = 1
                    newly_covered = 0
                    for idx in block_covers[i]:
                        if cover_count[idx] == 0:
                            newly_covered += 1
                    delta_uncovered = -newly_covered
                    delta = delta_selected + lambda_penalty * delta_uncovered
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i

            if best_i == -1:
                i = rng.randint(0, n_blocks - 1)
            else:
                i = best_i

        if selected[i]:
            selected[i] = False
            num_selected -= 1
            for idx in block_covers[i]:
                cover_count[idx] -= 1
                if cover_count[idx] == 0:
                    num_uncovered += 1
        else:
            selected[i] = True
            num_selected += 1
            for idx in block_covers[i]:
                if cover_count[idx] == 0:
                    num_uncovered -= 1
                cover_count[idx] += 1

        current_score = num_selected + lambda_penalty * num_uncovered

        if current_score < best_score:
            best_score = current_score
            best_selected = selected[:]
            best_num_selected = num_selected
            steps_since_improve = 0
        else:
            steps_since_improve += 1

    result = []
    for i in range(n_blocks):
        if best_selected[i]:
            result.append(blocks[i])

    result = _remove_redundant(result, v, k, t, deadline)
    return result
