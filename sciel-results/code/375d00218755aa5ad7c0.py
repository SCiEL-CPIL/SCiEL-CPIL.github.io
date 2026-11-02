import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 115.0
    t0 = time.time()

    # Precomputation
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_ts = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    c_k_t = 1
    for i in range(t):
        c_k_t = c_k_t * (k - i) // (i + 1)

    # block_tsubs[bi] = numpy array of t-subset indices covered by block bi
    block_tsubs = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for bi, block in enumerate(all_blocks):
        block_tsubs[bi] = np.array([tsub_idx[ts] for ts in itertools.combinations(block, t)], dtype=np.int32)

    # tsub_blocks[ts_idx] = numpy array of block indices covering t-subset ts_idx
    # Build via counting
    tsub_block_counts = np.zeros(n_ts, dtype=np.int32)
    for bi in range(n_blocks):
        tsub_block_counts[block_tsubs[bi]] += 1
    tsub_blocks = []
    for i in range(n_ts):
        tsub_blocks.append(np.where(tsub_block_counts[i] > 0)[0])  # placeholder, will rebuild

    # Actually build properly with CSR-like approach
    total_entries = n_blocks * c_k_t
    flat_ts = np.zeros(total_entries, dtype=np.int32)
    flat_bi = np.zeros(total_entries, dtype=np.int32)
    for bi in range(n_blocks):
        start = bi * c_k_t
        flat_ts[start:start+c_k_t] = block_tsubs[bi]
        flat_bi[start:start+c_k_t] = bi

    # Sort by ts index to build tsub_blocks
    sort_order = np.argsort(flat_ts, kind='stable')
    sorted_ts = flat_ts[sort_order]
    sorted_bi = flat_bi[sort_order]

    # Find boundaries for each ts
    ts_boundaries = np.searchsorted(sorted_ts, np.arange(n_ts + 1), side='left')
    tsub_blocks = []
    for i in range(n_ts):
        s = ts_boundaries[i]
        e = ts_boundaries[i + 1]
        tsub_blocks.append(sorted_bi[s:e].copy())

    # Helper: check if a set of blocks covers all t-subsets
    def is_valid(block_indices):
        if not block_indices:
            return n_ts == 0
        covered = np.zeros(n_ts, dtype=np.int8)
        for bi in block_indices:
            covered[block_tsubs[bi]] = 1
        return covered.all()

    # Greedy with penalty weights
    def greedy_with_penalties(seed=0, use_lp_weights=False):
        rng = random.Random(seed)
        uncovered = np.ones(n_ts, dtype=bool)
        n_uncovered = n_ts

        # Penalty weights: initially uniform, updated dynamically
        # Weight of a t-subset = 1 / (number of blocks covering it) + small noise
        # This makes hard t-subsets (few covering blocks) more valuable
        ts_weights = np.zeros(n_ts, dtype=np.float64)
        for i in range(n_ts):
            ts_weights[i] = 1.0 / max(1, len(tsub_blocks[i]))
        ts_weights += rng.random(n_ts) * 0.01  # small noise for diversity

        # Block scores: sum of ts_weights for uncovered t-subsets in block
        block_scores = np.zeros(n_blocks, dtype=np.float64)
        for bi in range(n_blocks):
            block_scores[bi] = np.sum(ts_weights[block_tsubs[bi]])

        selected = []

        while n_uncovered > 0:
            if time.time() > deadline - 5:
                break

            # Pick block with max score (with some randomness among top)
            top_k = min(5, n_blocks)
            top_idx = np.argsort(block_scores)[-top_k:]
            # Weighted random pick among top
            weights = block_scores[top_idx]
            weights = weights - weights.min() + 1e-9
            weights /= weights.sum()
            chosen = top_idx[rng.choices(range(len(top_idx)), p=weights)[0]]

            if block_scores[chosen] < 1e-9:
                break

            selected.append(chosen)

            # Update: mark newly covered t-subsets
            ts_list = block_tsubs[chosen]
            newly_covered = ts_list[uncovered[ts_list]]

            if len(newly_covered) > 0:
                uncovered[newly_covered] = False
                n_uncovered -= len(newly_covered)

                # Update block scores: subtract ts_weights of newly covered from all blocks containing them
                for ts_i in newly_covered:
                    w = ts_weights[ts_i]
                    if w > 0:
                        block_scores[tsub_blocks[ts_i]] -= w

            block_scores[chosen] = -1e10

            # Dynamic penalty update: increase weight of still-uncovered t-subsets
            # that are hard to cover (few remaining blocks can cover them)
            if n_uncovered > 0 and n_uncovered < n_ts:
                # For uncovered t-subsets, boost weight if few blocks can still cover them
                for ts_i in np.where(uncovered)[0]:
                    # Count how many unselected blocks cover this ts
                    covering = tsub_blocks[ts_i]
                    # Rough estimate: just boost based on original count
                    ts_weights[ts_i] *= 1.1  # small boost each iteration

        return selected

    # WalkSAT-style repair: given a set of blocks, try to fix uncovered t-subsets
    def walksat_repair(blocks, seed=0, max_steps=500):
        rng = random.Random(seed)
        blocks = list(blocks)

        # Compute coverage
        cover_count = np.zeros(n_ts, dtype=np.int16)
        for bi in blocks:
            cover_count[block_tsubs[bi]] += 1

        uncovered = np.where(cover_count == 0)[0]
        n_uncov = len(uncovered)

        if n_uncov == 0:
            return blocks

        # Penalty weights for uncovered t-subsets
        penalties = np.zeros(n_ts, dtype=np.float64)
        for ts_i in uncovered:
            penalties[ts_i] = 1.0 / max(1, len(tsub_blocks[ts_i]))

        block_set = set(blocks)

        for step in range(max_steps):
            if time.time() > deadline - 3:
                break
            if n_uncov == 0:
                break

            # Find best block to add: maximizes coverage of uncovered t-subsets weighted by penalty
            best_block = -1
            best_score = -1
            # Sample a subset of blocks for speed
            candidates = list(range(n_blocks))
            if len(candidates) > 200:
                candidates = rng.sample(candidates, 200)

            for bi in candidates:
                if bi in block_set:
                    continue
                ts_list = block_tsubs[bi]
                score = np.sum(penalties[ts_list] * (cover_count[ts_list] == 0))
                if score > best_score:
                    best_score = score
                    best_block = bi

            if best_block < 0 or best_score < 1e-9:
                break

            # Add the block
            blocks.append(best_block)
            block_set.add(best_block)
            ts_list = block_tsubs[best_block]
            cover_count[ts_list] += 1

            newly_covered = ts_list[cover_count[ts_list] == 1]
            for ts_i in newly_covered:
                n_uncov -= 1
                penalties[ts_i] = 0

            # Occasionally try to remove a redundant block
            if step % 3 == 0 and len(blocks) > 1:
                # Find a block whose removal doesn't uncover anything
                for bi in blocks:
                    ts_list = block_tsubs[bi]
                    if np.all(cover_count[ts_list] > 1):
                        # Safe to remove
                        blocks.remove(bi)
                        block_set.discard(bi)
                        cover_count[ts_list] -= 1
                        break

            # Update penalties for still-uncovered t-subsets
            if n_uncov > 0:
                for ts_i in np.where(cover_count == 0)[0]:
                    penalties[ts_i] = 1.0 / max(1, len(tsub_blocks[ts_i])) * (1 + 0.05 * step)

        return blocks

    # Pair-replacement local search: remove 2 blocks, add 1
    def pair_replacement(blocks, seed=0, max_iterations=200):
        rng = random.Random(seed)
        blocks = list(blocks)

        for iteration in range(max_iterations):
            if time.time() > deadline - 3:
                break
            if len(blocks) <= 1:
                break

            # Build coverage
            cover_count = np.zeros(n_ts, dtype=np.int16)
            for bi in blocks:
                cover_count[block_tsubs[bi]] += 1

            improved = False
            n = len(blocks)

            # Try all pairs (with limit for speed)
            max_pairs = min(500, n * (n - 1) // 2)
            pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.append((i, j))
            rng.shuffle(pairs)
            pairs = pairs[:max_pairs]

            for (i, j) in pairs:
                if time.time() > deadline - 3:
                    break

                bi = blocks[i]
                bj = blocks[j]

                # t-subsets uniquely covered by bi or bj (i.e., cover_count == 1)
                ts_bi = block_tsubs[bi]
                ts_bj = block_tsubs[bj]

                unique_to_bi = ts_bi[cover_count[ts_bi] == 1]
                unique_to_bj = ts_bj[cover_count[ts_bj] == 1]

                # Union of unique t-subsets that must be covered by replacement
                must_cover = np.unique(np.concatenate([unique_to_bi, unique_to_bj])) if len(unique_to_bi) + len(unique_to_bj) > 0 else np.array([], dtype=np.int32)

                if len(must_cover) == 0:
                    # Both blocks are redundant, remove both
                    cover_count[ts_bi] -= 1
                    cover_count[ts_bj] -= 1
                    blocks.pop(j)
                    blocks.pop(i)
                    improved = True
                    break

                # Find a block that covers all must_cover t-subsets
                # and is not already in blocks
                block_set = set(blocks)
                found = False
                # Candidate blocks: intersection of tsub_blocks for each must_cover ts
                if len(must_cover) <= 3:  # limit for speed
                    candidate_set = None
                    for ts_i in must_cover:
                        candidates = set(tsub_blocks[ts_i].tolist())
                        candidates -= block_set
                        if candidate_set is None:
                            candidate_set = candidates
                        else:
                            candidate_set &= candidates
                        if not candidate_set:
                            break

                    if candidate_set:
                        candidates = list(candidate_set)
                        rng.shuffle(candidates)
                        for ci in candidates[:20]:
                            # Verify: after removing bi, bj and adding ci, all t-subsets still covered
                            ok = True
                            # Check t-subsets in bi that are not in ci
                            ts_ci = block_tsubs[ci]
                            ts_ci_set = set(ts_ci.tolist())
                            for ts_i in ts_bi:
                                if cover_count[ts_i] == 1 and ts_i not in ts_ci_set:
                                    ok = False
                                    break
                            if ok:
                                for ts_i in ts_bj:
                                    if cover_count[ts_i] == 1 and ts_i not in ts_ci_set:
                                        ok = False
                                        break
                            if ok:
                                # Perform swap
                                cover_count[ts_bi] -= 1
                                cover_count[ts_bj] -= 1
                                cover_count[ts_ci] += 1
                                blocks[i] = ci
                                blocks.pop(j)
                                improved = True
                                found = True
                                break

                if found:
                    break

            if not improved:
                # Try single-block removal of redundant blocks
                cover_count = np.zeros(n_ts, dtype=np.int16)
                for bi in blocks:
                    cover_count[block_tsubs[bi]] += 1

                for i in range(len(blocks)):
                    ts_list = block_tsubs[blocks[i]]
                    if np.all(cover_count[ts_list] > 1):
                        cover_count[ts_list] -= 1
                        blocks.pop(i)
                        improved = True
                        break

            if not improved:
                break

        return blocks

    # Remove all redundant blocks
    def remove_redundant(blocks):
        if not blocks:
            return blocks

        cover_count = np.zeros(n_ts, dtype=np.int16)
        for bi in blocks:
            cover_count[block_tsubs[bi]] += 1

        kept = []
        for bi in blocks:
            ts_list = block_tsubs[bi]
            if np.all(cover_count[ts_list] > 1):
                cover_count[ts_list] -= 1
            else:
                kept.append(bi)

        # Iterate until no more changes
        changed = True
        while changed:
            changed = False
            cover_count = np.zeros(n_ts, dtype=np.int16)
            for bi in kept:
                cover_count[block_tsubs[bi]] += 1

            new_kept = []
            for bi in kept:
                ts_list = block_tsubs[bi]
                if np.all(cover_count[ts_list] > 1):
                    cover_count[ts_list] -= 1
                    changed = True
                else:
                    new_kept.append(bi)
            kept = new_kept

        return kept

    # Main search: multi-restart greedy + WalkSAT + pair-replacement
    best_blocks = None
    best_len = float('inf')

    num_greedy_trials = 40
    num_walksat_trials = 30

    # Phase 1: Multi-restart greedy
    for seed in range(num_greedy_trials):
        if time.time() > deadline - 20:
            break

        blocks = greedy_with_penalties(seed)
        if not blocks:
            continue

        # Remove redundant
        blocks = remove_redundant(blocks)

        # Pair-replacement local search
        blocks = pair_replacement(blocks, seed, max_iterations=100)

        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]

    # Phase 2: WalkSAT repair from greedy results
    for seed in range(num_walksat_trials):
        if time.time() > deadline - 15:
            break

        blocks = greedy_with_penalties(seed + 1000)
        if not blocks:
            continue

        blocks = walksat_repair(blocks, seed, max_steps=300)
        blocks = remove_redundant(blocks)
        blocks = pair_replacement(blocks, seed + 500, max_iterations=100)

        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]

    # Phase 3: Intensive local search on best
    if best_blocks is not None and time.time() < deadline - 10:
        final = pair_replacement(best_blocks, 9999, max_iterations=300)
        final = remove_redundant(final)
        if len(final) < best_len:
            best_blocks = final
            best_len = len(final)

    # Final verification and fallback
    if best_blocks is None:
        # Fallback: simple greedy
        blocks = greedy_with_penalties(0)
        if not blocks:
            # Absolute fallback: all blocks (shouldn't happen)
            best_blocks = list(range(n_blocks))
        else:
            best_blocks = blocks

    # Verify coverage
    if not is_valid(best_blocks):
        # Add missing blocks
        cover_count = np.zeros(n_ts, dtype=np.int16)
        for bi in best_blocks:
            cover_count[block_tsubs[bi]] += 1
        uncovered = np.where(cover_count == 0)[0]
        for ts_i in uncovered:
            # Add first block covering this t-subset
            for bi in tsub_blocks[ts_i]:
                if bi not in best_blocks:
                    best_blocks.append(bi)
                    break

    return [list(all_blocks[i]) for i in best_blocks]
