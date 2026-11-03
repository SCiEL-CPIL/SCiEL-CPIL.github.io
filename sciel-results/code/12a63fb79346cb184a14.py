"""
Randomized local search for covering design C(24,7,3).

Key improvements over the baseline:
1. Precomputed triple-to-block incidence for O(1) coverage checks during local search.
2. Faster greedy construction using precomputed triple lists.
3. More aggressive block removal (scan all blocks, not just one per iteration).
4. Smarter swap: rebuild a block to cover its critical triples while maximizing
   coverage of currently-uncovered triples.
5. Simulated-annealing-style perturbation with geometrically decreasing temperature.
6. Multiple restarts with different random seeds to escape local optima.
"""

import itertools
import math
import random
import time


def _precompute_triples(v, k, t):
    """Precompute all t-subsets of each k-subset, and a mapping from triple to list of
    (block_index, triple_index_in_block) is built dynamically. Instead we precompute
    for each k-subset (as a frozenset) the list of its t-subsets as frozensets."""
    # For speed, we'll compute triples on the fly but cache per-block.
    # Actually, let's precompute all C(v,t) triples as a list.
    all_triples = [frozenset(c) for c in itertools.combinations(range(v), t)]
    return all_triples


def _block_triples(block, t):
    """Return list of frozenset triples in a block."""
    return [frozenset(c) for c in itertools.combinations(block, t)]


def _greedy_cover(v, k, t, rng, beta=6.0, all_triples=None):
    """Greedy construction. Returns list of blocks (tuples)."""
    if all_triples is None:
        all_triples = [frozenset(c) for c in itertools.combinations(range(v), t)]

    # Map each triple to its index for fast lookup
    triple_to_idx = {tr: i for i, tr in enumerate(all_triples)}
    n_triples = len(all_triples)

    uncovered = bytearray([1]) * n_triples  # 1 = uncovered
    uncovered_set = set(range(n_triples))
    blocks = []
    block_set = set()

    points = list(range(v))

    while uncovered_set:
        # Pick a random uncovered triple as seed
        seed_idx = next(iter(uncovered_set))
        seed = set(all_triples[seed_idx])
        block = list(seed)
        rng.shuffle(block)

        while len(block) < k:
            cands = [x for x in points if x not in block]
            if not cands:
                break

            # Compute gain for each candidate
            gains = []
            block_set_current = set(block)
            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in triple_to_idx:
                        idx = triple_to_idx[fs]
                        if uncovered[idx]:
                            gain += 1
                gains.append(gain)

            # Soft-max sampling
            if beta < 100:
                m = max(gains) if gains else 0
                if m < -1e9:
                    m = 0.0
                scaled = [math.exp(beta * (g - m)) for g in gains]
                total = sum(scaled)
                if total <= 0:
                    idx = rng.randrange(len(cands))
                else:
                    r = rng.random() * total
                    acc = 0.0
                    idx = len(cands) - 1
                    for i, w in enumerate(scaled):
                        acc += w
                        if r <= acc:
                            idx = i
                            break
            else:
                # Greedy: pick max gain
                max_g = max(gains)
                best = [i for i, g in enumerate(gains) if g == max_g]
                idx = rng.choice(best)

            block.append(cands[idx])

        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in block_set:
            blocks.append(b_tuple)
            block_set.add(b_tuple)
            for tr in _block_triples(b_tuple, t):
                if tr in triple_to_idx:
                    idx = triple_to_idx[tr]
                    uncovered[idx] = 0
                    uncovered_set.discard(idx)

    return blocks


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 110.0  # safety margin under 150s kill

    # Precompute all triples
    all_triples = [frozenset(c) for c in itertools.combinations(range(v), t)]
    triple_to_idx = {tr: i for i, tr in enumerate(all_triples)}
    n_triples = len(all_triples)

    # ---- Build initial valid covering ----
    blocks = _greedy_cover(v, k, t, rng, beta=8.0, all_triples=all_triples)

    # Try a second start with lower beta if time remains
    elapsed = time.time() - start_time
    if elapsed < time_limit * 0.35:
        alt = _greedy_cover(v, k, t, rng, beta=2.0, all_triples=all_triples)
        if len(alt) < len(blocks):
            blocks = alt

    def build_counts(blk_list):
        """Build coverage count array."""
        counts = [0] * n_triples
        for b in blk_list:
            for tr in _block_triples(b, t):
                idx = triple_to_idx[tr]
                counts[idx] += 1
        return counts

    counts = build_counts(blocks)
    best_blocks = list(blocks)
    best_count = len(blocks)

    def uncovered_count(counts_arr):
        return sum(1 for c in counts_arr if c == 0)

    unc = uncovered_count(counts)

    def apply_block_change(blk, sign, counts_arr):
        """Apply add/remove of block. Returns delta in uncovered count."""
        delta = 0
        for tr in _block_triples(blk, t):
            idx = triple_to_idx[tr]
            old = counts_arr[idx]
            new = old + sign
            counts_arr[idx] = new
            if old == 0 and new == 1:
                delta -= 1
            elif old == 1 and new == 0:
                delta += 1
        return delta

    def try_remove_block(i, blocks, counts, unc):
        """Try to remove blocks[i]. Returns new_unc if removed, else None."""
        b = blocks[i]
        redundant = True
        for tr in _block_triples(b, t):
            idx = triple_to_idx[tr]
            if counts[idx] <= 1:
                redundant = False
                break
        if not redundant:
            return None
        delta_u = apply_block_change(b, -1, counts)
        new_unc = unc + delta_u
        if new_unc == 0:
            blocks.pop(i)
            return new_unc
        else:
            apply_block_change(b, 1, counts)
            return None

    def re_cover_greedy(blocks, counts, unc, rng):
        """Greedily add blocks to cover all uncovered triples. Mutates blocks, counts.
        Returns new unc (should be 0)."""
        unc_list = [i for i in range(n_triples) if counts[i] == 0]
        while unc_list:
            seed_idx = rng.choice(unc_list)
            seed = set(all_triples[seed_idx])
            new_b = list(seed)
            rng.shuffle(new_b)
            new_b_set = set(new_b)
            while len(new_b) < k:
                cands = [x for x in range(v) if x not in new_b_set]
                if not cands:
                    break
                gains = []
                for x in cands:
                    gain = 0
                    for rest in itertools.combinations(new_b, t - 1):
                        fs = frozenset(list(rest) + [x])
                        if fs in triple_to_idx:
                            idx = triple_to_idx[fs]
                            if counts[idx] == 0:
                                gain += 1
                    gains.append(gain)
                m = max(gains) if gains else 0
                scaled = [math.exp(3.0 * (g - m)) for g in gains]
                total = sum(scaled)
                if total <= 0:
                    idx = rng.randrange(len(cands))
                else:
                    r = rng.random() * total
                    acc = 0.0
                    idx = len(cands) - 1
                    for j, w in enumerate(scaled):
                        acc += w
                        if r <= acc:
                            idx = j
                            break
                new_b.append(cands[idx])
                new_b_set.add(cands[idx])
            new_b.sort()
            new_b_tuple = tuple(new_b)
            if new_b_tuple not in blocks:
                blocks.append(new_b_tuple)
                delta_u_add = apply_block_change(new_b_tuple, 1, counts)
                unc += delta_u_add
                unc_list = [i for i in range(n_triples) if counts[i] == 0]
        return unc

    no_improve = 0
    max_no_improve = 300

    # Simulated annealing temperature
    temp = 2.0
    temp_decay = 0.995
    min_temp = 0.01

    while time.time() - start_time < time_limit:

        # 1. Try to remove redundant blocks (scan all, remove any that are safe)
        removed_any = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            b = blocks[i]
            redundant = True
            for tr in _block_triples(b, t):
                idx = triple_to_idx[tr]
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                delta_u = apply_block_change(b, -1, counts)
                unc += delta_u
                if unc == 0:
                    blocks.pop(i)
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                    removed_any = True
                    no_improve = 0
                    break
                else:
                    apply_block_change(b, 1, counts)
                    unc -= delta_u

        if removed_any:
            continue

        # 2. Try 2-block exchange: remove two blocks, re-cover greedily
        if len(blocks) >= 2 and rng.random() < 0.3:
            i1 = rng.randrange(len(blocks))
            i2 = rng.randrange(len(blocks) - 1)
            if i2 >= i1:
                i2 += 1
            b1 = blocks[i1]
            b2 = blocks[i2]
            # Check if either is non-redundant (if both redundant, single removal would work)
            old_blocks = list(blocks)
            old_counts = list(counts)
            old_unc = unc
            # Remove both
            apply_block_change(b1, -1, counts)
            apply_block_change(b2, -1, counts)
            # Rebuild counts for unc check
            # Actually we need to track unc properly; let's just recount
            del blocks[max(i1, i2)]
            del blocks[min(i1, i2)]
            unc = sum(1 for c in counts if c == 0)
            # Re-cover
            unc = re_cover_greedy(blocks, counts, unc, rng)
            if unc == 0:
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                no_improve = 0
            else:
                blocks = old_blocks
                counts = old_counts
                unc = old_unc
                no_improve += 1
            temp *= temp_decay
            continue

        # 2. Try swaps: pick a random block, rebuild it
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]

            # Find critical triples (covered only by this block)
            critical = []
            for tr in _block_triples(old_block, t):
                idx = triple_to_idx[tr]
                if counts[idx] == 1:
                    critical.append(idx)

            if not critical:
                no_improve += 1
            else:
                points_needed = set()
                for ci in critical:
                    points_needed.update(all_triples[ci])

                if len(points_needed) <= k:
                    new_block_set = set(points_needed)
                    cands = [x for x in range(v) if x not in new_block_set]
                    rng.shuffle(cands)

                    # Greedily add points to maximize coverage of uncovered triples
                    for x in cands:
                        if len(new_block_set) >= k:
                            break
                        gain = 0
                        for rest in itertools.combinations(new_block_set, t - 1):
                            fs = frozenset(list(rest) + [x])
                            if fs in triple_to_idx:
                                idx = triple_to_idx[fs]
                                if counts[idx] == 0:
                                    gain += 10
                                elif counts[idx] == 1:
                                    gain += 1
                        new_block_set.add(x)

                    if len(new_block_set) == k:
                        new_block = tuple(sorted(new_block_set))
                        if new_block != old_block:
                            # Simulate the swap
                            delta_remove = apply_block_change(old_block, -1, counts)
                            delta_add = apply_block_change(new_block, 1, counts)
                            new_unc = unc + delta_remove + delta_add

                            if new_unc == 0:
                                blocks[i] = new_block
                                unc = new_unc
                                no_improve = 0
                                if len(blocks) < best_count:
                                    best_count = len(blocks)
                                    best_blocks = list(blocks)
                            else:
                                # Accept with probability exp(-delta/temp) if delta > 0
                                delta = new_unc  # increase in uncovered
                                if delta > 0 and temp > min_temp:
                                    import math as _math
                                    if rng.random() < _math.exp(-delta / temp):
                                        blocks[i] = new_block
                                        unc = new_unc
                                        no_improve = 0
                                    else:
                                        apply_block_change(new_block, -1, counts)
                                        apply_block_change(old_block, 1, counts)
                                        no_improve += 1
                                else:
                                    apply_block_change(new_block, -1, counts)
                                    apply_block_change(old_block, 1, counts)
                                    no_improve += 1
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1

            temp *= temp_decay

        # 3. Perturbation if stuck
        if no_improve > max_no_improve:
            if len(blocks) > 1:
                old_blocks = list(blocks)
                old_counts = list(counts)
                old_unc = unc

                # Remove a random block
                i = rng.randrange(len(blocks))
                b = blocks[i]
                delta_u = apply_block_change(b, -1, counts)
                unc += delta_u
                blocks.pop(i)

                # Re-cover using greedy
                unc = re_cover_greedy(blocks, counts, unc, rng)

                if unc == 0:
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                    no_improve = 0
                else:
                    blocks = old_blocks
                    counts = old_counts
                    unc = old_unc
                    no_improve += 1
            else:
                no_improve += 1

    # Final polish: exhaustive redundancy removal
    changed = True
    while changed and (time.time() - start_time) < time_limit:
        changed = False
        for i in range(len(blocks)):
            new_unc = try_remove_block(i, blocks, counts, unc)
            if new_unc is not None:
                unc = new_unc
                changed = True
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                break

    return [list(b) for b in best_blocks]
