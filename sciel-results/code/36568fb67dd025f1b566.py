"""Covering design C(19,8,5) - bitmask-accelerated randomized greedy with restarts."""
import time
import random
import itertools
import sys


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # ---- Precomputation ----
    all_t_subs = list(itertools.combinations(range(v), t))
    n_subs = len(all_t_subs)

    # Bitmask for each t-subset
    sub_masks = [0] * n_subs
    for i, sub in enumerate(all_t_subs):
        m = 0
        for x in sub:
            m |= (1 << x)
        sub_masks[i] = m

    # For each point x, the list of subset indices containing x
    point_to_subs = [[] for _ in range(v)]
    for i, m in enumerate(sub_masks):
        mm = m
        while mm:
            low = mm & (-mm)
            x = low.bit_length() - 1
            point_to_subs[x].append(i)
            mm &= mm - 1

    # For each point x, and each (t-1)-subset mask of the other points,
    # precompute the list of subset indices S such that x in S and (S \ {x}) == that mask.
    # Actually simpler: for gain when adding x to a block with mask B,
    # we need subsets S with x in S and (S & ~B) == (1<<x) i.e. S\{x} subset of B.
    # We'll compute gain by iterating point_to_subs[x] and checking (sub_masks[s] & ~(B | (1<<x))) == 0.
    # That's O(|point_to_subs[x]|) per candidate. |point_to_subs[x]| = C(v-1, t-1) = C(18,4) = 3060.
    # For v=19, k=8, t=5: per block we try up to (v - |block|) candidates, each O(3060).
    # That's manageable.

    best_blocks = None
    best_count = n_subs + 1

    # ---- Multi-restart randomized greedy ----
    max_restarts = 500
    rng = random.Random(12345)

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # uncovered[i] = 1 means subset i is NOT yet covered
        uncovered = bytearray([1] * n_subs)
        uncovered_count = n_subs
        blocks = []
        blocks_set = set()

        while uncovered_count > 0:
            if time.time() > deadline:
                break

            # Pick a target uncovered subset.
            # Strategy: with some probability pick the "most constrained" one,
            # otherwise pick randomly.
            # For simplicity and speed, pick randomly among uncovered.
            # To avoid O(n) scan every time, we keep a list of uncovered indices
            # but that's expensive to maintain. Instead, random probe.
            target_idx = -1
            # Random probe
            for _ in range(50):
                idx = rng.randint(0, n_subs - 1)
                if uncovered[idx]:
                    target_idx = idx
                    break
            if target_idx == -1:
                # Fallback scan
                for idx in range(n_subs):
                    if uncovered[idx]:
                        target_idx = idx
                        break
            if target_idx == -1:
                break

            # Start block with the target subset's elements
            start_sub = all_t_subs[target_idx]
            block_set = set(start_sub)
            block_mask = sub_masks[target_idx]

            # Greedily add points
            while len(block_set) < k:
                if time.time() > deadline:
                    break

                best_gain = -1
                best_point = None
                tie_points = []

                for x in range(v):
                    if x in block_set:
                        continue
                    # Compute gain: number of uncovered subsets S with x in S and S\{x} subset of block_mask
                    x_bit = 1 << x
                    gain = 0
                    for s_idx in point_to_subs[x]:
                        if not uncovered[s_idx]:
                            continue
                        sm = sub_masks[s_idx]
                        # sm contains x (guaranteed by point_to_subs), check sm\{x} subset of block_mask
                        if (sm & ~(block_mask | x_bit)) == 0:
                            gain += 1

                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        tie_points = [x]
                    elif gain == best_gain and gain > 0:
                        tie_points.append(x)

                if best_point is None or best_gain <= 0:
                    # No point adds coverage; pick random remaining
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_point = rng.choice(remaining)
                    else:
                        break

                block_set.add(best_point)
                block_mask |= (1 << best_point)

            block = sorted(block_set)
            block_tuple = tuple(block)
            if block_tuple in blocks_set:
                # Duplicate block, skip (shouldn't happen often with randomization)
                continue

            blocks_set.add(block_tuple)
            blocks.append(block)

            # Update uncovered: mark all t-subsets contained in block_mask as covered
            # For each subset i, check if sub_masks[i] & ~block_mask == 0
            # Only check those still uncovered for efficiency
            for i in range(n_subs):
                if uncovered[i]:
                    if (sub_masks[i] & ~block_mask) == 0:
                        uncovered[i] = 0
                        uncovered_count -= 1

            if uncovered_count == 0:
                break

        # Check completeness
        if uncovered_count == 0 and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            if best_count <= 335:
                break

    # Fallback: deterministic greedy if no complete covering found
    if best_blocks is None:
        uncovered = bytearray([1] * n_subs)
        uncovered_count = n_subs
        blocks = []
        while uncovered_count > 0:
            if time.time() > deadline:
                break
            # Pick first uncovered
            target_idx = -1
            for idx in range(n_subs):
                if uncovered[idx]:
                    target_idx = idx
                    break
            if target_idx == -1:
                break
            start_sub = all_t_subs[target_idx]
            block_set = set(start_sub)
            block_mask = sub_masks[target_idx]
            while len(block_set) < k:
                best_gain = -1
                best_point = None
                for x in range(v):
                    if x in block_set:
                        continue
                    x_bit = 1 << x
                    gain = 0
                    for s_idx in point_to_subs[x]:
                        if not uncovered[s_idx]:
                            continue
                        sm = sub_masks[s_idx]
                        if (sm & ~(block_mask | x_bit)) == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is None or best_gain <= 0:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_point = remaining[0]
                    else:
                        break
                block_set.add(best_point)
                block_mask |= (1 << best_point)
            block = sorted(block_set)
            blocks.append(block)
            for i in range(n_subs):
                if uncovered[i]:
                    if (sub_masks[i] & ~block_mask) == 0:
                        uncovered[i] = 0
                        uncovered_count -= 1
        best_blocks = blocks

    return best_blocks
