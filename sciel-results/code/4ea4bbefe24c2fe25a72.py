import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    all_tsubsets = list(itertools.combinations(range(v), t))
    all_tsubsets_set = set(all_tsubsets)
    n_tsub = len(all_tsubsets_set)

    best_blocks = None
    best_len = float('inf')

    trial = 0
    while time.time() < deadline:
        trial += 1
        blocks = _greedy_cover(v, k, t, all_tsubsets_set, deadline, trial)
        if blocks is not None:
            # Phase 1: Remove redundant blocks
            blocks = _remove_redundant(blocks, v, k, t, deadline)

            # Phase 2: Local search / refinement
            blocks = _local_search(blocks, v, k, t, all_tsubsets_set, deadline, trial)

            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks

        # Early termination if we hit the known best
        if best_len <= 76:
            break

    if best_blocks is None:
        return []

    return [list(b) for b in best_blocks]


def _greedy_cover(v, k, t, all_tsubsets_set, deadline, seed_val):
    """
    Greedy set cover with vectorized gain calculation.
    """
    rng = random.Random(seed_val)
    uncovered = set(all_tsubsets_set)
    blocks = []
    points = list(range(v))

    # Pre-compute all (t-1)-subsets for each point? No, too many.
    # Instead, for each candidate point x, we need to check which (t-1)-subsets of the current block
    # combined with x form a t-subset that is in `uncovered`.

    while uncovered:
        if time.time() > deadline:
            return None

        # Pick a random uncovered t-subset as the seed for the block
        seed_tuple = tuple(rng.choice(list(uncovered)))
        current_block_set = set(seed_tuple)

        # Greedily add points to maximize coverage of *uncovered* t-subsets
        while len(current_block_set) < k:
            if time.time() > deadline:
                return None

            best_point = -1
            best_gain = -1
            best_tiebreak = []

            # Pre-compute (t-1) subsets of current block
            t_minus_1_subsets = list(itertools.combinations(sorted(current_block_set), t - 1))

            # Vectorize the gain calculation
            # For each point x not in current_block_set, compute gain
            # gain(x) = number of sub in t_minus_1_subsets such that tuple(sorted(sub + (x,))) in uncovered

            # Since v=24, k=7, t=3, this is small enough to do in Python loops,
            # but we can optimize by pre-computing the set of uncovered (t-1)-subsets for each point?
            # Let's stick to the loop but make it tight.

            for x in points:
                if x in current_block_set:
                    continue

                gain = 0
                for sub in t_minus_1_subsets:
                    new_subset = tuple(sorted(sub + (x,)))
                    if new_subset in uncovered:
                        gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    best_tiebreak = [x]
                elif gain == best_gain and gain > 0:
                    best_tiebreak.append(x)

            if best_point == -1:
                available = [x for x in points if x not in current_block_set]
                if not available:
                    return None
                best_point = rng.choice(available)

            current_block_set.add(best_point)

        block = tuple(sorted(current_block_set))
        blocks.append(block)

        # Remove all t-subsets covered by this new block
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)

    return blocks


def _remove_redundant(blocks, v, k, t, deadline):
    """
    Iteratively removes blocks that are redundant.
    """
    if not blocks:
        return blocks

    cover_count = {}
    for block in blocks:
        for sub in itertools.combinations(block, t):
            cover_count[sub] = cover_count.get(sub, 0) + 1

    improved = list(blocks)
    changed = True

    while changed:
        if time.time() > deadline:
            break
        changed = False
        for i in range(len(improved) - 1, -1, -1):
            if time.time() > deadline:
                break
            block = improved[i]
            is_redundant = True
            for sub in itertools.combinations(block, t):
                if cover_count.get(sub, 0) <= 1:
                    is_redundant = False
                    break

            if is_redundant:
                for sub in itertools.combinations(block, t):
                    cover_count[sub] -= 1
                    if cover_count[sub] == 0:
                        del cover_count[sub]
                improved.pop(i)
                changed = True

    return improved


def _local_search(blocks, v, k, t, all_tsubsets_set, deadline, seed_val):
    """
    Attempts to improve the covering by replacing blocks to enable removal of others.
    """
    if not blocks:
        return blocks

    rng = random.Random(seed_val)
    improved = list(blocks)

    block_tsubs = []
    for b in improved:
        block_tsubs.append(list(itertools.combinations(b, t)))

    cover_count = {}
    for subs in block_tsubs:
        for sub in subs:
            cover_count[sub] = cover_count.get(sub, 0) + 1

    n_blocks = len(improved)
    points = list(range(v))

    max_iter = 50
    for _ in range(max_iter):
        if time.time() > deadline:
            break

        # Recompute cover count and unique subsets
        cover_count = {}
        for i in range(len(improved)):
            for sub in block_tsubs[i]:
                cover_count[sub] = cover_count.get(sub, 0) + 1

        unique_subs = [None] * len(improved)
        for i in range(len(improved)):
            unique_subs[i] = [s for s in block_tsubs[i] if cover_count.get(s, 0) == 1]

        redundant_found = False
        for i in range(len(improved)):
            if not unique_subs[i]:
                for sub in block_tsubs[i]:
                    cover_count[sub] -= 1
                    if cover_count[sub] == 0:
                        del cover_count[sub]
                del improved[i]
                del block_tsubs[i]
                redundant_found = True
                break

        if redundant_found:
            continue

        n = len(improved)
        if n == 0:
            break

        b_idx = rng.randint(0, n - 1)
        b_unique = unique_subs[b_idx]

        a_found = False
        a_candidates = list(range(n))
        rng.shuffle(a_candidates)

        for a_idx in a_candidates:
            if a_idx == b_idx:
                continue

            a_unique = unique_subs[a_idx]

            union_points = set()
            for s in a_unique:
                union_points.update(s)
            for s in b_unique:
                union_points.update(s)

            if len(union_points) <= k:
                a_prime = list(union_points)
                while len(a_prime) < k:
                    p = rng.randint(0, v - 1)
                    if p not in a_prime:
                        a_prime.append(p)

                a_prime = tuple(sorted(a_prime))

                # Remove A's contribution
                for sub in block_tsubs[a_idx]:
                    cover_count[sub] -= 1
                    if cover_count[sub] == 0:
                        del cover_count[sub]

                # Add A' contribution
                new_tsubs = list(itertools.combinations(a_prime, t))
                for sub in new_tsubs:
                    cover_count[sub] = cover_count.get(sub, 0) + 1

                improved[a_idx] = a_prime
                block_tsubs[a_idx] = new_tsubs

                b_redundant = True
                for sub in block_tsubs[b_idx]:
                    if cover_count.get(sub, 0) <= 1:
                        b_redundant = False
                        break

                if b_redundant:
                    for sub in block_tsubs[b_idx]:
                        cover_count[sub] -= 1
                        if cover_count[sub] == 0:
                            del cover_count[sub]
                    del improved[b_idx]
                    del block_tsubs[b_idx]
                    a_found = True
                    break

        if a_found:
            continue

    return improved
