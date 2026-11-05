import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 120s

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    all_tsubsets_set = set(all_tsubsets)
    n_tsubsets = len(all_tsubsets)

    best_blocks = None
    best_len = float('inf')

    # Phase 1: Multiple greedy strategies to get good initial solutions
    trial = 0
    while time.time() < deadline - 20:  # leave 20s for polish
        trial += 1
        seed = trial * 7919 + 42

        # Strategy 1: Global greedy (consider all blocks at each step)
        blocks, valid = _greedy_cover_global(v, k, t, all_tsubsets_set, deadline, seed)
        if valid:
            blocks = _remove_redundant(blocks, v, k, t, deadline)
            blocks = _local_search(blocks, v, k, t, all_tsubsets_set, deadline, seed)
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
                if best_len <= 76:
                    break

        # Strategy 2: Incremental greedy with ratio scoring
        if time.time() < deadline - 20:
            blocks2, valid2 = _greedy_cover_ratio(v, k, t, all_tsubsets_set, deadline, seed + 1)
            if valid2:
                blocks2 = _remove_redundant(blocks2, v, k, t, deadline)
                blocks2 = _local_search(blocks2, v, k, t, all_tsubsets_set, deadline, seed + 1)
                if len(blocks2) < best_len:
                    best_len = len(blocks2)
                    best_blocks = blocks2
                    if best_len <= 76:
                        break

        # Strategy 3: Incremental greedy with absolute gain
        if time.time() < deadline - 20:
            blocks3, valid3 = _greedy_cover_abs(v, k, t, all_tsubsets_set, deadline, seed + 2)
            if valid3:
                blocks3 = _remove_redundant(blocks3, v, k, t, deadline)
                blocks3 = _local_search(blocks3, v, k, t, all_tsubsets_set, deadline, seed + 2)
                if len(blocks3) < best_len:
                    best_len = len(blocks3)
                    best_blocks = blocks3
                    if best_len <= 76:
                        break

    # Phase 2: Deep local search / WalkSAT-style refinement on the best solution
    if best_blocks is not None and time.time() < deadline:
        polished = _walksat_refine(best_blocks, v, k, t, all_tsubsets_set, deadline)
        if len(polished) < best_len:
            best_len = len(polished)
            best_blocks = polished

    if best_blocks is None:
        # Fallback: deterministic greedy
        blocks, valid = _greedy_cover_abs(v, k, t, all_tsubsets_set, deadline, -1)
        if valid:
            best_blocks = _remove_redundant(blocks, v, k, t, deadline)
        else:
            best_blocks = []

    return [list(b) for b in best_blocks]


def _greedy_cover_global(v, k, t, all_tsubsets_set, deadline, seed_val):
    """
    Global greedy set cover: at each step, pick the block that covers the most
    uncovered t-subsets. This avoids local optima of incremental construction.
    """
    rng = random.Random(seed_val)
    uncovered = set(all_tsubsets_set)
    blocks = []
    points = list(range(v))

    while uncovered:
        if time.time() > deadline:
            return None, False

        # Find the block that covers the most uncovered t-subsets
        # We don't enumerate all C(v,k) blocks, but we can be smart about it.
        # Pick a random uncovered t-subset and try to build a good block around it.
        # Then evaluate a few candidate blocks and pick the best.

        seed_tuple = tuple(rng.choice(list(uncovered)))
        best_block = None
        best_gain = -1
        best_gain_set = None

        # Generate several candidate blocks around the seed
        candidates = []
        for _ in range(20):
            block = list(seed_tuple)
            block_set = set(block)
            while len(block) < k:
                # Greedily add the point that maximizes gain
                t_minus_1_subsets = list(itertools.combinations(block, t - 1))
                best_x = -1
                best_x_gain = -1
                for x in points:
                    if x in block_set:
                        continue
                    gain = 0
                    for p in t_minus_1_subsets:
                        s = tuple(sorted(p + (x,)))
                        if s in uncovered:
                            gain += 1
                    if gain > best_x_gain:
                        best_x_gain = gain
                        best_x = x
                if best_x == -1:
                    remaining = [x for x in points if x not in block_set]
                    if not remaining:
                        break
                    best_x = rng.choice(remaining)
                block.append(best_x)
                block_set.add(best_x)
            block.sort()
            cand = tuple(block)
            candidates.append(cand)

        # Evaluate each candidate and pick the one with highest gain
        for cand in candidates:
            cand_tsubs = set(itertools.combinations(cand, t))
            gain = len(cand_tsubs & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_block = cand
                best_gain_set = cand_tsubs

        if best_block is None:
            return None, False

        blocks.append(best_block)
        uncovered -= best_gain_set

    return blocks, True


def _greedy_cover_ratio(v, k, t, all_tsubsets_set, deadline, seed_val):
    """
    Greedy covering using ratio-based scoring with randomness for diversity.
    """
    rng = random.Random(seed_val)
    uncovered = set(all_tsubsets_set)
    blocks = []
    points = list(range(v))

    while uncovered:
        if time.time() > deadline:
            return None, False

        seed = tuple(rng.choice(list(uncovered)))
        block = list(seed)
        block_set = set(block)

        while len(block) < k:
            if time.time() > deadline:
                return None, False

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

                score = gain + rng.random() * 0.1
                candidates.append((score, x))

            if not candidates:
                remaining = [x for x in points if x not in block_set]
                if not remaining:
                    return None, False
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

    return blocks, True


def _greedy_cover_abs(v, k, t, all_tsubsets_set, deadline, seed_val):
    """
    Greedy covering using absolute gain scoring.
    """
    rng = random.Random(seed_val if seed_val >= 0 else time.time())
    uncovered = set(all_tsubsets_set)
    blocks = []
    points = list(range(v))

    while uncovered:
        if time.time() > deadline:
            return None, False

        seed = tuple(rng.choice(list(uncovered)))
        block = list(seed)
        block_set = set(block)

        while len(block) < k:
            if time.time() > deadline:
                return None, False

            t_minus_1_subsets = list(itertools.combinations(block, t - 1))
            best_x = -1
            best_score = -1
            tiebreakers = []

            for x in points:
                if x in block_set:
                    continue

                gain = 0
                for p in t_minus_1_subsets:
                    s = tuple(sorted(p + (x,)))
                    if s in uncovered:
                        gain += 1

                if gain > best_score:
                    best_score = gain
                    best_x = x
                    tiebreakers = [x]
                elif gain == best_score and gain > 0:
                    tiebreakers.append(x)

            if best_x == -1:
                remaining = [x for x in points if x not in block_set]
                if not remaining:
                    return None, False
                chosen_x = rng.choice(remaining)
            else:
                if len(tiebreakers) > 1:
                    chosen_x = rng.choice(tiebreakers[:min(10, len(tiebreakers))])
                else:
                    chosen_x = best_x

            block.append(chosen_x)
            block_set.add(chosen_x)

        block.sort()
        blocks.append(tuple(block))

        for s in itertools.combinations(block, t):
            uncovered.discard(s)

    return blocks, True


def _remove_redundant(blocks, v, k, t, deadline):
    """
    Remove redundant blocks. A block is redundant if removing it
    doesn't leave any t-subset uncovered.
    """
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


def _local_search(blocks, v, k, t, all_tsubsets_set, deadline, seed_val):
    """
    Local search: try to replace blocks to reduce total count.
    """
    if not blocks:
        return blocks

    rng = random.Random(seed_val)
    improved = list(blocks)

    # Build cover count
    cover_count = defaultdict(int)
    for block in improved:
        for s in itertools.combinations(block, t):
            cover_count[s] += 1

    n_blocks = len(improved)
    max_iterations = 150

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

            for _ in range(25):
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
                any_change = True
                break

        if not any_change:
            break

    improved = _remove_redundant(improved, v, k, t, deadline)

    return improved


def _walksat_refine(blocks, v, k, t, all_tsubsets_set, deadline):
    """
    WalkSAT-style refinement: treat the problem as a Max-SAT-like instance.
    Each block is a Boolean variable (selected or not). Each t-subset coverage
    is a clause that must be satisfied. We minimize the number of selected blocks.
    
    Strategy: iteratively try to remove a block. If that leaves some t-subsets
    uncovered, try to modify other blocks to cover those t-subsets. If we can
    cover them without adding new blocks, we've reduced the count.
    """
    if not blocks:
        return blocks

    rng = random.Random(42)
    improved = list(blocks)
    best = improved
    best_len = len(improved)

    # Build cover count
    cover_count = defaultdict(int)
    for block in improved:
        for s in itertools.combinations(block, t):
            cover_count[s] += 1

    max_rounds = 30
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break

        # Try to remove each block and see if we can fix the coverage
        n = len(improved)
        for i in range(n):
            if time.time() > deadline:
                break

            old_block = improved[i]
            old_tsubs = list(itertools.combinations(old_block, t))

            # Temporarily remove the block
            for s in old_tsubs:
                cover_count[s] -= 1

            # Find which t-subsets are now uncovered
            uncovered = [s for s in old_tsubs if cover_count[s] == 0]

            if not uncovered:
                # Block was redundant, keep it removed
                improved.pop(i)
                best_len -= 1
                break

            # Try to modify other blocks to cover the uncovered t-subsets
            fixed = False
            for j in range(len(improved)):
                if j == i:
                    continue
                if time.time() > deadline:
                    break

                other_block = improved[j]
                # Try to modify other_block to cover one of the uncovered t-subsets
                for unc in uncovered:
                    # Generate a candidate block that includes unc
                    cand = list(unc)
                    cand_set = set(cand)
                    # Add random points to fill up to k
                    while len(cand) < k:
                        x = rng.randint(0, v - 1)
                        if x not in cand_set:
                            cand.append(x)
                            cand_set.add(x)
                    cand = tuple(sorted(cand))

                    # Check if this candidate covers all uncovered t-subsets
                    cand_tsubs = set(itertools.combinations(cand, t))
                    if all(s in cand_tsubs for s in uncovered):
                        # Replace other_block with cand
                        for s in itertools.combinations(other_block, t):
                            cover_count[s] -= 1
                        for s in itertools.combinations(cand, t):
                            cover_count[s] += 1
                        improved[j] = cand
                        fixed = True
                        break
                    if fixed:
                        break

            if not fixed:
                # Restore the block
                for s in old_tsubs:
                    cover_count[s] += 1
            else:
                # Block was removed and coverage fixed
                break

        # Remove any newly redundant blocks
        improved = _remove_redundant(improved, v, k, t, deadline)

        if len(improved) < best_len:
            best_len = len(improved)
            best = improved

    return best
