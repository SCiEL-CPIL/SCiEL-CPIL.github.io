"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time


def construct(v, k, t):
    # Pairwise-greedy set cover: at each step evaluate all candidate k-subsets
    # and pick the one with maximum gain in uncovered t-subsets.
    # This avoids the myopic 'open-then-grow' trap.
    deadline = time.time() + 110.0  # safety margin before 120s limit

    # All t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # uncovered[i] = 1 if t-subset i is still uncovered
    uncovered = bytearray([1]) * n_tsubs
    remaining = n_tsubs

    # For each point x, precompute list of t-subset indices containing x
    point_to_tsubs = [[] for _ in range(v)]
    for i, sub in enumerate(all_tsubs):
        for x in sub:
            point_to_tsubs[x].append(i)

    blocks = []
    used_blocks = set()

    # Precompute all candidate k-subsets for efficiency
    # For v=19, k=7: C(19,7) = 50388 candidates
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # For each k-subset, precompute which t-subsets it covers
    # ksub_to_tsubs[j] = list of t-subset indices covered by k-subset j
    ksub_to_tsubs = []
    for ks in all_ksubs:
        ks_set = set(ks)
        covered = []
        for i, sub in enumerate(all_tsubs):
            if all(x in ks_set for x in sub):
                covered.append(i)
        ksub_to_tsubs.append(covered)

    while remaining > 0:
        if time.time() > deadline:
            break

        best_j = -1
        best_gain = -1
        best_tsubs = None

        for j, ks in enumerate(all_ksubs):
            if list(ks) in used_blocks:
                continue
            # Count uncovered t-subsets covered by this k-subset
            gain = 0
            for ti in ksub_to_tsubs[j]:
                if uncovered[ti]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_j = j
                best_tsubs = ksub_to_tsubs[j]

        if best_j == -1 or best_gain <= 0:
            # No candidate improves; break to avoid infinite loop
            break

        block = list(all_ksubs[best_j])
        blocks.append(block)
        used_blocks.add(tuple(block))

        # Mark covered t-subsets
        for ti in best_tsubs:
            if uncovered[ti]:
                uncovered[ti] = 0
                remaining -= 1

    # If we didn't finish in time, do a fast greedy repair
    if remaining > 0:
        uncovered_set = set()
        for i, sub in enumerate(all_tsubs):
            if uncovered[i]:
                uncovered_set.add(sub)

        while uncovered_set:
            if time.time() > deadline:
                break
            block = list(min(uncovered_set))
            block_set = set(block)
            while len(block) < k:
                if time.time() > deadline:
                    break
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        sub = tuple(sorted(rest + (x,)))
                        if sub in uncovered_set:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    for x in range(v):
                        if x not in block_set:
                            best_x = x
                            break
                block.append(best_x)
                block_set.add(best_x)
            if len(block) < k:
                break
            block.sort()
            if tuple(block) not in used_blocks:
                blocks.append(block)
                used_blocks.add(tuple(block))
            uncovered_set.difference_update(itertools.combinations(block, t))

    return blocks
# EVOLVE-BLOCK-END
