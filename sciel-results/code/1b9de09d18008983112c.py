"""Greedy covering + local search (redundancy removal and 1-for-1 swaps) for C(19,7,4)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

_DEADLINE = 150.0
_DEADLINE_MARGIN = 8.0  # seconds of safety margin before the 150s kill


def _time_ok(deadline):
    return (time.time() < deadline - _DEADLINE_MARGIN)


def _all_tsubs(v, t):
    return list(itertools.combinations(range(v), t))


def _validate(v, k, t, blocks):
    """Exhaustively check every t-subset is covered; return (ok, count)."""
    if not blocks:
        return (v == 0), 0
    cover = set()
    for b in blocks:
        if len(b) != k or len(set(b)) != k:
            return False, -1
        for x in b:
            if x < 0 or x >= v:
                return False, -1
        for sub in itertools.combinations(b, t):
            cover.add(sub)
    total = len(_all_tsubs(v, t))
    return (len(cover) == total), len(cover)


def _greedy(v, k, t, deadline, rng):
    """Greedy set-cover with randomized tie-breaking; builds a valid covering."""
    universe = _all_tsubs(v, t)
    covered = [False] * len(universe)
    idx_of = {sub: i for i, sub in enumerate(universe)}
    blocks = []
    remaining = len(universe)

    while remaining > 0 and _time_ok(deadline):
        # First uncovered t-subset
        first_uncovered = None
        for i in range(len(covered)):
            if not covered[i]:
                first_uncovered = i
                break
        if first_uncovered is None:
            break
        start = set(universe[first_uncovered])

        # Build block: repeatedly add the point that covers the most uncovered t-subsets
        block = list(start)
        while len(block) < k:
            best_x = None
            best_gain = -1
            best_ties = []
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    cand = tuple(sorted(rest + (x,)))
                    ci = idx_of.get(cand)
                    if ci is not None and not covered[ci]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    best_ties = [x]
                elif gain == best_gain and gain > 0:
                    best_ties.append(x)
            if best_gain <= 0:
                # fill arbitrarily
                for x in range(v):
                    if x not in block:
                        block.append(x)
                break
            # randomized tie-break
            block.append(best_ties[rng.randrange(len(best_ties))])

        block.sort()
        blocks.append(block)
        for sub in itertools.combinations(block, t):
            ci = idx_of[sub]
            if not covered[ci]:
                covered[ci] = True
                remaining -= 1
    return blocks, covered, idx_of


def _compute_membership(v, k, t, blocks, idx_of):
    """For each t-subset, the list of block indices covering it."""
    n = len(_all_tsubs(v, t))
    members = [[] for _ in range(n)]
    for bi, b in enumerate(blocks):
        for sub in itertools.combinations(b, t):
            ci = idx_of[sub]
            members[ci].append(bi)
    return members


def _local_search(v, k, t, blocks, covered, idx_of, deadline, rng):
    """Redundancy removal + 1-for-1 swaps to reduce block count."""
    n = len(covered)
    # recompute membership from current blocks
    members = _compute_membership(v, k, t, blocks, idx_of)

    improved = True
    while improved and _time_ok(deadline):
        improved = False

        # --- Pass A: remove redundant blocks ---
        i = 0
        while i < len(blocks) and _time_ok(deadline):
            bi = i
            # check if every t-subset of this block is covered by another block
            redundant = True
            for sub in itertools.combinations(blocks[bi], t):
                ci = idx_of[sub]
                cnt = 0
                for m in members[ci]:
                    if m == bi:
                        continue
                    cnt += 1
                    if cnt >= 1:
                        break
                if cnt == 0:
                    redundant = False
                    break
            if redundant:
                # remove block bi
                for sub in itertools.combinations(blocks[bi], t):
                    ci = idx_of[sub]
                    members[ci].remove(bi)
                blocks.pop(bi)
                # decrement all block indices in members that are > bi
                for lst in members:
                    for j in range(len(lst)):
                        if lst[j] > bi:
                            lst[j] -= 1
                improved = True
                # do not advance i; re-examine same position
                continue
            i += 1

        if not _time_ok(deadline):
            break

        # --- Pass B: 1-for-1 swaps ---
        # For each block, try replacing it with a candidate that covers more
        # "unique" t-subsets (t-subsets only covered by the current block).
        nblocks = len(blocks)
        for bi in range(nblocks):
            if not _time_ok(deadline):
                break
            # unique t-subsets of this block (only covered by bi)
            unique_count = 0
            for sub in itertools.combinations(blocks[bi], t):
                ci = idx_of[sub]
                if len(members[ci]) == 1:
                    unique_count += 1
            if unique_count == 0:
                continue
            # candidate: pick a point not in block, form a new block, count how many
            # of the current block's unique t-subsets it would cover
            cur = set(blocks[bi])
            best_gain = 0
            best_newblock = None
            for x in range(v):
                if x in cur:
                    continue
                newblock = list(cur)
                newblock.append(x)
                # count unique t-subsets of newblock that are currently unique to bi
                gain = 0
                for sub in itertools.combinations(sorted(newblock), t):
                    ci = idx_of[sub]
                    if len(members[ci]) == 1:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_newblock = sorted(newblock)
            if best_gain > 0 and best_newblock is not None:
                # apply swap
                old = blocks[bi]
                for sub in itertools.combinations(old, t):
                    ci = idx_of[sub]
                    members[ci].remove(bi)
                blocks[bi] = best_newblock
                for sub in itertools.combinations(best_newblock, t):
                    ci = idx_of[sub]
                    members[ci].append(bi)
                improved = True
    return blocks


def construct(v, k, t):
    deadline = time.time() + _DEADLINE
    rng = random.Random(12345)

    # Build an initial greedy covering
    blocks, covered, idx_of = _greedy(v, k, t, deadline, rng)

    # Local search to reduce block count
    blocks = _local_search(v, k, t, blocks, covered, idx_of, deadline, rng)

    # If we ran out of time during greedy and the covering is incomplete,
    # finish it quickly (greedy is cheap per block).
    while _time_ok(deadline):
        ok, cnt = _validate(v, k, t, blocks)
        if ok:
            break
        # find an uncovered t-subset and add a block
        cover = set()
        for b in blocks:
            for sub in itertools.combinations(b, t):
                cover.add(sub)
        target = None
        for sub in _all_tsubs(v, t):
            if sub not in cover:
                target = sub
                break
        if target is None:
            break
        block = list(target)
        while len(block) < k:
            # add point covering most uncovered t-subsets
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    cand = tuple(sorted(rest + (x,)))
                    if cand not in cover:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_gain <= 0:
                for x in range(v):
                    if x not in block:
                        block.append(x)
                break
            block.append(best_x)
        block.sort()
        blocks.append(block)

    # Final validation; if invalid (should not happen), return as-is
    return blocks
# EVOLVE-BLOCK-END
