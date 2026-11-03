"""Covering design C(19,7,4): greedy with multiple randomized trials + local search polish."""
import itertools
import random
import time


def _all_tsubs(v, t):
    return set(itertools.combinations(range(v), t))


def _tsubs_of_block(block, t):
    return set(itertools.combinations(block, t))


def _greedy_once(v, k, t, uncovered, rng, deadline):
    """One greedy pass. Returns list of blocks (sorted), possibly None if time up."""
    blocks = []
    start = time.time()
    while uncovered:
        if time.time() >= deadline:
            return None
        # pick a t-subset to cover: first uncovered (deterministic) but shuffle start
        # Use a random uncovered element to diversify
        first = next(iter(uncovered))
        block = list(first)
        block_set = set(block)
        while len(block) < k:
            best_x = -1
            best_gain = -1
            # candidate points not in block
            for x in range(v):
                if x in block_set:
                    continue
                # gain = number of uncovered t-subsets contained in block ∪ {x} that contain x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    if sub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x < 0:
                # no gain; pick a random unused point to finish
                remaining = [x for x in range(v) if x not in block_set]
                best_x = rng.choice(remaining)
            block.append(best_x)
            block_set.add(best_x)
        block.sort()
        blocks.append(block)
        # remove covered t-subsets
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)
    return blocks


def _remove_redundant(blocks, v, k, t):
    """Remove blocks that are redundant (all their t-subsets covered by others)."""
    n = len(blocks)
    # precompute tsub -> list of block indices containing it
    tsub_to_blocks = {}
    for bi, b in enumerate(blocks):
        for sub in itertools.combinations(b, t):
            tsub_to_blocks.setdefault(sub, []).append(bi)
    keep = [True] * n
    for bi in range(n):
        if not keep[bi]:
            continue
        # check if removing block bi still covers all its t-subsets
        redundant = True
        for sub in itertools.combinations(blocks[bi], t):
            others = [j for j in tsub_to_blocks[sub] if j != bi and keep[j]]
            if not others:
                redundant = False
                break
        if redundant:
            keep[bi] = False
    return [blocks[i] for i in range(n) if keep[i]]


def _local_search(blocks, v, k, t, rng, deadline, max_passes=3):
    """1-for-1 swaps: try replacing a block with a new one that allows removing another."""
    current = list(blocks)
    for _pass in range(max_passes):
        if time.time() >= deadline:
            break
        improved = False
        # drop redundant blocks first
        before = len(current)
        current = _remove_redundant(current, v, k, t)
        if len(current) < before:
            improved = True
        # try 1-for-1: for a random block, try to find a replacement that covers
        # its unique t-subsets and lets us drop a different block
        # Simplify: try removing one block and greedily re-covering its unique t-subsets
        # with a single new block (if possible), keeping total count same or lower.
        if len(current) > 1:
            # build coverage count
            tsub_count = {}
            for b in current:
                for sub in itertools.combinations(b, t):
                    tsub_count[sub] = tsub_count.get(sub, 0) + 1
            # find a block whose removal leaves only a few uncovered t-subsets
            # that can be covered by one new block
            order = list(range(len(current)))
            rng.shuffle(order)
            for bi in order:
                if time.time() >= deadline:
                    break
                # uncovered t-subsets if we drop block bi
                uncov = []
                for sub in itertools.combinations(current[bi], t):
                    if tsub_count.get(sub, 0) <= 1:
                        uncov.append(sub)
                if not uncov:
                    # can drop entirely
                    current = [b for i, b in enumerate(current) if i != bi]
                    improved = True
                    break
                # try to build one block covering all uncov
                # uncov t-subsets must all fit in a k-block
                union = set()
                for s in uncov:
                    union.update(s)
                if len(union) > k:
                    continue
                # build block = union plus fillers
                block = sorted(union)
                block_set = set(block)
                while len(block) < k:
                    # add a point that maximizes coverage of uncov (already all covered by union)
                    # just add arbitrary
                    for x in range(v):
                        if x not in block_set:
                            block.append(x)
                            block_set.add(x)
                            break
                block.sort()
                # verify it covers all uncov
                bset_tsubs = set(itertools.combinations(block, t))
                if all(s in bset_tsubs for s in uncov):
                    new_current = [b for i, b in enumerate(current) if i != bi]
                    new_current.append(block)
                    current = new_current
                    improved = True
                    break
        if not improved:
            break
    return current


def construct(v, k, t):
    deadline = time.time() + 115.0  # safety margin under 120s
    all_tsubs = _all_tsubs(v, t)
    best_blocks = None
    best_len = float('inf')
    rng = random.Random(12345)

    # First deterministic greedy
    unc = set(all_tsubs)
    res = _greedy_once(v, k, t, unc, rng, deadline)
    if res is not None:
        best_blocks = res
        best_len = len(res)

    # Multiple randomized trials
    trial = 0
    while time.time() < deadline - 5.0:
        trial += 1
        # shuffle the order of uncovered by randomizing which first we pick
        # We'll just rerun greedy with a different rng seed and random tie-breaks
        unc = set(all_tsubs)
        res = _greedy_once(v, k, t, unc, rng, deadline)
        if res is None:
            break
        if len(res) < best_len:
            best_blocks = res
            best_len = len(res)

    if best_blocks is None:
        # fallback: return a trivial (possibly invalid) but should not happen
        return []

    # Polish
    if time.time() < deadline - 3.0:
        best_blocks = _local_search(best_blocks, v, k, t, rng, deadline, max_passes=3)

    # Final redundancy removal
    best_blocks = _remove_redundant(best_blocks, v, k, t)

    # Ensure distinct and valid format
    seen = set()
    out = []
    for b in best_blocks:
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            out.append(list(key))
    return out
