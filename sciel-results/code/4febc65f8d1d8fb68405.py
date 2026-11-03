"""Rewritten covering solver for C(v,k,t) with deadline guard and faster greedy."""
# EVOLVE-BLOCK-START
import itertools
import time

# Hard wall-clock limit for the evaluator is ~150s; keep a safety margin.
DEADLINE = 120.0


def _all_tsubsets(v, t):
    """Return a list of all t-subsets of {0..v-1} as sorted tuples."""
    return list(itertools.combinations(range(v), t))


def _build_incidence(blocks, v, t):
    """
    For each block, compute the set of t-subset indices it covers.
    Returns list of sets of ints (indices into the t-subset list).
    """
    # Map each t-subset tuple to its index
    tset_to_idx = {}
    for idx, ts in enumerate(itertools.combinations(range(v), t)):
        tset_to_idx[ts] = idx

    inc = []
    for b in blocks:
        covered = set()
        for ts in itertools.combinations(b, t):
            idx = tset_to_idx[ts]
            covered.add(idx)
        inc.append(covered)
    return inc


def _greedy_step(uncovered_set, blocks, v, k, t, tset_to_idx, inc):
    """
    Greedily extend a partial block to size k, maximizing newly covered t-subsets.
    Returns the completed block (sorted list of ints).
    """
    block = list(min(uncovered_set))  # start with the lexicographically smallest uncovered t-subset
    # Track which t-subsets are currently covered by existing blocks
    # We'll use a boolean array for speed: covered[i] = True if t-subset i is in any existing block
    n_tsubsets = len(tset_to_idx)
    covered_arr = bytearray(n_tsubsets)
    for cov in inc:
        for idx in cov:
            covered_arr[idx] = 1

    # The block currently covers some t-subsets; we want to add points to cover more uncovered ones.
    # For a partial block B, the t-subsets it can cover are those contained in B.
    # When we add point x, new t-subsets covered are those containing x and t-1 elements from B.

    while len(block) < k:
        best_x = -1
        best_gain = -1
        for x in range(v):
            if x in block:
                continue
            # Count how many uncovered t-subsets would be newly covered by adding x
            gain = 0
            for rest in itertools.combinations(block, t - 1):
                ts = tuple(sorted(rest + (x,)))
                idx = tset_to_idx.get(ts)
                if idx is not None and not covered_arr[idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_x = x
        if best_x == -1:
            break
        block.append(best_x)

    block.sort()
    return block


def construct(v, k, t):
    """
    Construct a covering of t-subsets of {0..v-1} by k-subsets (blocks).
    Returns a list of blocks (each a sorted list of k distinct ints in 0..v-1).
    """
    start_time = time.time()

    # Generate all t-subsets
    all_tsubsets = _all_tsubsets(v, t)
    n_tsubsets = len(all_tsubsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Uncovered t-subsets as a set of indices
    uncovered = set(range(n_tsubsets))

    blocks = []
    inc = []  # incidence: list of sets of t-subset indices covered by each block

    while uncovered:
        # Check deadline
        if time.time() - start_time > DEADLINE - 5:
            break

        # Pick the lexicographically smallest uncovered t-subset to seed a new block
        min_idx = min(uncovered)
        min_ts = all_tsubsets[min_idx]

        # Build a new block greedily
        block = list(min_ts)
        # covered_arr: which t-subsets are already covered by existing blocks
        covered_arr = bytearray(n_tsubsets)
        for cov in inc:
            for idx in cov:
                covered_arr[idx] = 1

        # Extend block to size k
        while len(block) < k:
            if time.time() - start_time > DEADLINE - 5:
                break
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    idx = tset_to_idx.get(ts)
                    if idx is not None and not covered_arr[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1:
                break
            block.append(best_x)

        block.sort()

        # Compute which t-subsets this block covers
        block_tsubsets = set()
        for ts in itertools.combinations(block, t):
            idx = tset_to_idx[ts]
            block_tsubsets.add(idx)

        # Add to blocks and incidence
        blocks.append(block)
        inc.append(block_tsubsets)

        # Update uncovered
        uncovered -= block_tsubsets

    return blocks
# EVOLVE-BLOCK-END
