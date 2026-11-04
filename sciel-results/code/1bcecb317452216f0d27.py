"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time


def _combo_to_mask(combo):
    m = 0
    for x in combo:
        m |= (1 << x)
    return m


def construct(v, k, t):
    """Bitmask inverse-indexed greedy covering design with deadline guard."""
    deadline = time.time() + 145.0  # safety margin before 150s kill
    total_tsubsets = 0
    for r in itertools.combinations(range(v), t):
        total_tsubsets += 1

    # Map each t-subset (as frozenset of points) to a dense index 0..N-1
    tset_to_idx = {}
    idx_to_tset = []
    for combo in itertools.combinations(range(v), t):
        tset_to_idx[combo] = len(idx_to_tset)
        idx_to_tset.append(combo)
    N = len(idx_to_tset)

    # coverage[i] = number of blocks covering t-subset i; we need coverage[i] >= 1
    coverage = bytearray(N)  # counts can exceed 255? C(19,5)=11628, max coverage per tset is small in practice, but use array('H') to be safe
    import array
    coverage = array.array('H', [0]) * N

    # For incremental gain tracking: gain[x] = number of currently-uncovered t-subsets
    # that contain point x. When we add a block, we decrement coverage for all its
    # t-subsets; for each t-subset whose coverage drops from 1 to 0... wait, we go
    # from 0 to 1. Let's think: initially all 0. When we add a block, for each t-subset
    # in the block, coverage goes 0->1 (if was 0) or stays >=1.
    # gain[x] should count t-subsets with coverage==0 that contain x.
    # When coverage[i] goes from 0 to 1, for each point x in tset_to_idx[i], gain[x] -= 1.
    gain = [0] * v
    for i, combo in enumerate(idx_to_tset):
        for x in combo:
            gain[x] += 1

    blocks = []
    block_masks = []

    while True:
        # Check if all covered
        if all(c >= 1 for c in coverage):
            break
        if time.time() > deadline:
            break

        # Find an uncovered t-subset to anchor the new block
        # Pick the first uncovered one (or one with min coverage=0)
        anchor = None
        for i in range(N):
            if coverage[i] == 0:
                anchor = idx_to_tset[i]
                break
        if anchor is None:
            break

        # Build block greedily: start with anchor points, add point with max gain
        block_set = set(anchor)
        block_list = list(anchor)

        while len(block_list) < k:
            # Find point not in block with highest gain
            best_x = -1
            best_g = -1
            for x in range(v):
                if x not in block_set:
                    g = gain[x]
                    if g > best_g:
                        best_g = g
                        best_x = x
            if best_x < 0:
                break
            block_set.add(best_x)
            block_list.append(best_x)

        block_list.sort()
        blocks.append(block_list)
        block_masks.append(_combo_to_mask(block_list))

        # Update coverage and gain for all t-subsets in this block
        for tcombo in itertools.combinations(block_list, t):
            i = tset_to_idx[tcombo]
            if coverage[i] == 0:
                # This t-subset becomes covered; decrement gain for each point in it
                for x in tcombo:
                    gain[x] -= 1
            coverage[i] += 1

    # Redundancy removal pass: try to remove blocks whose t-subsets are all still covered
    if time.time() < deadline - 5.0:
        # Build list of which t-subset indices each block covers
        block_tsets = []
        for bl in blocks:
            ts = [tset_to_idx[tc] for tc in itertools.combinations(bl, t)]
            block_tsets.append(ts)

        # coverage2[i] = how many blocks cover t-subset i
        cov2 = array.array('H', [0]) * N
        for ts in block_tsets:
            for i in ts:
                cov2[i] += 1

        removed = set()
        # Try removing blocks in reverse order (later blocks added are more likely redundant)
        for bi in range(len(blocks) - 1, -1, -1):
            if bi in removed:
                continue
            if time.time() > deadline - 3.0:
                break
            # Check if removing block bi keeps all its t-subsets covered
            redundant = True
            for i in block_tsets[bi]:
                if cov2[i] <= 1:
                    redundant = False
                    break
            if redundant:
                removed.add(bi)
                for i in block_tsets[bi]:
                    cov2[i] -= 1

        if removed:
            blocks = [b for idx, b in enumerate(blocks) if idx not in removed]

    return blocks
# EVOLVE-BLOCK-END
