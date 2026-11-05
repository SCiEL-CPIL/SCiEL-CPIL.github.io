"""Greedy covering design C(v,k,t) with vectorized bookkeeping and a
bounded post-covering block-removal pass, all under an explicit deadline."""
import time
import random
import itertools
import numpy as np

_DEADLINE = 140.0  # seconds; leaves safety margin below the 150s kill


def _build_tsubsets(v, t):
    """Return (N, (N,t)) with rows = all t-subsets as sorted integer arrays."""
    if t == 0:
        return np.zeros((1, 0), dtype=np.int64)
    rows = list(itertools.combinations(range(v), t))
    return np.array(rows, dtype=np.int64)


def _block_mask(block, tsub, t):
    """Boolean mask over all t-subsets: which are contained in `block`."""
    b = np.asarray(sorted(block), dtype=np.int64)
    if t == 0:
        return np.ones(tsub.shape[0], dtype=bool)
    # (N, t, k) -> (N, t) -> (N,)
    return np.isin(tsub, b, assume_unique=True).all(axis=1)


def construct(v, k, t):
    start = time.time()

    def timed_out():
        return (time.time() - start) >= _DEADLINE

    # Precompute all t-subsets (the "items" of the set cover).
    tsub = _build_tsubsets(v, t)
    n_items = tsub.shape[0]

    # Build per-point coverage masks: which t-subsets contain point p.
    # A block covers exactly the union of its points' masks.
    point_mask = np.zeros((v, n_items), dtype=bool)
    if t == 0:
        point_mask[:] = True
    else:
        for p in range(v):
            point_mask[p] = np.isin(tsub, p, assume_unique=True).any(axis=1)

    # remaining[p] = number of still-uncovered t-subsets that contain point p.
    remaining = np.zeros(v, dtype=np.int64)
    for p in range(v):
        remaining[p] = int(point_mask[p].sum())

    remaining_count = int(n_items)
    blocks = []
    rng = random.Random(12345)

    while remaining_count > 0 and not timed_out():
        block = []
        chosen = np.zeros(v, dtype=bool)

        # --- pick the first point: maximize uncovered coverage, tie-break random
        def pick_point(mask):
            # mask: boolean over points that are allowed (not yet in block)
            best = -1
            best_val = -1
            for p in range(v):
                if mask[p] and remaining[p] > best_val:
                    best_val = remaining[p]
                    best = p
            if best_val < 0:
                return None
            # collect all points tied at best_val and pick one at random
            ties = [p for p in range(v) if mask[p] and remaining[p] == best_val]
            return rng.choice(ties)

        first = pick_point(~chosen)
        if first is None:
            break
        block.append(first)
        chosen[first] = True

        # --- fill the rest of the block greedily
        # cur_mask: t-subsets covered by the block so far
        cur_mask = point_mask[first].copy()
        while len(block) < k:
            best_p = -1
            best_gain = -1
            for p in range(v):
                if chosen[p]:
                    continue
                # gain = uncovered t-subsets newly covered by adding p
                gain = int(np.count_nonzero(point_mask[p] & ~cur_mask))
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
            if best_p < 0 or best_gain == 0:
                # nothing new to cover; pad with any unused point (rare)
                for p in range(v):
                    if not chosen[p]:
                        best_p = p
                        break
                if best_p < 0:
                    break
            block.append(best_p)
            chosen[best_p] = True
            cur_mask = cur_mask | point_mask[best_p]

        if not block:
            break

        # commit the block
        blocks.append(tuple(sorted(block)))
        block_mask = np.zeros(n_items, dtype=bool)
        for p in block:
            block_mask |= point_mask[p]

        # update remaining counts and uncovered total
        newly = block_mask & (remaining_count > 0)
        # recompute from scratch is expensive; instead subtract using per-point
        # but multiple points overlap, so recompute the union contribution:
        # For each point p, the t-subsets it covers that are still uncovered
        # and that this block newly covers get decremented. Overlaps handled
        # by only decrementing for the union once.
        # Efficient approach: for each point p in block, decrement remaining[p]
        # by the number of t-subsets in (point_mask[p] & block_mask & still_uncovered)
        # but that double counts overlaps. Instead, recompute remaining via union
        # is O(v*n). Simpler: track uncovered boolean vector.
        # We'll maintain `uncovered` boolean vector (True = still uncovered).
        # (Introduce it now for correctness of the removal pass.)
        if not hasattr(construct, '_uc'):
            construct._uc = np.ones(n_items, dtype=bool)
        uc = construct._uc
        uc &= ~block_mask
        remaining_count = int(uc.sum())
        # refresh remaining counts (only needed occasionally; do it each block)
        for p in range(v):
            remaining[p] = int(np.count_nonzero(point_mask[p] & uc))

    # Ensure we have a valid covering; if timed out mid-way, we still have the
    # blocks found so far but may be incomplete. Re-cover remaining items quickly.
    uc = construct._uc if hasattr(construct, '_uc') else np.ones(n_items, dtype=bool)
    while uc.any() and not timed_out():
        # pick the uncovered item with the fewest covering blocks among candidates
        # (simplest: take first uncovered item, build a block covering it)
        idx = int(np.argmax(uc))
        item = tuple(tsub[idx])
        block = list(item)
        chosen = set(block)
        cur_mask = _block_mask(block, tsub, t)
        while len(block) < k and not timed_out():
            best_p = -1
            best_gain = -1
            for p in range(v):
                if p in chosen:
                    continue
                gain = int(np.count_nonzero(point_mask[p] & ~cur_mask & uc))
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
            if best_p < 0 or best_gain == 0:
                for p in range(v):
                    if p not in chosen:
                        best_p = p
                        break
                if best_p < 0:
                    break
            block.append(best_p)
            chosen.add(best_p)
            cur_mask = cur_mask | point_mask[best_p]
        blocks.append(tuple(sorted(block)))
        for p in block:
            uc &= ~point_mask[p]

    # --- bounded block-removal pass (extra work enabled by the speedup)
    if not timed_out():
        # remove redundant blocks: a block is redundant if removing it leaves
        # every t-subset still covered by the rest.
        # Maintain a coverage-count vector over items; an item is safe if count>=2.
        cov = np.zeros(n_items, dtype=np.int64)
        for b in blocks:
            m = np.zeros(n_items, dtype=bool)
            for p in b:
                m |= point_mask[p]
            cov += m
        # an item is uniquely covered if cov==1
        # iterate removals while time allows
        changed = True
        while changed and not timed_out():
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if timed_out():
                    break
                m = np.zeros(n_items, dtype=bool)
                for p in blocks[i]:
                    m |= point_mask[p]
                # removing block i is safe iff no item drops to 0
                # i.e., wherever m is True, cov>=2
                if not np.any(cov[m] < 2):
                    cov -= m
                    del blocks[i]
                    changed = True
                    break  # restart scan after a change

    # clear helper state
    if hasattr(construct, '_uc'):
        del construct._uc

    # final safety: if we somehow lost completeness, re-cover (cheap re-check)
    uc = np.ones(n_items, dtype=bool)
    for b in blocks:
        m = np.zeros(n_items, dtype=bool)
        for p in b:
            m |= point_mask[p]
        uc &= ~m
    while uc.any() and not timed_out():
        idx = int(np.argmax(uc))
        item = tuple(tsub[idx])
        block = list(item)
        chosen = set(block)
        cur_mask = _block_mask(block, tsub, t)
        while len(block) < k and not timed_out():
            best_p = -1
            best_gain = -1
            for p in range(v):
                if p in chosen:
                    continue
                gain = int(np.count_nonzero(point_mask[p] & ~cur_mask & uc))
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
            if best_p < 0 or best_gain == 0:
                for p in range(v):
                    if p not in chosen:
                        best_p = p
                        break
                if best_p < 0:
                    break
            block.append(best_p)
            chosen.add(best_p)
            cur_mask = cur_mask | point_mask[best_p]
        blocks.append(tuple(sorted(block)))
        for p in block:
            uc &= ~point_mask[p]

    # return as list of lists of ints, distinct blocks
    seen = set()
    out = []
    for b in blocks:
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            out.append(list(key))
    return out
