"""Fresh design: bitmask greedy covering with randomized restarts + reduction."""
import time
import random
import itertools

_DEADLINE = 145.0  # safety margin under the 150s wall-clock kill


def construct(v, k, t):
    start = time.time()

    def remaining():
        return _DEADLINE - (time.time() - start)

    points = list(range(v))
    all_t = list(itertools.combinations(points, t))
    n_t = len(all_t)
    index = {c: i for i, c in enumerate(all_t)}

    # Precompute, for each point x, the bitmask of all t-subsets containing x.
    # Each bit i corresponds to all_t[i].
    full_mask = (1 << n_t) - 1
    point_masks = []
    for x in points:
        m = 0
        for c in all_t:
            if x in c:
                m |= (1 << index[c])
        point_masks.append(m)

    # Precompute, for each point x, the bitmask of (t-1)-subsets containing x
    # used to compute the "new coverage" of adding x to a partial block.
    # new_mask(x, partial) = OR over (t-1)-subsets S of partial of mask(S ∪ {x})
    # We precompute for each ordered pair (x, S) where S is a (t-1)-subset not
    # containing x: the bitmask of t-subsets equal to S ∪ {x} (a single bit).
    # Actually S ∪ {x} is exactly one t-subset, so it's a single bit.
    # So new_mask = sum of bits. Let's precompute triple_bit[x][S] = bit for S∪{x}.
    # Instead, precompute for each x and each (t-1)-subset S: the bit if x not in S.
    # We'll compute gain on the fly using a helper.

    # Build a fast "bit for t-subset from (t-1)-subset + point" lookup.
    # For each x, a dict from (t-1)-subset (tuple) -> bit, only for S not containing x.
    # But (t-1)-subsets of a partial block: number is C(len(block), t-1).
    # For t=3, that's C(len,2). Small.

    # Precompute: for each x, map (t-1)-subset tuple -> bit index.
    t1_subs_by_x = []
    for x in points:
        d = {}
        for c in all_t:
            if x in c:
                # c without x is a (t-1)-subset
                rest = tuple(p for p in c if p != x)
                d[rest] = index[c]
        t1_subs_by_x.append(d)

    def greedy_once(seed):
        rng = random.Random(seed)
        uncovered = full_mask
        blocks = []
        while uncovered:
            # Pick a random uncovered t-subset to anchor the block.
            # Find an uncovered bit index.
            u = uncovered
            # get lowest set bit index
            lb = u & (-u)
            idx = lb.bit_length() - 1
            anchor = all_t[idx]
            # Build block: must include anchor's points.
            block = list(anchor)
            block_set = set(anchor)
            # covered-by-block-so-far mask (for the anchor points)
            block_mask = 0
            for p in anchor:
                block_mask |= point_masks[p]
            while len(block) < k:
                # For each candidate x not in block, compute new coverage:
                #   gain_mask = OR over (t-1)-subs S of block of bit(S ∪ {x})
                #   new = gain_mask & uncovered
                #   gain = popcount(new)
                best_x = None
                best_gain = -1
                best_new = 0
                # Precompute list of (t-1)-subsets of block.
                t1_of_block = list(itertools.combinations(block, t - 1))
                # Score all candidates.
                scores = []
                for x in points:
                    if x in block_set:
                        continue
                    nm = 0
                    d = t1_subs_by_x[x]
                    for S in t1_of_block:
                        bi = d.get(S)
                        if bi is not None:
                            nm |= (1 << bi)
                    nm &= uncovered
                    g = nm.bit_count()
                    scores.append((g, nm, x))
                # Pick among top: best gain, with random tie-break among near-best.
                scores.sort(key=lambda z: z[0], reverse=True)
                top_gain = scores[0][0]
                # Choose randomly among entries with gain == top_gain (or within a small band)
                # To add diversity, allow a small band: gain >= top_gain - band
                band = 0
                cands = [z for z in scores if z[0] >= top_gain - band]
                if not cands:
                    cands = scores[:1]
                g, nm, x = rng.choice(cands)
                block.append(x)
                block_set.add(x)
                block_mask |= point_masks[x]
            # Add block.
            blocks.append(sorted(block))
            uncovered &= ~block_mask
            if remaining() <= 5.0:
                # Time pressure: stop adding blocks; we may be incomplete.
                break
        return blocks, uncovered

    def is_complete(blocks, uncovered):
        return uncovered == 0

    def reduce_blocks(blocks):
        # Remove redundant blocks: a block is redundant if its covered t-subsets
        # are all covered by the other blocks.
        n = len(blocks)
        if n == 0:
            return blocks
        block_masks = [0] * n
        for i, b in enumerate(blocks):
            m = 0
            for p in b:
                m |= point_masks[p]
            block_masks[i] = m
        # union mask
        union = 0
        for m in block_masks:
            union |= m
        kept = list(blocks)
        changed = True
        # Iterate removal passes.
        while changed:
            changed = False
            union = 0
            for m in block_masks:
                union |= m
            n = len(kept)
            for i in range(n):
                # coverage if we remove block i
                others = union & ~block_masks[i]
                if others == union:
                    # block i is redundant
                    kept.pop(i)
                    block_masks.pop(i)
                    changed = True
                    break
        return kept

    best_blocks = None
    best_uncovered = None

    # First deterministic run (seed fixed) to get a baseline quickly.
    b, u = greedy_once(12345)
    if u == 0:
        best_blocks = reduce_blocks(b)
        best_uncovered = 0
    else:
        best_blocks = b
        best_uncovered = u

    # Now randomized restarts to find smaller complete coverings.
    seed = 0
    while remaining() > 10.0:
        seed += 1
        b, u = greedy_once(seed)
        if u == 0:
            rb = reduce_blocks(b)
            if best_blocks is None or len(rb) < len(best_blocks):
                best_blocks = rb
                best_uncovered = 0
        elif best_uncovered is not None and u < best_uncovered:
            # incomplete but fewer uncovered than before; keep as fallback
            best_uncovered = u
            # don't replace best_blocks with incomplete unless we have none
            if best_blocks is None:
                best_blocks = b

    # Final: if best is incomplete, try to finish it with a deterministic greedy
    # on the remaining uncovered using the same point_masks, but that's complex.
    # Instead, if incomplete, run a plain deterministic greedy to guarantee validity.
    if best_uncovered != 0:
        # deterministic completion: greedy on remaining uncovered
        uncovered = best_uncovered
        blocks = list(best_blocks) if best_blocks else []
        while uncovered and remaining() > 2.0:
            u = uncovered
            lb = u & (-u)
            idx = lb.bit_length() - 1
            anchor = all_t[idx]
            block = list(anchor)
            block_set = set(anchor)
            block_mask = 0
            for p in anchor:
                block_mask |= point_masks[p]
            while len(block) < k:
                t1_of_block = list(itertools.combinations(block, t - 1))
                best_x = None
                best_gain = -1
                for x in points:
                    if x in block_set:
                        continue
                    nm = 0
                    d = t1_subs_by_x[x]
                    for S in t1_of_block:
                        bi = d.get(S)
                        if bi is not None:
                            nm |= (1 << bi)
                    nm &= uncovered
                    g = nm.bit_count()
                    if g > best_gain:
                        best_gain = g
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
                block_set.add(best_x)
                block_mask |= point_masks[best_x]
            blocks.append(sorted(block))
            uncovered &= ~block_mask
        best_blocks = reduce_blocks(blocks)
        best_uncovered = uncovered

    return best_blocks


# EVOLVE-BLOCK-END
