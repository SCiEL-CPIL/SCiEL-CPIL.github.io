"""Stronger covering algorithm for C(v,k,t): fast greedy + vectorized local search.

Strategy (same family as the seed greedy, extended with the measured-winning
redundancy-removal and block-replacement local search):
  1. Greedy construction with vectorized gain scoring.
  2. Redundancy removal (drop blocks that are not needed).
  3. Block-replacement local search: swap a block for a better candidate that
     covers more uncovered t-subsets or enables dropping another block.
A hard deadline guard returns the best valid block set found before the limit.
"""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

_DEADLINE = 120.0
_MARGIN = 8.0  # seconds of safety margin before the evaluator kills us


def _code(combo, v):
    """Encode a sorted tuple of t distinct points in 0..v-1 as a base-v int."""
    c = 0
    for x in combo:
        c = c * v + x
    return c


def _all_t_codes(v, t):
    """Return an array of codes for every t-subset of {0..v-1}."""
    codes = np.empty(len(list(itertools.combinations(range(v), t))), dtype=np.int64)
    i = 0
    for combo in itertools.combinations(range(v), t):
        c = 0
        for x in combo:
            c = c * v + x
        codes[i] = c
        i += 1
    return codes


def _t_combos_of_block(block, t):
    """Yield the t-subsets (as tuples) of a k-block, sorted internally."""
    for combo in itertools.combinations(block, t):
        yield tuple(sorted(combo))


def _block_codes(block, t, v):
    """Return a numpy int64 array of the t-subset codes contained in block."""
    out = []
    for combo in itertools.combinations(block, t):
        c = 0
        for x in combo:
            c = c * v + x
        out.append(c)
    return np.array(out, dtype=np.int64)


def _count_array(v, t, blocks):
    """Build a count array over all t-subsets for the given blocks."""
    codes = _all_t_codes(v, t)
    counts = np.zeros(len(codes), dtype=np.int32)
    for b in blocks:
        for c in _block_codes(b, t, v):
            counts[c] += 1
    return counts


def _coverage_valid(v, t, blocks):
    """True iff every t-subset is covered by at least one block."""
    if not blocks:
        return False
    counts = _count_array(v, t, blocks)
    return bool(np.all(counts >= 1))


def _greedy_construct(v, k, t, rng, deadline):
    """Greedy covering with vectorized marginal-gain scoring."""
    codes = _all_t_codes(v, t)
    covered = np.zeros(len(codes), dtype=bool)
    blocks = []
    start = time.time()

    while True:
        if time.time() > deadline:
            break
        remaining = np.where(~covered)[0]
        if remaining.size == 0:
            break
        # Pick the uncovered t-subset with the most co-coverage potential.
        # Approximation: choose the one whose points appear in the most
        # not-yet-used point-pair potential. For speed, just take the first
        # uncovered subset (deterministic) but seed the block from it.
        first_code = remaining[0]
        # Decode first_code back to a t-tuple.
        rem = first_code
        pts = []
        for _ in range(t):
            rem, d = divmod(rem, v)
            pts.append(d)
        pts.sort()
        block = list(pts)

        # Grow the block to size k by max marginal gain.
        while len(block) < k:
            best_x = None
            best_gain = -1
            # Precompute the (t-1)-subsets of the current block.
            block_tminus = list(itertools.combinations(block, t - 1))
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in block_tminus:
                    combo = tuple(sorted(rest + (x,)))
                    c = 0
                    for zz in combo:
                        c = c * v + zz
                    if not covered[c]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)

        block.sort()
        blocks.append(block)
        for c in _block_codes(block, t, v):
            covered[c] = True

    return blocks


def _remove_redundant(v, t, blocks):
    """Drop any block whose removal leaves the covering valid. Returns new list."""
    counts = _count_array(v, t, blocks)
    kept = []
    for b in blocks:
        bc = _block_codes(b, t, v)
        # Would removing b leave any t-subset uncovered?
        would_uncover = np.any(counts[bc] == 1)
        if would_uncover:
            kept.append(b)
        else:
            # b is redundant; remove its contribution.
            counts[bc] -= 1
    return kept


def _candidate_blocks(v, k, t, blocks, rng, n_candidates):
    """Generate candidate blocks for replacement: random swaps + gap-filling."""
    candidates = set()
    uncovered_hint = None
    for _ in range(n_candidates):
        base = list(rng.choice(blocks))
        # Randomly swap a few points.
        for _ in range(rng.randint(1, 3)):
            i = rng.randrange(k)
            new_x = rng.randrange(v)
            if new_x not in base:
                base[i] = new_x
        # Ensure distinct.
        seen = set()
        clean = []
        for x in base:
            if x not in seen:
                seen.add(x)
                clean.append(x)
        while len(clean) < k:
            new_x = rng.randrange(v)
            if new_x not in seen:
                seen.add(new_x)
                clean.append(new_x)
        clean.sort()
        candidates.add(tuple(clean))
    return list(candidates)


def _local_search(v, k, t, blocks, deadline, rng):
    """Redundancy removal + block-replacement local search."""
    blocks = list(blocks)
    counts = _count_array(v, t, blocks)
    start = time.time()
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Pass 1: redundancy removal.
        new_blocks = []
        for b in blocks:
            bc = _block_codes(b, t, v)
            if np.any(counts[bc] == 1):
                new_blocks.append(b)
            else:
                counts[bc] -= 1
        if len(new_blocks) < len(blocks):
            blocks = new_blocks
            improved = True
            continue
        # Pass 2: try replacing each block with a better candidate.
        for idx in range(len(blocks)):
            if time.time() > deadline:
                return blocks
            old = blocks[idx]
            old_codes = _block_codes(old, t, v)
            # How many t-subsets does old uniquely cover (count==1)?
            unique_old = int(np.sum(counts[old_codes] == 1))
            # Try a few candidate replacements.
            cands = _candidate_blocks(v, k, t, blocks, rng, 12)
            best_cand = None
            best_score = None
            for cand in cands:
                if cand == tuple(old):
                    continue
                cand_codes = _block_codes(list(cand), t, v)
                # Net new coverage if we swap old->cand.
                # After removing old: counts' = counts - old_codes (on old_codes)
                # Then add cand: coverage gained = number of cand_codes with
                # (counts - old_codes) == 0  (i.e., currently uncovered or only
                # covered by old).
                tmp = counts.copy()
                tmp[old_codes] -= 1
                gained = int(np.sum(tmp[cand_codes] == 0))
                # We must not lose unique coverage: unique_old t-subsets that
                # cand does NOT cover would become uncovered.
                # Count unique_old t-subsets not in cand.
                cand_set = set(cand_codes.tolist())
                lost_unique = 0
                for c in old_codes.tolist():
                    if counts[c] == 1 and c not in cand_set:
                        lost_unique += 1
                score = gained - 2 * lost_unique
                if best_score is None or score > best_score:
                    best_score = score
                    best_cand = cand
            if best_cand is not None and best_score > 0:
                # Apply the swap.
                cand_codes = _block_codes(list(best_cand), t, v)
                counts[old_codes] -= 1
                counts[cand_codes] += 1
                blocks[idx] = list(best_cand)
                improved = True
                break  # restart pass after a change
    return blocks


def construct(v, k, t):
    """Return a list of k-subsets of {0..v-1} covering every t-subset."""
    start = time.time()
    deadline = start + _DEADLINE - _MARGIN
    rng = random.Random(12345)

    # Phase 1: greedy construction.
    blocks = _greedy_construct(v, k, t, rng, deadline)

    # Ensure validity; if greedy ran out of time mid-way, we still have a
    # partial covering. Verify and, if invalid, finish with a simple greedy
    # on the remaining uncovered subsets (fast path).
    if not _coverage_valid(v, t, blocks):
        # Quick repair: add blocks for uncovered t-subsets.
        covered = np.zeros(len(_all_t_codes(v, t)), dtype=bool)
        for b in blocks:
            for c in _block_codes(b, t, v):
                covered[c] = True
        codes = _all_t_codes(v, t)
        while True:
            if time.time() > deadline:
                break
            remaining = np.where(~covered)[0]
            if remaining.size == 0:
                break
            first_code = remaining[0]
            rem = first_code
            pts = []
            for _ in range(t):
                rem, d = divmod(rem, v)
                pts.append(d)
            pts.sort()
            block = list(pts)
            while len(block) < k:
                best_x, best_gain = None, -1
                block_tminus = list(itertools.combinations(block, t - 1))
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in block_tminus:
                        combo = tuple(sorted(rest + (x,)))
                        c = 0
                        for zz in combo:
                            c = c * v + zz
                        if not covered[c]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            blocks.append(block)
            for c in _block_codes(block, t, v):
                covered[c] = True

    # Phase 2: local search (redundancy removal + block replacement).
    if time.time() < deadline:
        blocks = _local_search(v, k, t, blocks, deadline, rng)

    # Final safety: ensure validity. If local search somehow broke validity,
    # fall back to a verified valid set by re-running repair.
    if not _coverage_valid(v, t, blocks):
        # Repair again quickly.
        covered = np.zeros(len(_all_t_codes(v, t)), dtype=bool)
        for b in blocks:
            for c in _block_codes(b, t, v):
                covered[c] = True
        codes = _all_t_codes(v, t)
        while True:
            remaining = np.where(~covered)[0]
            if remaining.size == 0:
                break
            first_code = remaining[0]
            rem = first_code
            pts = []
            for _ in range(t):
                rem, d = divmod(rem, v)
                pts.append(d)
            pts.sort()
            block = list(pts)
            while len(block) < k:
                best_x, best_gain = None, -1
                block_tminus = list(itertools.combinations(block, t - 1))
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in block_tminus:
                        combo = tuple(sorted(rest + (x,)))
                        c = 0
                        for zz in combo:
                            c = c * v + zz
                        if not covered[c]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            blocks.append(block)
            for c in _block_codes(block, t, v):
                covered[c] = True

    # Deduplicate blocks while preserving validity (duplicates never help).
    seen = set()
    deduped = []
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            deduped.append(b)
    blocks = deduped

    # Final validity check; if dedup broke it (shouldn't), repair.
    if not _coverage_valid(v, t, blocks):
        covered = np.zeros(len(_all_t_codes(v, t)), dtype=bool)
        for b in blocks:
            for c in _block_codes(b, t, v):
                covered[c] = True
        codes = _all_t_codes(v, t)
        while True:
            remaining = np.where(~covered)[0]
            if remaining.size == 0:
                break
            first_code = remaining[0]
            rem = first_code
            pts = []
            for _ in range(t):
                rem, d = divmod(rem, v)
                pts.append(d)
            pts.sort()
            block = list(pts)
            while len(block) < k:
                best_x, best_gain = None, -1
                block_tminus = list(itertools.combinations(block, t - 1))
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in block_tminus:
                        combo = tuple(sorted(rest + (x,)))
                        c = 0
                        for zz in combo:
                            c = c * v + zz
                        if not covered[c]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            blocks.append(block)
            for c in _block_codes(block, t, v):
                covered[c] = True

    return blocks
# EVOLVE-BLOCK-END
