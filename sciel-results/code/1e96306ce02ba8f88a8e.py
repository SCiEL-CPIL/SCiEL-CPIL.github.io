"""Stronger greedy covering with vectorized gain, randomized multi-start, and swap polish."""
import time
import random
import itertools
import numpy as np

_DEADLINE = 150.0
_SAFETY = 10.0  # seconds to keep before kill


def _remaining():
    return _DEADLINE - time.time() - _SAFETY


def construct(v, k, t):
    start = time.time()
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    # Map each t-subset to an index for fast coverage bookkeeping
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubs)}

    best_blocks = None
    best_len = None

    def time_ok():
        return time.time() - start < _DEADLINE - _SAFETY

    # ---------- Core greedy build ----------
    def greedy_build(seed_rng):
        """Build a covering using greedy with randomized tie-breaking."""
        # uncovered: boolean array over all t-subsets
        uncovered = np.ones(n_tsubs, dtype=np.uint8)
        blocks = []
        # Precompute for each point x, the list of t-subset indices that contain x
        # and for each (block_partial, x) we need count of uncovered t-subsets
        # that contain x and are subsets of block_partial U {x}.

        while np.any(uncovered) and time_ok():
            # Pick the first uncovered t-subset (deterministic anchor for stability)
            # But to diversify, we can pick randomly among uncovered
            unc_idx = np.where(uncovered > 0)[0]
            if len(unc_idx) == 0:
                break
            anchor_idx = unc_idx[seed_rng.randint(0, len(unc_idx) - 1)]
            anchor = list(all_tsubs[anchor_idx])
            block = list(anchor)
            block_set = set(block)

            # Build the block by adding points one at a time
            while len(block) < k and time_ok():
                # For each candidate x not in block, compute gain:
                # number of uncovered t-subsets that are subsets of block U {x} and contain x
                gains = np.zeros(v, dtype=np.int64)
                for x in range(v):
                    if x in block_set:
                        continue
                    # t-subsets of (block U {x}) that contain x and are currently uncovered
                    # A t-subset S is subset of block U {x} and contains x iff
                    # S \ {x} is a (t-1)-subset of block
                    # So gain(x) = count of (t-1)-subsets R of block such that R U {x} is uncovered
                    if len(block) < t - 1:
                        # Can't form a t-subset containing x with only block points
                        # Actually we need exactly t-1 from block + x. If len(block) < t-1, gain=0
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        s = tuple(sorted(rest + (x,)))
                        idx = tsub_to_idx.get(s, -1)
                        if idx >= 0 and uncovered[idx]:
                            gain += 1
                    gains[x] = gain

                # Pick x with max gain; break ties randomly
                max_gain = np.max(gains)
                candidates = [x for x in range(v) if x not in block_set and gains[x] == max_gain]
                if not candidates:
                    # Fallback: pick any unused point
                    candidates = [x for x in range(v) if x not in block_set]
                x = candidates[seed_rng.randint(0, len(candidates) - 1)]
                block.append(x)
                block_set.add(x)

            block.sort()
            blocks.append(block)

            # Update uncovered: mark all t-subsets of this block as covered
            for s in itertools.combinations(block, t):
                idx = tsub_to_idx[s]
                uncovered[idx] = 0

        # Check if complete
        if not np.any(uncovered):
            return blocks
        return None

    # ---------- Swap local search ----------
    def swap_polish(blocks, seed_rng, max_rounds=3):
        """Try 1-for-1 block swaps to reduce block count. Returns improved blocks or None."""
        if len(blocks) <= 1:
            return blocks
        # Build coverage: for each t-subset, list of block indices containing it
        # This is expensive; do it once
        tsub_blocks = [[] for _ in range(n_tsubs)]
        for bi, block in enumerate(blocks):
            for s in itertools.combinations(block, t):
                tsub_blocks[tsub_to_idx[s]].append(bi)

        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time_ok():
            improved = False
            rounds += 1
            # For each block, see if we can replace it with a better one
            # A block is "removable" if every t-subset it covers is also covered by another block
            # Try: for each block bi, check if it's redundant. If so, remove and re-add greedily.
            # Simpler: try swapping each block with a newly greedy-constructed block that covers
            # the unique t-subsets of the old block.
            for bi in range(len(blocks)):
                if not time_ok():
                    break
                # Find t-subsets uniquely covered by block bi
                unique_to_bi = []
                for s in itertools.combinations(blocks[bi], t):
                    idx = tsub_to_idx[s]
                    if len(tsub_blocks[idx]) == 1 and tsub_blocks[idx][0] == bi:
                        unique_to_bi.append(s)

                if not unique_to_bi:
                    # Block is redundant; try to remove it
                    new_blocks = [b for j, b in enumerate(blocks) if j != bi]
                    # Check validity: every t-subset still covered
                    # Quick check: for each t-subset, is it in some remaining block?
                    # This is expensive; skip full check, just verify unique ones are gone (they are, by definition unique to removed block)
                    # But other t-subsets might have been only in bi? No, unique_to_bi captures those.
                    # Actually we need to verify ALL t-subsets are covered.
                    # Do a quick validation
                    valid = True
                    covered_set = set()
                    for b in new_blocks:
                        for s in itertools.combinations(b, t):
                            covered_set.add(s)
                    # Check all t-subsets
                    for s in all_tsubs:
                        if s not in covered_set:
                            valid = False
                            break
                    if valid:
                        blocks = new_blocks
                        # Rebuild tsub_blocks
                        tsub_blocks = [[] for _ in range(n_tsubs)]
                        for bidx, block in enumerate(blocks):
                            for s in itertools.combinations(block, t):
                                tsub_blocks[tsub_to_idx[s]].append(bidx)
                        improved = True
                        break
                else:
                    # Try to replace block bi with a new block that covers all unique_to_bi
                    # Greedily build a new block covering unique_to_bi
                    new_block = set()
                    remaining_unique = list(unique_to_bi)
                    for _ in range(k):
                        if not time_ok():
                            break
                        if not remaining_unique:
                            break
                        # Pick point that covers most remaining_unique
                        best_x = None
                        best_gain = -1
                        for x in range(v):
                            if x in new_block:
                                continue
                            gain = 0
                            for s in remaining_unique:
                                if x in s:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                        if best_x is None or best_gain <= 0:
                            break
                        new_block.add(best_x)
                        # Remove covered unique t-subsets
                        remaining_unique = [s for s in remaining_unique if not (set(s) <= new_block)]

                    if len(new_block) == k and not remaining_unique:
                        new_block_list = sorted(new_block)
                        # Check if this new block is already in blocks
                        if new_block_list not in blocks:
                            # Replace
                            blocks[bi] = new_block_list
                            # Rebuild tsub_blocks
                            tsub_blocks = [[] for _ in range(n_tsubs)]
                            for bidx, block in enumerate(blocks):
                                for s in itertools.combinations(block, t):
                                    tsub_blocks[tsub_to_idx[s]].append(bidx)
                            improved = True
                            break

        return blocks

    # ---------- Main loop: multi-start ----------
    n_starts = 5
    for si in range(n_starts):
        if not time_ok():
            break
        seed = 42 + si * 1000
        rng = random.Random(seed)
        blocks = greedy_build(rng)
        if blocks is None:
            continue
        # Polish
        blocks = swap_polish(blocks, rng)
        if best_blocks is None or len(blocks) < len(best_blocks):
            best_blocks = blocks
            best_len = len(blocks)
            # If we've hit a good number, try more starts if time allows
            print(f"Start {si}: {len(blocks)} blocks", flush=True)

    # If no valid covering found yet, do a deterministic greedy as fallback
    if best_blocks is None:
        if not time_ok():
            # Emergency: return whatever we can (shouldn't happen with deterministic greedy)
            # Fall back to simple greedy
            rng = random.Random(0)
            best_blocks = greedy_build(rng)
            if best_blocks is None:
                # Last resort: exhaustive-ish greedy
                uncovered = set(all_tsubs)
                best_blocks = []
                while uncovered and time_ok():
                    block = list(min(uncovered))
                    while len(block) < k:
                        best_x = None
                        best_gain = -1
                        for x in range(v):
                            if x in block:
                                continue
                            gain = sum(1 for rest in itertools.combinations(block, t - 1)
                                       if tuple(sorted(rest + (x,))) in uncovered)
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                        if best_x is None:
                            break
                        block.append(best_x)
                    block.sort()
                    best_blocks.append(block)
                    uncovered.difference_update(set(itertools.combinations(block, t)))
        else:
            rng = random.Random(0)
            best_blocks = greedy_build(rng)
            if best_blocks is None:
                # Should not happen
                best_blocks = []

    return best_blocks


if __name__ == "__main__":
    # Quick self-test
    t0 = time.time()
    blocks = construct(19, 7, 4)
    t1 = time.time()
    print(f"Blocks: {len(blocks)}, Time: {t1 - t0:.1f}s")
    # Verify
    all_tsubs = set(itertools.combinations(range(19), 4))
    covered = set()
    for b in blocks:
        for s in itertools.combinations(b, 4):
            covered.add(s)
    print(f"Covered: {len(covered)}/{len(all_tsubs)}")
    assert covered == all_tsubs, "Not a valid covering!"
