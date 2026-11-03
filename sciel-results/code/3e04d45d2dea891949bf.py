"""Covering design C(19,7,4) with randomized greedy + redundancy removal."""
import itertools
import random
import time
import numpy as np


def construct(v, k, t):
    deadline = time.time() + 130.0  # safety margin under 150s limit

    # Precompute all t-subsets and map to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Bitset for coverage: bit i set means t-subset i is covered
    covered = np.zeros(n_tsubs, dtype=bool)

    # Precompute for each k-subset which t-subsets it covers
    # We'll do this lazily / on the fly for speed

    def tsubs_in_block(block):
        """Return indices of t-subsets covered by a block (list of k points)."""
        idxs = []
        for ts in itertools.combinations(block, t):
            idxs.append(tsub_to_idx[ts])
        return idxs

    # Randomized greedy: repeatedly pick an uncovered t-subset, then build a block
    # around it by greedily adding points that maximize new coverage.
    # After building, remove redundant blocks.

    best_blocks = None
    best_count = n_tsubs * k  # worst case upper bound

    # Multiple restarts with different random seeds
    rng = random.Random(42)

    for restart in range(200):
        if time.time() > deadline - 10.0:
            break

        covered = np.zeros(n_tsubs, dtype=bool)
        blocks = []
        uncovered_set = set(range(n_tsubs))

        while uncovered_set:
            if time.time() > deadline - 5.0:
                break

            # Pick a random uncovered t-subset (or the one with most potential)
            # Strategy: pick randomly to diversify
            target_idx = rng.choice(sorted(uncovered_set))
            target = all_tsubs[target_idx]

            # Build a block starting with the target t-subset
            block = list(target)
            rng.shuffle(block)
            block = sorted(block)

            # Greedily add points to maximize new coverage
            while len(block) < k:
                if time.time() > deadline - 3.0:
                    break

                best_gain = -1
                best_point = None
                for x in range(v):
                    if x in block:
                        continue
                    # Count how many uncovered t-subsets would be covered
                    # by adding x to the current block
                    gain = 0
                    for ts in itertools.combinations(block, t - 1):
                        new_ts = tuple(sorted(ts + (x,)))
                        if new_ts in tsub_to_idx:
                            idx = tsub_to_idx[new_ts]
                            if not covered[idx]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    elif gain == best_gain and best_point is not None:
                        # Tie-break: random
                        if rng.random() < 0.5:
                            best_point = x

                if best_point is None:
                    break
                block.append(best_point)

            block.sort()

            # Check if this block adds any new coverage
            new_idxs = tsubs_in_block(block)
            new_count = sum(1 for i in new_idxs if not covered[i])
            if new_count == 0:
                continue

            # Mark covered
            for i in new_idxs:
                covered[i] = True
            blocks.append(block)
            for i in new_idxs:
                if covered[i]:
                    uncovered_set.discard(i)

        # Post-hoc redundancy removal
        if blocks and time.time() < deadline - 5.0:
            # Remove redundant blocks: a block is redundant if all its t-subsets
            # are still covered without it
            n_blocks = len(blocks)
            # For each block, compute which t-subsets it uniquely covers
            # (i.e., t-subsets covered by this block and no other block)
            # Build a coverage count
            tsub_cover_count = np.zeros(n_tsubs, dtype=int)
            for blk in blocks:
                for i in tsubs_in_block(blk):
                    tsub_cover_count[i] += 1

            # A block is redundant if for every t-subset it covers, count > 1
            # But we need to be careful: removing one block may make another redundant.
            # Do iterative removal.
            changed = True
            while changed:
                changed = False
                for bi in range(len(blocks) - 1, -1, -1):
                    if time.time() > deadline - 3.0:
                        break
                    blk = blocks[bi]
                    redundant = True
                    for i in tsubs_in_block(blk):
                        if tsub_cover_count[i] <= 1:
                            redundant = False
                            break
                    if redundant:
                        # Remove this block
                        for i in tsubs_in_block(blk):
                            tsub_cover_count[i] -= 1
                        blocks.pop(bi)
                        changed = True

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]

        # If we've hit the theoretical lower bound or best known, stop early
        if best_count <= 151:
            break

    if best_blocks is None:
        # Fallback: return empty (shouldn't happen)
        return []

    return best_blocks
