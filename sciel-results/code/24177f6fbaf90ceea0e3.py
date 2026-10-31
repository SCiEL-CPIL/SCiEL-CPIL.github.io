"""Improved greedy covering for C(19,7,4) with local optimization and time guard."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from math import comb, log
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 120s

    all_points = list(range(v))
    all_tsubs = list(itertools.combinations(all_points, t))
    n_tsubs = len(all_tsubs)

    # Map each t-subset to an index for fast lookup
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    def block_tsubs(block):
        """Yield indices of t-subsets contained in block."""
        for c in itertools.combinations(block, t):
            yield tsub_to_idx[c]

    # --- Greedy construction with randomization and multiple restarts ---
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=bool)
        blocks = []

        while True:
            if time.time() > deadline - 2.0:
                break
            if not uncovered.any():
                break

            # Pick a random uncovered t-subset to start
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
            start_idx = unc_idx[rng.randint(0, len(unc_idx) - 1)]
            block = list(all_tsubs[start_idx])
            covered_by_block = set(block_tsubs(block))
            uncovered[list(covered_by_block)] = False

            # Greedily add points
            while len(block) < k:
                if time.time() > deadline - 2.0:
                    break
                best_gain = -1
                best_x = -1
                for x in all_points:
                    if x in block:
                        continue
                    # Count how many currently-uncovered t-subsets would be covered
                    gain = 0
                    for c in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(c + (x,)))
                        idx = tsub_to_idx.get(ts)
                        if idx is not None and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x != -1:
                        # Tie-break: prefer point that appears in more uncovered t-subsets overall
                        pass
                if best_x == -1 or best_gain <= 0:
                    # No improvement possible; pick random remaining point
                    remaining = [x for x in all_points if x not in block]
                    if not remaining:
                        break
                    best_x = rng.choice(remaining)
                block.append(best_x)

                # Update uncovered
                for c in itertools.combinations(block, t):
                    idx = tsub_to_idx[c]
                    uncovered[idx] = False

            block.sort()
            blocks.append(tuple(block))

            # Check deadline
            if time.time() > deadline - 2.0:
                break

        return blocks, uncovered

    # --- Local optimization: try swapping points in blocks ---
    def local_optimize(blocks, uncovered, max_rounds=50):
        """Try to reduce block count by removing redundant blocks or swapping points."""
        if time.time() > deadline - 3.0:
            return blocks, uncovered

        # First pass: remove blocks that don't cover any unique t-subset
        # Build coverage count for each t-subset
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for block in blocks:
            for idx in block_tsubs(block):
                cover_count[idx] += 1

        # A block is redundant if all its t-subsets are covered by other blocks
        new_blocks = []
        for i, block in enumerate(blocks):
            is_redundant = True
            for idx in block_tsubs(block):
                if cover_count[idx] == 1:
                    is_redundant = False
                    break
            if not is_redundant:
                new_blocks.append(block)

        if len(new_blocks) < len(blocks):
            blocks = new_blocks
            # Recompute uncovered
            covered = np.zeros(n_tsubs, dtype=bool)
            for block in blocks:
                for idx in block_tsubs(block):
                    covered[idx] = True
            uncovered = ~covered

        # Second pass: try swapping one point in a block with another to improve coverage
        for _ in range(max_rounds):
            if time.time() > deadline - 3.0:
                break
            improved = False
            for bi in range(len(blocks)):
                if time.time() > deadline - 3.0:
                    break
                block = list(blocks[bi])
                for pi in range(len(block)):
                    for x in all_points:
                        if x in block:
                            continue
                        # Try replacing block[pi] with x
                        new_block = block[:pi] + [x] + block[pi+1:]
                        new_block.sort()
                        # Count new t-subsets covered
                        gain = 0
                        for c in itertools.combinations(new_block, t):
                            idx = tsub_to_idx[c]
                            if not covered[idx]:
                                gain += 1
                        # Count lost t-subsets
                        loss = 0
                        for c in itertools.combinations(block, t):
                            idx = tsub_to_idx[c]
                            if not covered[idx]:
                                loss += 1
                        if gain > loss:
                            # Accept swap
                            blocks[bi] = tuple(new_block)
                            for c in itertools.combinations(new_block, t):
                                covered[tsub_to_idx[c]] = True
                            for c in itertools.combinations(block, t):
                                idx = tsub_to_idx[c]
                                if not any(c2 == c for c2 in itertools.combinations(new_block, t)):
                                    # This t-subset is no longer covered by this block
                                    # Check if covered by other blocks
                                    still_covered = False
                                    for bj in range(len(blocks)):
                                        if bj == bi:
                                            continue
                                        for c2 in itertools.combinations(list(blocks[bj]), t):
                                            if c2 == c:
                                                still_covered = True
                                                break
                                        if still_covered:
                                            break
                                    if not still_covered:
                                        covered[idx] = False
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break

        # Recompute uncovered
        covered = np.zeros(n_tsubs, dtype=bool)
        for block in blocks:
            for idx in block_tsubs(block):
                covered[idx] = True
        uncovered = ~covered

        return blocks, uncovered

    # --- Multi-restart greedy with local optimization ---
    best_blocks = None
    best_uncovered = None
    best_count = float('inf')

    n_restarts = 8
    for seed in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        blocks, uncovered = greedy_cover(seed)
        n_uncov = int(uncovered.sum())

        if n_uncov < best_count:
            best_count = n_uncov
            best_blocks = blocks
            best_uncovered = uncovered

        # If fully covered, try to optimize
        if n_uncov == 0:
            blocks, uncovered = local_optimize(blocks, uncovered, max_rounds=30)
            n_uncov = int(uncovered.sum())
            if n_uncov < best_count:
                best_count = n_uncov
                best_blocks = blocks
                best_uncovered = uncovered

        if best_count == 0:
            break

    # If we still have uncovered t-subsets, finish with pure greedy
    if best_blocks is None or int(best_uncovered.sum()) > 0:
        # Finish the best partial solution
        if best_blocks is None:
            blocks, uncovered = greedy_cover(0)
        else:
            blocks = list(best_blocks)
            uncovered = best_uncovered.copy()

        # Continue greedy to cover remaining
        while int(uncovered.sum()) > 0:
            if time.time() > deadline - 1.0:
                break
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
            start_idx = unc_idx[0]
            block = list(all_tsubs[start_idx])
            for idx in block_tsubs(block):
                uncovered[idx] = False

            while len(block) < k:
                if time.time() > deadline - 1.0:
                    break
                best_gain = -1
                best_x = -1
                for x in all_points:
                    if x in block:
                        continue
                    gain = 0
                    for c in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(c + (x,)))
                        idx = tsub_to_idx.get(ts)
                        if idx is not None and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    remaining = [x for x in all_points if x not in block]
                    if not remaining:
                        break
                    best_x = remaining[0]
                block.append(best_x)
                for c in itertools.combinations(block, t):
                    uncovered[tsub_to_idx[c]] = False

            block.sort()
            blocks.append(tuple(block))

        best_blocks = blocks

    # Final safety check: ensure all t-subsets are covered
    # If not, add blocks until covered (shouldn't happen if greedy completed)
    if best_blocks is not None:
        covered = np.zeros(n_tsubs, dtype=bool)
        for block in best_blocks:
            for idx in block_tsubs(block):
                covered[idx] = True
        if not covered.all():
            uncovered = ~covered
            while int(uncovered.sum()) > 0:
                if time.time() > deadline - 0.5:
                    break
                unc_idx = np.where(uncovered)[0]
                if len(unc_idx) == 0:
                    break
                start_idx = unc_idx[0]
                block = list(all_tsubs[start_idx])
                for idx in block_tsubs(block):
                    uncovered[idx] = False
                while len(block) < k:
                    remaining = [x for x in all_points if x not in block]
                    if not remaining:
                        break
                    best_gain = -1
                    best_x = remaining[0]
                    for x in remaining:
                        gain = 0
                        for c in itertools.combinations(block, t - 1):
                            ts = tuple(sorted(c + (x,)))
                            idx = tsub_to_idx.get(ts)
                            if idx is not None and uncovered[idx]:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    block.append(best_x)
                    for c in itertools.combinations(block, t):
                        uncovered[tsub_to_idx[c]] = False
                block.sort()
                best_blocks.append(tuple(block))

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
