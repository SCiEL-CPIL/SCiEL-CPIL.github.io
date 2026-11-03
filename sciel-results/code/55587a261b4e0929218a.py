"""Structural change: simulated annealing on block removal.
Start from a valid greedy cover, then use SA to find minimal subsets of blocks
that still cover all t-subsets. Uses numpy for fast coverage counting.
"""
import time
import itertools
import random
import numpy as np

_DEADLINE = 145.0


def _time_left():
    return _DEADLINE - time.time()


def _all_tsubsets(v, t):
    return list(itertools.combinations(range(v), t))


def _tsubsets_of_block(block, t):
    return list(itertools.combinations(block, t))


def _greedy_construct(v, k, t, rng, time_limit):
    """Fast greedy set cover construction."""
    all_points = list(range(v))
    all_tsubs = _all_tsubsets(v, t)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)

    uncovered = set(range(n_tsubs))
    blocks = []

    start = time.time()
    while uncovered and (time.time() - start) < time_limit:
        anchor = min(uncovered)
        anchor_tsub = all_tsubs[anchor]
        block = list(anchor_tsub)
        block_set = set(block)

        while len(block) < k:
            best_point = -1
            best_gain = -1
            for p in all_points:
                if p in block_set:
                    continue
                gain = 0
                if len(block) >= t - 1:
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (p,)))
                        idx = tsub_index.get(ts, -1)
                        if idx >= 0 and idx in uncovered:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = p

            if best_point == -1 or best_gain <= 0:
                for p in all_points:
                    if p not in block_set:
                        best_point = p
                        break
                if best_point == -1:
                    break

            block.append(best_point)
            block_set.add(best_point)

        block.sort()
        blocks.append(tuple(block))

        for ts in _tsubsets_of_block(block, t):
            idx = tsub_index.get(ts, -1)
            if idx >= 0:
                uncovered.discard(idx)

    if not uncovered:
        return [list(b) for b in blocks]
    return None


def _build_block_cover_indices(blocks, all_tsubs, tsub_index, n_tsubs):
    """For each block, compute the set of t-subset indices it covers."""
    block_cover = []
    for block in blocks:
        covered = set()
        for ts in _tsubsets_of_block(block, 5):  # t is passed via closure
            idx = tsub_index.get(ts, -1)
            if idx >= 0:
                covered.add(idx)
        block_cover.append(covered)
    return block_cover


def _sa_minimize(blocks, v, k, t, rng, time_limit):
    """Simulated annealing to minimize number of blocks while maintaining coverage."""
    all_tsubs = _all_tsubsets(v, t)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)

    # Build coverage: for each t-subset, how many blocks cover it
    tsub_cover_count = np.zeros(n_tsubs, dtype=np.int32)
    block_cover_sets = []
    for block in blocks:
        covered = set()
        for ts in _tsubsets_of_block(block, t):
            idx = tsub_index.get(ts, -1)
            if idx >= 0:
                covered.add(idx)
                tsub_cover_count[idx] += 1
        block_cover_sets.append(covered)

    n_blocks = len(blocks)
    current_blocks = list(blocks)
    best_blocks = list(blocks)
    best_n = n_blocks

    start_time = time.time()
    temp = 1.0
    cooling = 0.9995
    min_temp = 0.01

    while temp > min_temp and (time.time() - start_time) < time_limit:
        # Try removing a random block
        if n_blocks <= 1:
            break

        bi = rng.randint(0, n_blocks - 1)

        # Check if removing block bi would leave any t-subset uncovered
        # A t-subset is uncovered if its cover count drops to 0
        can_remove = True
        for ts_idx in block_cover_sets[bi]:
            if tsub_cover_count[ts_idx] == 1:
                can_remove = False
                break

        if can_remove:
            # Remove the block
            for ts_idx in block_cover_sets[bi]:
                tsub_cover_count[ts_idx] -= 1
            del block_cover_sets[bi]
            del current_blocks[bi]
            n_blocks -= 1

            if n_blocks < best_n:
                best_n = n_blocks
                best_blocks = list(current_blocks)
        else:
            # Can't remove; maybe swap with another block
            # Try to find a block j such that swapping helps
            # For now, just accept the failed removal (no state change)
            pass

        # Occasionally try a more aggressive move: remove a block and add a new one
        # that covers the gaps
        if rng.random() < 0.1 and n_blocks > 10:
            # Pick a random block to remove
            bi = rng.randint(0, n_blocks - 1)
            removed_cover = block_cover_sets[bi]
            
            # Find t-subsets that would become uncovered
            would_uncover = set()
            for ts_idx in removed_cover:
                if tsub_cover_count[ts_idx] == 1:
                    would_uncover.add(ts_idx)

            if not would_uncover:
                # Safe to remove
                for ts_idx in removed_cover:
                    tsub_cover_count[ts_idx] -= 1
                del block_cover_sets[bi]
                del current_blocks[bi]
                n_blocks -= 1
            else:
                # Need to add a block that covers would_uncover
                # Build a new block greedily from would_uncover
                if would_uncover:
                    anchor_idx = rng.choice(list(would_uncover))
                    anchor_tsub = all_tsubs[anchor_idx]
                    new_block = list(anchor_tsub)
                    new_block_set = set(new_block)
                    
                    remaining_uncover = set(would_uncover)
                    while len(new_block) < k and remaining_uncover:
                        best_point = -1
                        best_gain = -1
                        for p in range(v):
                            if p in new_block_set:
                                continue
                            gain = 0
                            if len(new_block) >= t - 1:
                                for rest in itertools.combinations(new_block, t - 1):
                                    ts = tuple(sorted(rest + (p,)))
                                    idx = tsub_index.get(ts, -1)
                                    if idx >= 0 and idx in remaining_uncover:
                                        gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_point = p
                        if best_point == -1:
                            for p in range(v):
                                if p not in new_block_set:
                                    best_point = p
                                    break
                            if best_point == -1:
                                break
                        new_block.append(best_point)
                        new_block_set.add(best_point)
                    
                    new_block.sort()
                    # Check if this new block is different from existing
                    new_block_tuple = tuple(new_block)
                    if new_block_tuple not in [tuple(b) for b in current_blocks]:
                        # Add the new block
                        new_covered = set()
                        for ts in _tsubsets_of_block(new_block, t):
                            idx = tsub_index.get(ts, -1)
                            if idx >= 0:
                                new_covered.add(idx)
                                tsub_cover_count[idx] += 1
                        # Remove the old block
                        for ts_idx in removed_cover:
                            tsub_cover_count[ts_idx] -= 1
                        del block_cover_sets[bi]
                        del current_blocks[bi]
                        # Add new block
                        current_blocks.append(new_block)
                        block_cover_sets.append(new_covered)
                        # n_blocks stays the same (removed 1, added 1)

        temp *= cooling

    return best_blocks


def _is_valid_cover(blocks, v, t):
    covered = set()
    for block in blocks:
        for ts in _tsubsets_of_block(block, t):
            covered.add(ts)
    all_tsubs = set(_all_tsubsets(v, t))
    return all_tsubs <= covered


def construct(v, k, t):
    start_time = time.time()
    rng = random.Random(42)

    # Phase 1: Greedy construction
    greedy_time = min(20, _time_left() * 0.2)
    blocks = _greedy_construct(v, k, t, rng, greedy_time)

    if blocks is None:
        # Fallback
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    for x in range(v):
                        if x not in block:
                            best_x = x
                            break
                block.append(best_x)
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))

    # Verify validity
    if not _is_valid_cover(blocks, v, t):
        uncovered = set(itertools.combinations(range(v), t))
        for ts in sorted(uncovered):
            blocks.append(list(ts) + [p for p in range(v) if p not in ts][:k - t])
        seen = set()
        unique_blocks = []
        for b in blocks:
            bt = tuple(sorted(b))
            if bt not in seen:
                seen.add(bt)
                unique_blocks.append(list(bt))
        blocks = unique_blocks

    # Phase 2: Simulated annealing minimization
    remaining = _time_left()
    if remaining > 5:
        sa_time = min(remaining - 5, 120)
        blocks = _sa_minimize(blocks, v, k, t, rng, sa_time)

    # Final validation
    if not _is_valid_cover(blocks, v, t):
        uncovered = set(itertools.combinations(range(v), t))
        for ts in sorted(uncovered):
            block = list(ts)
            for p in range(v):
                if p not in block and len(block) < k:
                    block.append(p)
            block.sort()
            if tuple(block) not in set(tuple(sorted(b)) for b in blocks):
                blocks.append(block)

    # Deduplicate
    seen = set()
    unique_blocks = []
    for b in blocks:
        bt = tuple(sorted(b))
        if bt not in seen:
            seen.add(bt)
            unique_blocks.append(list(bt))

    return unique_blocks
