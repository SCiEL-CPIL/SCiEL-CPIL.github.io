"""Improved covering design C(19,7,4) using greedy + simulated annealing."""
import time
import random
import itertools
from collections import defaultdict

# Fixed deadline with safety margin
DEADLINE = 130.0  # seconds, leaving 20s margin under 150s limit


def _count_tsubsets_in_block(block, t):
    """Count how many t-subsets a block covers (all C(k,t) of them)."""
    k = len(block)
    # C(k, t)
    if t > k:
        return 0
    if t == 0:
        return 1
    r = 1
    for i in range(t):
        r = r * (k - i) // (i + 1)
    return r


def _build_greedy_cover(v, k, t, rng=None):
    """Standard greedy covering: repeatedly pick the block that covers the most uncovered t-subsets."""
    if rng is None:
        rng = random.Random(42)

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)

    # For each t-subset, track how many blocks cover it
    cover_count = [0] * n_tsubs
    # Map t-subset to index
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Uncovered set (use a list of indices for uncovered t-subsets)
    uncovered = set(range(n_tsubs))

    blocks = []

    # Precompute for speed: for each point x and each (t-1)-subset s, the t-subset s ∪ {x}
    # We'll build a mapping from (frozen tuple of t-1 subset, x) -> index
    # Actually, let's use a different approach for speed.

    # For greedy: at each step, try all C(v,k) blocks? That's C(19,7)=50388 which is fine.
    # But we need to compute gain quickly.

    # Alternative: for each candidate block, gain = number of uncovered t-subsets it contains.
    # We can precompute for each block the list of t-subset indices it covers.

    # Precompute: for each 7-subset (block), list of t-subset indices it covers
    # C(19,7) = 50388 blocks, each covers C(7,4)=35 t-subsets
    # Total: 50388 * 35 = 1,763,580 entries. Manageable.

    block_list = list(itertools.combinations(range(v), k))
    n_blocks = len(block_list)

    # For each block, precompute the list of t-subset indices it covers
    block_tsubs = []
    for b in block_list:
        tsubs_in_b = []
        for ts in itertools.combinations(b, t):
            tsubs_in_b.append(tsub_index[ts])
        block_tsubs.append(tsubs_in_b)

    while uncovered:
        best_gain = -1
        best_block_idx = -1
        for bi in range(n_blocks):
            gain = 0
            for ti in block_tsubs[bi]:
                if ti in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = bi
            # Early termination if we found a perfect gain
            if best_gain == len(uncovered):
                break

        if best_block_idx < 0:
            break

        b = block_list[best_block_idx]
        blocks.append(list(b))

        # Update uncovered
        for ti in block_tsubs[best_block_idx]:
            uncovered.discard(ti)
            cover_count[ti] += 1

    return blocks, cover_count, tsub_index, all_tsubs


def _simulate_annealing(v, k, t, initial_blocks, rng, time_limit):
    """Simulated annealing to reduce block count via removals and swaps."""
    start_time = time.time()
    blocks = [list(b) for b in initial_blocks]
    n_blocks = len(blocks)

    # Build coverage structure
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # For each block, precompute t-subsets it covers
    def get_block_tsubs(b):
        return [tsub_index[ts] for ts in itertools.combinations(b, t)]

    block_tsubs = [get_block_tsubs(b) for b in blocks]

    # Coverage count: how many blocks cover each t-subset
    cover_count = [0] * n_tsubs
    for bi in range(n_blocks):
        for ti in block_tsubs[bi]:
            cover_count[ti] += 1

    # Objective: number of blocks (we want to minimize)
    # We'll use simulated annealing with moves:
    # 1. Remove a block (if it doesn't break coverage)
    # 2. Swap a point in a block with another point (to enable future removals)

    best_blocks = [list(b) for b in blocks]
    best_n = n_blocks
    current_n = n_blocks

    # Temperature schedule
    T_start = 10.0
    T_end = 0.01
    cooling = 0.9995

    T = T_start
    step = 0

    while time.time() - start_time < time_limit:
        step += 1

        # Choose move type
        if current_n <= 2:
            break

        move_type = rng.random()

        if move_type < 0.4:
            # Try to remove a block
            bi = rng.randrange(current_n)
            # Check if block bi is redundant
            redundant = True
            for ti in block_tsubs[bi]:
                if cover_count[ti] <= 1:
                    redundant = False
                    break

            if redundant:
                # Remove it
                for ti in block_tsubs[bi]:
                    cover_count[ti] -= 1
                block_tsubs.pop(bi)
                blocks.pop(bi)
                current_n -= 1

                if current_n < best_n:
                    best_n = current_n
                    best_blocks = [list(b) for b in blocks]

                T *= cooling
            else:
                # Not redundant, small penalty
                T *= cooling

        elif move_type < 0.7:
            # Swap a point in a block
            if current_n < 2:
                continue
            bi = rng.randrange(current_n)
            # Pick a point to swap out
            pi = rng.randrange(k)
            old_point = blocks[bi][pi]
            # Pick a new point
            new_point = rng.choice([x for x in range(v) if x not in blocks[bi]])

            # Apply swap
            blocks[bi][pi] = new_point
            blocks[bi].sort()

            # Update coverage
            old_tsubs = get_block_tsubs(blocks[bi])
            # Remove old coverage
            for ti in block_tsubs[bi]:
                cover_count[ti] -= 1
            # Add new coverage
            for ti in old_tsubs:
                cover_count[ti] += 1
            block_tsubs[bi] = old_tsubs

            # Check validity (all t-subsets still covered)
            valid = all(cover_count[ti] >= 1 for ti in range(n_tsubs))

            if not valid:
                # Revert
                blocks[bi][pi] = old_point
                blocks[bi].sort()
                for ti in block_tsubs[bi]:
                    cover_count[ti] -= 1
                for ti in old_tsubs:
                    cover_count[ti] += 1
                block_tsubs[bi] = get_block_tsubs(blocks[bi])
            else:
                T *= cooling

        else:
            # Try to add a block (rarely, to escape local minima)
            if current_n < 200:
                # Pick a random block
                b = sorted(rng.sample(range(v), k))
                blocks.append(b)
                block_tsubs.append(get_block_tsubs(b))
                for ti in block_tsubs[-1]:
                    cover_count[ti] += 1
                current_n += 1
                T *= cooling

        T = max(T, T_end)

        # Cool down
        T *= cooling

    return best_blocks


def construct(v, k, t):
    """Construct a covering design C(v,k,t) with as few blocks as possible."""
    start_time = time.time()
    rng = random.Random(12345)

    # Phase 1: Greedy construction
    greedy_blocks, cover_count, tsub_index, all_tsubs = _build_greedy_cover(v, k, t, rng)
    n_greedy = len(greedy_blocks)

    # Phase 2: Simulated annealing to reduce
    elapsed = time.time() - start_time
    sa_time_limit = max(1.0, DEADLINE - elapsed - 5.0)  # Leave 5s margin

    sa_blocks = _simulate_annealing(v, k, t, greedy_blocks, rng, sa_time_limit)

    # Verify validity of SA result
    all_tsubs_list = list(itertools.combinations(range(v), t))
    covered = set()
    for b in sa_blocks:
        for ts in itertools.combinations(b, t):
            covered.add(ts)

    if covered == set(all_tsubs_list):
        return sa_blocks
    else:
        # Fall back to greedy if SA result is invalid
        return greedy_blocks


if __name__ == "__main__":
    # Quick test
    result = construct(19, 7, 4)
    print(f"Blocks: {len(result)}")
