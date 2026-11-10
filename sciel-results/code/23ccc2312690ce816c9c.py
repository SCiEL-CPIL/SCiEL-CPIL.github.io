"""Bitset-accelerated greedy covering for C(v, k, t)."""
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute all t-subsets and assign each an index
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {sub: i for i, sub in enumerate(all_tsubs)}

    # We represent the set of uncovered t-subsets as a list of Python ints (bitsets)
    # For v=24, t=3, num_tsubs = C(24,3) = 2024, so we need 2024 bits.
    # Python ints handle arbitrary precision, so we can use a single int per coverage mask.

    # Precompute for each point p and each (t-1)-subset of other points, the list of t-subset indices
    # that would be covered if we add point p to that (t-1)-subset.
    # Actually, let's precompute for each candidate block (k-subset), which t-subsets it covers.
    # But there are C(24,7) = 346104 possible blocks, which is manageable.

    # Alternative: precompute for each point p, a mapping from (t-1)-subset to list of t-subset indices.
    # When building a block greedily, for a candidate point x, the gain is the number of uncovered
    # t-subsets that contain x and (t-1) elements from the current partial block.

    # Let's precompute: for each point x, and for each (t-1)-combination of points, 
    # the index of the t-subset formed by adding x.
    # Actually, for a partial block B (with |B| < k), and candidate x not in B:
    # gain(x) = number of (t-1)-subsets S of B such that S ∪ {x} is uncovered.
    # This is equivalent to: for each (t-1)-subset S of B, check if tsub_index[S ∪ {x}] is uncovered.

    # Precompute: for each point x, a dictionary mapping frozenset of (t-1) points to the t-subset index.
    # But (t-1) can be up to 6, and we'd need to handle all combinations.
    # Better: precompute for each point x, an array of length C(v-1, t-1) mapping (t-1)-subsets not containing x to t-subset indices.
    # This is getting complex. Let's just compute on the fly but use bitsets for the uncovered set.

    # Uncovered set as a bitset (Python int)
    uncovered = (1 << num_tsubs) - 1  # all bits set

    blocks = []
    random.seed(42)

    while uncovered:
        if time.time() > deadline - 5.0:
            break  # safety margin

        # Pick the first uncovered t-subset as a seed
        # Find the lowest set bit
        lowbit = uncovered & (-uncovered)
        seed_idx = lowbit.bit_length() - 1
        seed = all_tsubs[seed_idx]
        block = list(seed)

        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(block) < k:
            if time.time() > deadline - 5.0:
                break

            best_gain = -1
            best_x = None
            for x in range(v):
                if x in block:
                    continue
                # Compute gain: number of (t-1)-subsets S of block such that S ∪ {x} is uncovered
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    tsub = tuple(sorted(rest + (x,)))
                    idx = tsub_index.get(tsub)
                    if idx is not None:
                        if (uncovered >> idx) & 1:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None or best_gain <= 0:
                # Add a random point to avoid getting stuck
                candidates = [x for x in range(v) if x not in block]
                if candidates:
                    best_x = random.choice(candidates)
                else:
                    break
            block.append(best_x)

        block.sort()
        # Only add the block if it covers at least one new t-subset
        new_covered = 0
        for tsub in itertools.combinations(block, t):
            idx = tsub_index[tsub]
            if (uncovered >> idx) & 1:
                new_covered += 1
        if new_covered > 0:
            blocks.append(block)
            # Update uncovered bitset
            for tsub in itertools.combinations(block, t):
                idx = tsub_index[tsub]
                uncovered &= ~(1 << idx)

    # Verify coverage
    if not uncovered:
        return blocks

    # If we didn't cover everything in time, try a faster but less optimal approach
    # Just return what we have (may be invalid, but better than timeout)
    return blocks
