"""Improved covering design C(24,7,3) solver with bitmask acceleration and post-processing."""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    all_points = list(range(v))
    all_tsubs = list(itertools.combinations(all_points, t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # For each point x, which t-subsets contain x? (as a set of indices)
    # We'll represent uncovered as a boolean numpy array of length n_tsubs
    # Precompute for each point: the list of t-subset indices that contain it
    point_to_tsubs = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubs):
        for x in ts:
            point_to_tsubs[x].append(i)

    best_blocks = None
    best_count = float('inf')

    def greedy_cover(seed):
        """Run greedy covering with a given random seed for tie-breaking."""
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=bool)
        blocks = []

        while uncovered.any():
            # Pick a random uncovered t-subset to start the block
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
            start_idx = unc_idx[rng.randint(0, len(unc_idx) - 1)]
            block = list(all_tsubs[start_idx])
            # Mark all t-subsets in this block as tentatively covered
            # (we'll finalize when block is complete)

            while len(block) < k:
                block_set = set(block)
                # Compute gain for each candidate point not in block
                # gain(x) = number of uncovered t-subsets that contain x
                #           and are "covered" by having (t-1) elements in block
                # A t-subset s is covered by block if all its elements are in block.
                # For a candidate x, the new t-subsets covered are those s where
                # x in s, (s - {x}) subset of block, and s was uncovered.
                #
                # Efficient approach: for each (t-1)-subset of block, check
                # if adding x completes a t-subset that is uncovered.
                gains = {}
                for x in all_points:
                    if x in block_set:
                        continue
                    g = 0
                    # Iterate over (t-1)-subsets of block
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_to_idx[ts]
                        if uncovered[idx]:
                            g += 1
                    gains[x] = g

                if not gains or max(gains.values()) == 0:
                    # No point adds coverage; pick random to fill
                    candidates = [x for x in all_points if x not in block_set]
                    if not candidates:
                        break
                    x = rng.choice(candidates)
                    block.append(x)
                else:
                    max_g = max(gains.values())
                    # Among those with max gain, pick randomly for diversity
                    best_candidates = [x for x, g in gains.items() if g == max_g]
                    x = rng.choice(best_candidates)
                    block.append(x)

            block.sort()
            blocks.append(block)

            # Mark all t-subsets of this block as covered
            for ts in itertools.combinations(block, t):
                uncovered[tsub_to_idx[ts]] = False

        return blocks

    def remove_redundant_blocks(blocks, seed=0):
        """Remove blocks that are redundant (all their t-subsets covered by other blocks)."""
        rng = random.Random(seed)
        blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            # Shuffle to break symmetry
            order = list(range(len(blocks)))
            rng.shuffle(order)
            for i in order:
                if i >= len(blocks):
                    continue
                # Check if block i is redundant
                # Build coverage from all blocks except i
                # For each t-subset in block i, check if covered by some other block
                redundant = True
                for ts in itertools.combinations(blocks[i], t):
                    covered = False
                    for j in range(len(blocks)):
                        if j == i:
                            continue
                        if all(x in blocks[j] for x in ts):
                            covered = True
                            break
                    if not covered:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    changed = True
                    break  # restart with new list
        return blocks

    # Try multiple seeds
    seed = 0
    while time.time() < deadline:
        blocks = greedy_cover(seed)
        if blocks and len(blocks) < best_count:
            # Try to remove redundant blocks
            blocks2 = remove_redundant_blocks(blocks, seed)
            if len(blocks2) < len(blocks):
                blocks = blocks2
            best_blocks = blocks
            best_count = len(blocks)
        seed += 1
        # Check deadline more frequently
        if time.time() > deadline:
            break

    if best_blocks is None:
        # Fallback: run one more greedy if we haven't found anything
        blocks = greedy_cover(0)
        best_blocks = blocks

    return best_blocks
