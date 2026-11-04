"""Fresh design for C(24,7,3) covering: fast greedy with multi-start and local improvement."""
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s

    # Precompute all t-subsets and assign IDs
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_id = {ts: i for i, ts in enumerate(all_tsubs)}

    # For each point, precompute which t-subsets it participates in
    point_tsubs = defaultdict(set)
    for ts in all_tsubs:
        for p in ts:
            point_tsubs[p].add(tsub_to_id[ts])

    def greedy_cover(seed_blocks=None, randomize=False, rng=None):
        """Greedy covering. Returns list of blocks."""
        if rng is None:
            rng = random.Random()
        # uncovered: set of t-subset IDs not yet covered
        uncovered = set(range(n_tsubs))
        blocks = []

        if seed_blocks:
            for blk in seed_blocks:
                blocks.append(blk)
                for ts in itertools.combinations(blk, t):
                    uncovered.discard(tsub_to_id[ts])

        while uncovered:
            if time.time() > deadline:
                break

            if randomize:
                # Pick a random uncovered t-subset to start
                start_tsub = rng.choice(list(uncovered))
                start_points = list(all_tsubs[start_tsub])
                rng.shuffle(start_points)
                block = start_points[:1]
            else:
                # Pick the lexicographically smallest uncovered t-subset
                min_id = min(uncovered)
                block = list(all_tsubs[min_id])[:1]

            while len(block) < k:
                # Find the point that maximizes coverage of uncovered t-subsets
                best_point = -1
                best_gain = -1
                block_set = set(block)

                # Candidate points: those not in block
                # We only need to consider points that appear in some uncovered t-subset with the current block
                # For efficiency, compute gain for each candidate
                for x in range(v):
                    if x in block_set:
                        continue
                    # Gain = number of uncovered t-subsets that contain x and all (t-1) subsets of block
                    # A t-subset containing x and (t-1) elements from block
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        tid = tsub_to_id[ts]
                        if tid in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        if gain == 0:
                            # No point helps; break early (shouldn't happen if uncovered is non-empty)
                            pass

                if best_point == -1:
                    # Fallback: pick any point not in block
                    for x in range(v):
                        if x not in block_set:
                            best_point = x
                            break

                block.append(best_point)

            block.sort()
            blocks.append(block)

            # Update uncovered
            for ts in itertools.combinations(block, t):
                uncovered.discard(tsub_to_id[ts])

        return blocks

    # Try deterministic greedy first
    best_blocks = greedy_cover(randomize=False)
    best_len = len(best_blocks)

    # Multi-start with randomization
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Randomly perturb: start with a random t-subset
        blocks = greedy_cover(randomize=True, rng=rng)
        if len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
            # If we got a significant improvement, try more perturbations around it
            rng.seed(rng.randint(0, 2**31))

        # Also try a variant: after greedy, do a local search to remove redundant blocks
        if len(blocks) == best_len and attempts % 5 == 0:
            # Try to remove a block if it's redundant
            for i in range(len(blocks)):
                if time.time() > deadline:
                    break
                candidate = blocks[:i] + blocks[i+1:]
                # Check if all t-subsets are still covered
                covered = set()
                for blk in candidate:
                    for ts in itertools.combinations(blk, t):
                        covered.add(ts)
                if len(covered) == n_tsubs:
                    best_blocks = candidate
                    best_len = len(best_blocks)
                    break

    # Final check: ensure we have a valid covering
    # If we broke out early, verify coverage
    covered = set()
    for blk in best_blocks:
        for ts in itertools.combinations(blk, t):
            covered.add(ts)
    if len(covered) < n_tsubs:
        # Complete the covering with additional greedy blocks
        uncovered = set(all_tsubs) - covered
        while uncovered and time.time() < deadline:
            min_id = min(tsub_to_id[ts] for ts in uncovered)
            block = list(all_tsubs[min_id])
            while len(block) < k:
                best_point = -1
                best_gain = -1
                block_set = set(block)
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        tid = tsub_to_id[ts]
                        if tid in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    for x in range(v):
                        if x not in block_set:
                            best_point = x
                            break
                block.append(best_point)
            block.sort()
            best_blocks.append(block)
            for ts in itertools.combinations(block, t):
                uncovered.discard(ts)

    return best_blocks
