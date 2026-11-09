"""Covering design C(19,7,4) via fast bitmask greedy + local search."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin

    # Precompute all t-subsets and index
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    total_tsubs = len(all_tsubs)
    full_mask = (1 << total_tsubs) - 1

    # Precompute block masks: for each k-subset, the bitmask of t-subsets it covers
    block_mask = {}
    # Also precompute tsub_to_blocks: for each t-subset index, list of blocks containing it
    tsub_to_blocks = [[] for _ in range(total_tsubs)]

    for block in itertools.combinations(range(v), k):
        mask = 0
        for ts in itertools.combinations(block, t):
            mask |= (1 << tsub_index[ts])
        block_mask[block] = mask
        for ts in itertools.combinations(block, t):
            tsub_to_blocks[tsub_index[ts]].append(block)

    def coverage_mask(blks):
        m = 0
        for b in blks:
            m |= block_mask[b]
        return m

    def is_valid(blocks):
        return coverage_mask(blocks) == full_mask

    def greedy_cover(rng, deadline, target_subset=None):
        uncovered = full_mask
        blocks = []
        while uncovered and time.time() < deadline:
            # Pick a t-subset to cover
            if target_subset is not None:
                ts_idx = tsub_index.get(target_subset)
                if ts_idx is not None and (uncovered >> ts_idx) & 1:
                    chosen = ts_idx
                else:
                    # Random uncovered
                    bits = [i for i in range(total_tsubs) if (uncovered >> i) & 1]
                    if not bits:
                        break
                    chosen = rng.choice(bits)
            else:
                bits = [i for i in range(total_tsubs) if (uncovered >> i) & 1]
                if not bits:
                    break
                chosen = rng.choice(bits)

            # Among blocks covering this t-subset, pick max coverage of uncovered
            candidates = tsub_to_blocks[chosen]
            rng.shuffle(candidates)
            best_block = None
            best_gain = -1
            for b in candidates:
                gain = (block_mask[b] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
                elif gain == best_gain and best_block is not None and rng.random() < 0.3:
                    best_block = b

            if best_block is None:
                break

            blocks.append(best_block)
            uncovered &= ~block_mask[best_block]

        return blocks

    def local_search(blocks, deadline):
        blocks = list(blocks)
        # Phase 1: Remove redundant
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline:
                    break
                without = coverage_mask(blocks[:i] + blocks[i+1:])
                if (without | block_mask[blocks[i]]) == without:
                    blocks.pop(i)
                    changed = True
                    break

        # Phase 2: Try to merge two blocks into one
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(blocks)
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i+1, n):
                    if time.time() > deadline:
                        break
                    needed = block_mask[blocks[i]] | block_mask[blocks[j]]
                    # Try to find a single block covering all of 'needed'
                    found = False
                    for _ in range(30):
                        b = tuple(sorted(random.sample(range(v), k)))
                        if block_mask.get(b, 0) & needed == needed:
                            new_blocks = blocks[:i] + [b] + blocks[i+1:j] + blocks[j+1:]
                            if coverage_mask(new_blocks) == full_mask:
                                blocks = new_blocks
                                improved = True
                                found = True
                                break
                    if found:
                        break
                if improved:
                    break

        # Phase 3: Random swaps
        for _ in range(150):
            if time.time() > deadline:
                break
            if len(blocks) < 2:
                break
            i = random.randrange(len(blocks))
            new_b = tuple(sorted(random.sample(range(v), k)))
            if new_b in blocks:
                continue
            test = blocks[:i] + [new_b] + blocks[i+1:]
            if coverage_mask(test) == full_mask:
                blocks = test
                # Remove redundant
                changed = True
                while changed and time.time() < deadline:
                    changed = False
                    for j in range(len(blocks)):
                        if time.time() > deadline:
                            break
                        without = coverage_mask(blocks[:j] + blocks[j+1:])
                        if (without | block_mask[blocks[j]]) == without:
                            blocks.pop(j)
                            changed = True
                            break
        return blocks

    rng = random.Random(42)
    best_blocks = None
    best_count = float('inf')

    # Phase 1: Multiple greedy runs
    while time.time() < deadline - 30:
        target = rng.choice(all_tsubs) if rng.random() < 0.3 else None
        blocks = greedy_cover(rng, deadline, target_subset=target)
        if not blocks:
            continue
        if not is_valid(blocks):
            # Fix uncovered
            covered = coverage_mask(blocks)
            uncovered = full_mask & ~covered
            while uncovered and time.time() < deadline:
                bits = [i for i in range(total_tsubs) if (uncovered >> i) & 1]
                if not bits:
                    break
                ts_idx = rng.choice(bits)
                ts = all_tsubs[ts_idx]
                candidates = [b for b in block_mask if all(x in b for x in ts)]
                if candidates:
                    b = rng.choice(candidates)
                    blocks.append(b)
                    uncovered &= ~block_mask[b]
                else:
                    break
        count = len(blocks)
        if count < best_count:
            best_count = count
            best_blocks = list(blocks)

    if best_blocks is None:
        blocks = greedy_cover(rng, deadline)
        return [list(b) for b in blocks]

    # Phase 2: Local search
    if time.time() < deadline - 5:
        improved = local_search(best_blocks, deadline)
        if len(improved) < best_count and is_valid(improved):
            best_blocks = improved
            best_count = len(improved)

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
