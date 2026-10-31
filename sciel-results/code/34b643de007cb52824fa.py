import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute
    all_tsubs = list(itertools.combinations(range(v), t))
    all_kblocks = list(itertools.combinations(range(v), k))
    block_tsubs = {B: list(itertools.combinations(B, t)) for B in all_kblocks}
    # For fast subset check: precompute set of each k-block
    block_sets = {B: set(B) for B in all_kblocks}

    def greedy_once(seed, time_budget):
        rng = random.Random(seed)
        uncovered = set(all_tsubs)
        blocks = []

        while uncovered:
            if time.time() > time_budget:
                break

            pivot = rng.choice(tuple(uncovered))
            pivot_set = set(pivot)

            # Find candidates containing pivot
            candidates = [B for B in all_kblocks if pivot_set.issubset(block_sets[B])]
            if not candidates:
                break

            rng.shuffle(candidates)
            best_block = None
            best_gain = -1
            for B in candidates:
                gain = sum(1 for ts in block_tsubs[B] if ts in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = B

            if best_block is not None:
                blocks.append(best_block)
                for ts in block_tsubs[best_block]:
                    uncovered.discard(ts)

        return blocks

    def remove_redundant(blocks, time_budget):
        if not blocks:
            return blocks
        cover_count = {}
        for B in blocks:
            for ts in block_tsubs[B]:
                cover_count[ts] = cover_count.get(ts, 0) + 1

        changed = True
        while changed and time.time() < time_budget:
            changed = False
            new_blocks = []
            for B in blocks:
                redundant = True
                for ts in block_tsubs[B]:
                    if cover_count.get(ts, 0) == 1:
                        redundant = False
                        break
                if redundant:
                    for ts in block_tsubs[B]:
                        cover_count[ts] -= 1
                    changed = True
                else:
                    new_blocks.append(B)
            blocks = new_blocks
        return blocks

    def local_optimize(blocks, time_budget):
        blocks = list(blocks)
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            blocks = remove_redundant(blocks, time_budget)
            n = len(blocks)
            if n == 0:
                break

            cover_count = {}
            for B in blocks:
                for ts in block_tsubs[B]:
                    cover_count[ts] = cover_count.get(ts, 0) + 1

            # Try to replace each block with one that covers its unique t-subsets and more
            for i in range(n):
                if time.time() > time_budget:
                    break
                Bi = blocks[i]
                unique_to_i = [ts for ts in block_tsubs[Bi] if cover_count.get(ts, 0) == 1]
                if not unique_to_i:
                    for ts in block_tsubs[Bi]:
                        cover_count[ts] -= 1
                    blocks.pop(i)
                    improved = True
                    break

                Bi_set = block_sets[Bi]
                # Try candidate replacements
                for B in all_kblocks:
                    if B == Bi:
                        continue
                    B_set = block_sets[B]
                    # Must cover all unique_to_i
                    if not all(set(ts).issubset(B_set) for ts in unique_to_i):
                        continue
                    # Compute net gain
                    gain = 0
                    for ts in block_tsubs[B]:
                        if ts not in unique_to_i and cover_count.get(ts, 0) == 0:
                            gain += 1
                    if gain > 0:
                        for ts in block_tsubs[Bi]:
                            cover_count[ts] -= 1
                        for ts in block_tsubs[B]:
                            cover_count[ts] = cover_count.get(ts, 0) + 1
                        blocks[i] = B
                        improved = True
                        break
                if improved:
                    break

        return blocks

    best_blocks = None
    best_len = float('inf')

    seeds = list(range(500))
    random.shuffle(seeds)

    for idx, seed in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = deadline - time.time() - 3
        if remaining < 2.0:
            break

        # Allocate time: greedy gets up to 15s, local opt gets rest
        greedy_budget = min(remaining - 3, 15.0)
        if greedy_budget < 1.0:
            break

        blocks = greedy_once(seed, start_time + greedy_budget)
        if not blocks:
            continue

        blocks = remove_redundant(blocks, deadline - 3)

        remaining_for_opt = deadline - time.time() - 2
        if remaining_for_opt > 1.0:
            blocks = local_optimize(blocks, start_time + remaining_for_opt)
            blocks = remove_redundant(blocks, deadline - 1)

        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks

        # If good, spend more time optimizing best
        if best_len <= 145 and time.time() < deadline - 5:
            extra = deadline - time.time() - 2
            if extra > 3.0:
                best_blocks = local_optimize(best_blocks, start_time + extra)
                best_blocks = remove_redundant(best_blocks, deadline - 1)
                best_len = len(best_blocks)

    if best_blocks is None:
        blocks = greedy_once(0, deadline - 1)
        return remove_redundant(blocks, deadline - 0.5)

    return best_blocks
