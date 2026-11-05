"""Combined program: vectorized greedy + GA-style crossover/repair + redundancy polish."""
import itertools
import random
import time
import numpy as np


def construct(v, k, t):
    start_time = time.monotonic()
    deadline = start_time + 115.0  # Safety margin under 120s

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)

    # Map each block to the list of tsub indices it covers
    block_tsubs_map = []
    for b in all_kblocks:
        tsubs = []
        for ts in itertools.combinations(b, t):
            tsubs.append(tsub_to_idx[ts])
        block_tsubs_map.append(np.array(tsubs, dtype=np.int32))

    # Map each tsub to the list of block indices that cover it
    tsub_blocks_map = [[] for _ in range(num_tsubs)]
    for b_idx, tsubs in enumerate(block_tsubs_map):
        for ts_idx in tsubs:
            tsub_blocks_map[ts_idx].append(b_idx)

    rng = random.Random(42)

    def greedy_construct(seed_tsub=None, time_limit=30.0):
        """Greedy set cover with optional random anchor."""
        counts = np.zeros(num_tsubs, dtype=np.int8)
        selected = []
        uncovered_mask = np.ones(num_tsubs, dtype=bool)
        num_uncovered = num_tsubs
        t_start = time.monotonic()

        while num_uncovered > 0:
            if time.monotonic() > start_time + deadline - t_start - 2.0 or time.monotonic() > start_time + 110.0:
                break

            if seed_tsub is not None and len(selected) == 0:
                anchor_idx = seed_tsub
            else:
                unc_idx = np.where(uncovered_mask)[0]
                if len(unc_idx) == 0:
                    break
                anchor_idx = unc_idx[rng.randrange(len(unc_idx))]

            candidates = tsub_blocks_map[anchor_idx]
            if not candidates:
                break

            best_candidate = -1
            best_gain = -1
            for b_idx in candidates:
                tsubs = block_tsubs_map[b_idx]
                gain = np.sum(uncovered_mask[tsubs])
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = b_idx

            if best_candidate == -1:
                break

            selected.append(best_candidate)
            tsubs = block_tsubs_map[best_candidate]
            counts[tsubs] += 1
            uncovered_mask[tsubs] = False
            num_uncovered -= best_gain

        return selected, counts

    def remove_redundant(block_indices, counts):
        """Remove blocks that don't uniquely cover any t-subset."""
        if not block_indices:
            return [], counts
        current = list(block_indices)
        changed = True
        while changed:
            changed = False
            to_remove = set()
            for b_idx in current:
                tsubs = block_tsubs_map[b_idx]
                if np.all(counts[tsubs] > 1):
                    to_remove.add(b_idx)
            if to_remove:
                for b_idx in to_remove:
                    counts[block_tsubs_map[b_idx]] -= 1
                current = [b for b in current if b not in to_remove]
                changed = True
        return current, counts

    def crossover(parent1, parent2, counts1, counts2):
        """Merge two coverings: pick blocks that cover most uncovered tsubs in union."""
        union_counts = np.maximum(counts1, counts2)
        uncovered = union_counts == 0
        all_blocks = list(set(parent1) | set(parent2))
        rng.shuffle(all_blocks)

        selected = []
        cur_counts = np.zeros(num_tsubs, dtype=np.int8)
        num_uncovered = np.sum(uncovered)

        for b_idx in all_blocks:
            if num_uncovered <= 0:
                break
            tsubs = block_tsubs_map[b_idx]
            gain = np.sum((cur_counts[tsubs] == 0) & uncovered)
            if gain > 0:
                selected.append(b_idx)
                cur_counts[tsubs] += 1
                num_uncovered -= gain

        # Repair: greedily add blocks for remaining uncovered
        while num_uncovered > 0:
            unc_idx = np.where((cur_counts == 0))[0]
            if len(unc_idx) == 0:
                break
            anchor = unc_idx[rng.randrange(len(unc_idx))]
            candidates = tsub_blocks_map[anchor]
            best_b = -1
            best_gain = -1
            for b_idx in candidates:
                if b_idx in selected:
                    continue
                tsubs = block_tsubs_map[b_idx]
                gain = np.sum(cur_counts[tsubs] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
            if best_b == -1:
                break
            selected.append(best_b)
            tsubs = block_tsubs_map[best_b]
            cur_counts[tsubs] += 1
            num_uncovered -= best_gain

        return selected, cur_counts

    # Phase 1: Generate diverse initial solutions
    population = []
    num_initial = 4
    for i in range(num_initial):
        if time.monotonic() > deadline - 5.0:
            break
        seed = i * 137 % num_tsubs
        blocks, counts = greedy_construct(seed_tsub=seed)
        blocks, counts = remove_redundant(blocks, counts)
        population.append((blocks, counts))

    best_blocks = None
    best_len = float('inf')
    for blocks, counts in population:
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]

    # Phase 2: Genetic improvement via crossover + repair
    if time.monotonic() < deadline - 10.0:
        gen = 0
        max_gens = 6
        while gen < max_gens and time.monotonic() < deadline - 5.0:
            gen += 1
            new_pop = []
            for _ in range(2):
                if len(population) < 2:
                    break
                p1, c1 = population[rng.randrange(len(population))]
                p2, c2 = population[rng.randrange(len(population))]
                child_blocks, child_counts = crossover(p1, p2, c1, c2)
                child_blocks, child_counts = remove_redundant(child_blocks, child_counts)
                new_pop.append((child_blocks, child_counts))

            # Keep best from new pop and old pop
            for blocks, counts in new_pop:
                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best_blocks = blocks[:]
            population.extend(new_pop)
            # Truncate to top 6
            population.sort(key=lambda x: len(x[0]))
            population = population[:6]

    # Final polish: try removing more redundant blocks
    if best_blocks is not None and time.monotonic() < deadline - 1.0:
        final_counts = np.zeros(num_tsubs, dtype=np.int8)
        for b_idx in best_blocks:
            final_counts[block_tsubs_map[b_idx]] += 1
        best_blocks, final_counts = remove_redundant(best_blocks, final_counts)

    if best_blocks is None:
        # Fallback
        blocks, counts = greedy_construct()
        best_blocks = blocks

    return [list(all_kblocks[i]) for i in best_blocks]
