import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_t_subsets)}
    num_triples = len(all_t_subsets)

    def is_valid_cover(blocks):
        if not blocks:
            return False
        covered = np.zeros(num_triples, dtype=np.uint8)
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[tuple(triple)]
                covered[idx] = 1
        return np.all(covered == 1)

    def block_triple_indices(block):
        indices = []
        for triple in itertools.combinations(block, t):
            indices.append(triple_to_idx[tuple(triple)])
        return indices

    def compute_gains(block, uncovered, candidates):
        gains = np.zeros(len(candidates), dtype=np.int32)
        if t == 3:
            block_pairs = list(itertools.combinations(block, 2))
            for x_i, x in enumerate(candidates):
                gain = 0
                for a, b in block_pairs:
                    triple = tuple(sorted((a, b, x)))
                    idx = triple_to_idx[triple]
                    gain += uncovered[idx]
                gains[x_i] = gain
        else:
            block_subsets = list(itertools.combinations(block, t - 1))
            for x_i, x in enumerate(candidates):
                gain = 0
                for subset in block_subsets:
                    triple = tuple(sorted(subset + (x,)))
                    idx = triple_to_idx[triple]
                    gain += uncovered[idx]
                gains[x_i] = gain
        return gains

    def random_greedy_cover(rng):
        uncovered = np.ones(num_triples, dtype=np.uint8)
        blocks = []
        while True:
            if time.time() > deadline - 10.0:
                return None
            if not np.any(uncovered):
                break
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            seed_idx = rng.choice(uncovered_indices.tolist())
            seed_triple = all_t_subsets[seed_idx]
            block = list(seed_triple)
            block_set = set(block)
            while len(block) < k:
                if time.time() > deadline - 10.0:
                    return None
                candidates = [x for x in all_points if x not in block_set]
                if not candidates:
                    break
                gains = compute_gains(block, uncovered, candidates)
                max_gain = int(np.max(gains))
                if max_gain <= 0:
                    block.append(rng.choice(candidates))
                else:
                    max_gain_val = np.max(gains)
                    best_candidates = [c for c, g in zip(candidates, gains) if g == max_gain_val]
                    block.append(rng.choice(best_candidates))
                block_set.add(block[-1])
            if len(block) < k:
                return None
            block.sort()
            blocks.append(block)
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[tuple(triple)]
                uncovered[idx] = 0
        return blocks

    def remove_redundant_blocks(blocks):
        if not blocks:
            return blocks
        triple_count = np.zeros(num_triples, dtype=np.int32)
        block_indices = []
        for block in blocks:
            indices = block_triple_indices(block)
            block_indices.append(indices)
            triple_count[indices] += 1
        changed = True
        while changed:
            changed = False
            to_remove = []
            for i in range(len(blocks)):
                is_redundant = True
                for idx in block_indices[i]:
                    if triple_count[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    to_remove.append(i)
            if to_remove:
                remove_set = set(to_remove)
                new_blocks = []
                new_block_indices = []
                for i in range(len(blocks)):
                    if i not in remove_set:
                        new_blocks.append(blocks[i])
                        new_block_indices.append(block_indices[i])
                for i in to_remove:
                    triple_count[block_indices[i]] -= 1
                blocks = new_blocks
                block_indices = new_block_indices
                changed = True
        return blocks

    def local_search(blocks):
        if not blocks:
            return blocks
        triple_count = np.zeros(num_triples, dtype=np.int32)
        block_indices = []
        for block in blocks:
            indices = block_triple_indices(block)
            block_indices.append(indices)
            triple_count[indices] += 1
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            to_remove = []
            for i in range(len(blocks)):
                is_redundant = True
                for idx in block_indices[i]:
                    if triple_count[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    to_remove.append(i)
            if to_remove:
                remove_set = set(to_remove)
                new_blocks = []
                new_block_indices = []
                for i in range(len(blocks)):
                    if i not in remove_set:
                        new_blocks.append(blocks[i])
                        new_block_indices.append(block_indices[i])
                for i in to_remove:
                    triple_count[block_indices[i]] -= 1
                blocks = new_blocks
                block_indices = new_block_indices
                improved = True
                continue

            found_swap = False
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                i = random.randint(0, len(blocks) - 1)
                block = blocks[i]
                old_x = random.choice(block)
                new_x = random.randint(0, v - 1)
                if new_x == old_x or new_x in block:
                    continue
                other_points = [p for p in block if p != old_x]
                new_block = sorted(other_points + [new_x])
                old_indices = set(block_triple_indices(block))
                new_indices = set(block_triple_indices(new_block))
                lost = old_indices - new_indices
                gained = new_indices - old_indices
                valid = True
                for tri_idx in lost:
                    if triple_count[tri_idx] <= 1:
                        valid = False
                        break
                if not valid:
                    continue
                score = sum(1 for tri_idx in lost if triple_count[tri_idx] == 2)
                if score > 0:
                    triple_count[list(old_indices)] -= 1
                    triple_count[list(new_indices)] += 1
                    blocks[i] = new_block
                    block_indices[i] = block_triple_indices(new_block)
                    found_swap = True
                    improved = True
                    break
        return blocks

    best_blocks = None
    best_len = float('inf')
    seed = 42

    while time.time() < deadline - 15.0:
        rng = random.Random(seed)
        seed += 1
        blocks = random_greedy_cover(rng)
        if blocks is None:
            continue
        if not is_valid_cover(blocks):
            continue
        blocks = remove_redundant_blocks(blocks)
        blocks = local_search(blocks)
        if not is_valid_cover(blocks):
            continue
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [b[:] for b in blocks]

    if best_blocks is None:
        blocks = random_greedy_cover(random.Random(seed))
        if blocks:
            best_blocks = blocks
        else:
            uncovered = np.ones(num_triples, dtype=np.uint8)
            blocks = []
            while np.any(uncovered) and time.time() < deadline - 2.0:
                uncovered_indices = np.where(uncovered)[0]
                seed_idx = uncovered_indices[0]
                seed_triple = all_t_subsets[seed_idx]
                block = list(seed_triple)
                block_set = set(block)
                while len(block) < k:
                    best_x = -1
                    best_gain = -1
                    for x in all_points:
                        if x in block_set: continue
                        gain = 0
                        for subset in itertools.combinations(block, t - 1):
                            triple = tuple(sorted(subset + (x,)))
                            idx = triple_to_idx[triple]
                            gain += uncovered[idx]
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x < 0: break
                    block.append(best_x)
                    block_set.add(best_x)
                if len(block) < k:
                    available = [x for x in all_points if x not in block_set]
                    while len(block) < k and available:
                        block.append(available.pop())
                block.sort()
                blocks.append(block)
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx[tuple(triple)]
                    uncovered[idx] = 0
            best_blocks = blocks

    return best_blocks
