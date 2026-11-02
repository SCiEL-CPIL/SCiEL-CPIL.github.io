import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    def greedy_cover(seed, time_limit):
        rng = random.Random(seed)
        uncovered = np.ones(num_tsubsets, dtype=bool)
        blocks = []

        while np.any(uncovered) and time.time() < time_limit:
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            start_idx = rng.choice(uncovered_indices)
            start_combo = all_tsubsets[start_idx]
            block = list(start_combo)

            while len(block) < k:
                best_x = -1
                best_gain = -1
                block_set = set(block)
                block_pairs = list(itertools.combinations(block, t - 1))

                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for p1, p2 in block_pairs:
                        triple = tuple(sorted((x, p1, p2)))
                        idx = tsubset_to_idx.get(triple, -1)
                        if idx >= 0 and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_gain > 0:
                        if rng.random() < 0.5:
                            best_x = x

                if best_x < 0:
                    break
                block.append(best_x)

            if len(block) < k:
                continue

            block.sort()
            blocks.append(tuple(block))

            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    uncovered[idx] = False

        if not np.any(uncovered):
            return [list(b) for b in blocks]
        return None

    def remove_redundant(blocks, time_limit):
        if not blocks:
            return []
        if time.time() > time_limit:
            return blocks

        cover_count = np.zeros(num_tsubsets, dtype=np.int32)
        block_tsubs = []
        for b in blocks:
            ts_list = list(itertools.combinations(b, t))
            block_tsubs.append(ts_list)
            for ts in ts_list:
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    cover_count[idx] += 1

        result_blocks = list(blocks)
        result_tsubs = list(block_tsubs)

        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(len(result_blocks)):
                if time.time() > time_limit:
                    break
                redundant = True
                for ts in result_tsubs[i]:
                    idx = tsubset_to_idx.get(ts, -1)
                    if idx >= 0 and cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for ts in result_tsubs[i]:
                        idx = tsubset_to_idx.get(ts, -1)
                        if idx >= 0:
                            cover_count[idx] -= 1
                    result_blocks.pop(i)
                    result_tsubs.pop(i)
                    changed = True
                    break

        return [list(b) for b in result_blocks]

    def local_search_refine(blocks, seed, time_limit):
        if not blocks or len(blocks) <= 1:
            return blocks

        rng = random.Random(seed)
        blocks = [list(b) for b in blocks]

        cover_count = np.zeros(num_tsubsets, dtype=np.int32)
        block_tsubs = []
        for b in blocks:
            ts_list = list(itertools.combinations(b, t))
            block_tsubs.append(ts_list)
            for ts in ts_list:
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    cover_count[idx] += 1

        best_blocks = [list(b) for b in blocks]
        best_count = len(blocks)
        no_improve_count = 0
        max_no_improve = 30

        while time.time() < time_limit and no_improve_count < max_no_improve:
            if len(blocks) <= best_count:
                no_improve_count += 1
            else:
                no_improve_count = 0

            if len(blocks) == 0:
                break

            # Try to delete a block
            deleted = False
            for _ in range(20):
                if time.time() > time_limit:
                    break
                idx_remove = rng.randint(0, len(blocks) - 1)
                block_to_remove = blocks[idx_remove]
                tsubs_to_remove = block_tsubs[idx_remove]

                unique_triples = []
                for ts in tsubs_to_remove:
                    idx = tsubset_to_idx.get(ts, -1)
                    if idx >= 0 and cover_count[idx] == 1:
                        unique_triples.append(ts)

                if not unique_triples:
                    for ts in tsubs_to_remove:
                        idx = tsubset_to_idx.get(ts, -1)
                        if idx >= 0:
                            cover_count[idx] -= 1
                    blocks.pop(idx_remove)
                    block_tsubs.pop(idx_remove)
                    deleted = True
                    no_improve_count = 0
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = [list(b) for b in blocks]
                    break

                # Try to fix unique triples by swapping points in other blocks
                fixed = True
                for triple in unique_triples:
                    a, b, c = triple
                    found = False
                    for i, b in enumerate(blocks):
                        if i == idx_remove:
                            continue
                        b_set = set(b)
                        count_in_b = sum(1 for p in (a, b, c) if p in b_set)
                        if count_in_b == 2:
                            missing = [p for p in (a, b, c) if p not in b_set][0]
                            # Try to swap a point in block i with missing
                            # Find a point y in block i that can be swapped out
                            for y in b:
                                if y == missing:
                                    continue
                                # Check if swapping y with missing is valid
                                # Simulate: remove y, add missing
                                new_block = [p for p in b if p != y] + [missing]
                                new_block.sort()
                                new_tsubs = list(itertools.combinations(new_block, t))
                                old_tsubs = list(itertools.combinations(b, t))

                                # Check coverage loss
                                valid = True
                                for ts in old_tsubs:
                                    idx = tsubset_to_idx.get(ts, -1)
                                    if idx >= 0 and cover_count[idx] == 1 and ts not in new_tsubs:
                                        valid = False
                                        break
                                if valid:
                                    # Apply swap
                                    for ts in old_tsubs:
                                        idx = tsubset_to_idx.get(ts, -1)
                                        if idx >= 0:
                                            cover_count[idx] -= 1
                                    for ts in new_tsubs:
                                        idx = tsubset_to_idx.get(ts, -1)
                                        if idx >= 0:
                                            cover_count[idx] += 1
                                    blocks[i] = new_block
                                    block_tsubs[i] = new_tsubs
                                    found = True
                                    break
                        if found:
                            break
                    if not found:
                        fixed = False
                        break

                if fixed:
                    # Now remove the original block
                    for ts in tsubs_to_remove:
                        idx = tsubset_to_idx.get(ts, -1)
                        if idx >= 0:
                            cover_count[idx] -= 1
                    blocks.pop(idx_remove)
                    block_tsubs.pop(idx_remove)
                    deleted = True
                    no_improve_count = 0
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = [list(b) for b in blocks]
                    break

            if not deleted:
                # Try random point swaps to perturb
                improved = False
                for _ in range(30):
                    if time.time() > time_limit:
                        break
                    idx = rng.randint(0, len(blocks) - 1)
                    old_x = rng.choice(blocks[idx])
                    new_x = rng.randint(0, v - 1)
                    if new_x == old_x or new_x in blocks[idx]:
                        continue

                    old_block = blocks[idx]
                    old_tsubs = list(itertools.combinations(old_block, t))
                    new_block = sorted([p for p in old_block if p != old_x] + [new_x])
                    new_tsubs = list(itertools.combinations(new_block, t))

                    valid = True
                    for ts in old_tsubs:
                        idx_ts = tsubset_to_idx.get(ts, -1)
                        if idx_ts >= 0 and cover_count[idx_ts] == 1 and ts not in new_tsubs:
                            valid = False
                            break
                    if valid:
                        for ts in old_tsubs:
                            idx_ts = tsubset_to_idx.get(ts, -1)
                            if idx_ts >= 0:
                                cover_count[idx_ts] -= 1
                        for ts in new_tsubs:
                            idx_ts = tsubset_to_idx.get(ts, -1)
                            if idx_ts >= 0:
                                cover_count[idx_ts] += 1
                        blocks[idx] = new_block
                        block_tsubs[idx] = new_tsubs
                        improved = True
                        break

                if improved:
                    no_improve_count = 0
                    # After perturbation, try to delete blocks
                    for _ in range(10):
                        if time.time() > time_limit:
                            break
                        idx_del = rng.randint(0, len(blocks) - 1)
                        tsubs_del = block_tsubs[idx_del]
                        all_redundant = True
                        for ts in tsubs_del:
                            idx_ts = tsubset_to_idx.get(ts, -1)
                            if idx_ts >= 0 and cover_count[idx_ts] == 1:
                                all_redundant = False
                                break
                        if all_redundant:
                            for ts in tsubs_del:
                                idx_ts = tsubset_to_idx.get(ts, -1)
                                if idx_ts >= 0:
                                    cover_count[idx_ts] -= 1
                            blocks.pop(idx_del)
                            block_tsubs.pop(idx_del)
                            if len(blocks) < best_count:
                                best_count = len(blocks)
                                best_blocks = [list(b) for b in blocks]
                            break

        return best_blocks

    best_blocks = None
    best_count = float('inf')

    seed = 0
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 10:
            break

        attempt_time = min(remaining - 5, 8)
        blocks = greedy_cover(seed, time.time() + attempt_time)

        if blocks is not None:
            red_time = min(remaining - 5, 5)
            blocks = remove_redundant(blocks, time.time() + red_time)

            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks

            # Local search refinement
            ls_time = min(remaining - 3, 15)
            refined = local_search_refine(blocks, seed, time.time() + ls_time)
            if refined and len(refined) < best_count:
                best_count = len(refined)
                best_blocks = refined

        seed += 1

    if best_blocks is None:
        uncovered = np.ones(num_tsubsets, dtype=bool)
        blocks = []
        while np.any(uncovered):
            start_idx = np.where(uncovered)[0][0]
            start_combo = all_tsubsets[start_idx]
            block = list(start_combo)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                block_set = set(block)
                block_pairs = list(itertools.combinations(block, t - 1))
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for p1, p2 in block_pairs:
                        triple = tuple(sorted((x, p1, p2)))
                        idx = tsubset_to_idx.get(triple, -1)
                        if idx >= 0 and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x < 0:
                    break
                block.append(best_x)
            if len(block) < k:
                break
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    uncovered[idx] = False
        return blocks

    return best_blocks
