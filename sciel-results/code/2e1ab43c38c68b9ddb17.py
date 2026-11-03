import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 115.0

    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)

    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Precompute which t-subsets each block covers
    block_tsubs = [None] * n_blocks
    tsub_blocks = [[] for _ in range(n_tsubsets)]

    for bi, block in enumerate(all_blocks):
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs[bi] = tsubs
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)

    # Precompute point frequencies in blocks for tie-breaking
    point_block_count = np.zeros(v, dtype=np.int32)
    for bi in range(n_blocks):
        for p in all_blocks[bi]:
            point_block_count[p] += 1

    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = [True] * n_tsubsets
        gain = [0] * n_blocks
        for bi in range(n_blocks):
            gain[bi] = len(block_tsubs[bi])

        selected = []

        while True:
            if time.time() > deadline - 5:
                break

            # Find block with max gain, with random tie-breaking
            max_gain = -1
            best_bi = -1
            candidates = []
            for bi in range(n_blocks):
                if gain[bi] > max_gain:
                    max_gain = gain[bi]
                    candidates = [bi]
                elif gain[bi] == max_gain and gain[bi] > 0:
                    candidates.append(bi)

            if max_gain <= 0:
                break

            bi = candidates[rng.randint(0, len(candidates) - 1)]
            selected.append(bi)

            for ts_idx in block_tsubs[bi]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    for bj in tsub_blocks[ts_idx]:
                        gain[bj] -= 1

            gain[bi] = -1

        return selected

    def local_search(blocks, seed, max_rounds=10):
        rng = random.Random(seed)
        block_set = set(blocks)
        cover_count = [0] * n_tsubsets

        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1

        improved = True
        rounds = 0
        while improved and time.time() < deadline - 2 and rounds < max_rounds:
            improved = False
            rounds += 1

            # Phase 1: Remove redundant blocks
            to_remove = []
            for bi in block_set:
                redundant = True
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(bi)

            if to_remove:
                for bi in to_remove:
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                improved = True
                continue

            # Phase 2: Try single-block swaps
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            for bi in current_blocks:
                if time.time() > deadline - 2:
                    break
                if bi not in block_set:
                    continue

                unique_tsubs = []
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)

                if not unique_tsubs:
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break

                min_ts = min(unique_tsubs, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_blocks[min_ts])

                for ts_idx in unique_tsubs[1:]:
                    if not candidates:
                        break
                    candidates &= set(tsub_blocks[ts_idx])

                candidates.discard(bi)

                if not candidates:
                    continue

                cand_list = list(candidates)
                rng.shuffle(cand_list)

                swapped = False
                for bj in cand_list[:30]:
                    if bj in block_set:
                        continue

                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1

                    block_set.add(bj)
                    for ts_idx in block_tsubs[bj]:
                        cover_count[ts_idx] += 1

                    improved = True
                    swapped = True
                    break

                if swapped:
                    break

        return list(block_set)

    def two_block_swap_search(blocks, seed, max_rounds=5):
        rng = random.Random(seed)
        block_set = set(blocks)
        cover_count = [0] * n_tsubsets

        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1

        improved = True
        rounds = 0
        while improved and time.time() < deadline - 2 and rounds < max_rounds:
            improved = False
            rounds += 1
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            n_checked = 0
            for i, bi in enumerate(current_blocks):
                if time.time() > deadline - 2:
                    break
                if bi not in block_set:
                    continue
                if len(block_set) < 2:
                    break

                others = [b for b in block_set if b != bi]
                if not others:
                    continue
                bj = rng.choice(others)

                n_checked += 1
                if n_checked > 8000:
                    break

                req_bi = [ts for ts in block_tsubs[bi] if cover_count[ts] == 1]
                req_bj = [ts for ts in block_tsubs[bj] if cover_count[ts] == 1]

                set_bi = set(block_tsubs[bi])
                set_bj = set(block_tsubs[bj])

                required = set(req_bi)
                required.update(req_bj)

                inter = set_bi & set_bj
                for ts in inter:
                    if cover_count[ts] == 2:
                        required.add(ts)

                if not required:
                    continue

                min_ts = min(required, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_blocks[min_ts])

                for ts in required:
                    if ts == min_ts:
                        continue
                    if not candidates:
                        break
                    candidates &= set(tsub_blocks[ts])

                candidates.discard(bi)
                candidates.discard(bj)

                if not candidates:
                    continue

                cand_list = list(candidates)
                rng.shuffle(cand_list)

                swapped = False
                for bk in cand_list[:15]:
                    if bk in block_set:
                        continue

                    block_set.remove(bi)
                    for ts in block_tsubs[bi]:
                        cover_count[ts] -= 1

                    block_set.remove(bj)
                    for ts in block_tsubs[bj]:
                        cover_count[ts] -= 1

                    block_set.add(bk)
                    for ts in block_tsubs[bk]:
                        cover_count[ts] += 1

                    improved = True
                    swapped = True
                    break

                if swapped:
                    break

        return list(block_set)

    def three_block_swap_search(blocks, seed, max_rounds=3):
        """Try to replace three blocks with one or two blocks."""
        rng = random.Random(seed)
        block_set = set(blocks)
        cover_count = [0] * n_tsubsets

        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1

        improved = True
        rounds = 0
        while improved and time.time() < deadline - 2 and rounds < max_rounds:
            improved = False
            rounds += 1
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            n_checked = 0
            for i, bi in enumerate(current_blocks):
                if time.time() > deadline - 2:
                    break
                if bi not in block_set:
                    continue
                if len(block_set) < 3:
                    break

                others = [b for b in block_set if b != bi]
                if len(others) < 2:
                    continue
                bj = rng.choice(others)
                remaining = [b for b in others if b != bj]
                bk = rng.choice(remaining)

                n_checked += 1
                if n_checked > 5000:
                    break

                # Required t-subsets: those covered only by bi, bj, or bk
                required = set()
                for ts in block_tsubs[bi]:
                    if cover_count[ts] == 1:
                        required.add(ts)
                for ts in block_tsubs[bj]:
                    if cover_count[ts] == 1:
                        required.add(ts)
                for ts in block_tsubs[bk]:
                    if cover_count[ts] == 1:
                        required.add(ts)

                # Also handle pairwise intersections with count 2
                set_bi = set(block_tsubs[bi])
                set_bj = set(block_tsubs[bj])
                set_bk = set(block_tsubs[bk])

                for ts in (set_bi & set_bj):
                    if cover_count[ts] == 2:
                        required.add(ts)
                for ts in (set_bi & set_bk):
                    if cover_count[ts] == 2:
                        required.add(ts)
                for ts in (set_bj & set_bk):
                    if cover_count[ts] == 2:
                        required.add(ts)

                if not required:
                    continue

                # Try to find a single block covering all required
                min_ts = min(required, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_blocks[min_ts])

                for ts in required:
                    if ts == min_ts:
                        continue
                    if not candidates:
                        break
                    candidates &= set(tsub_blocks[ts])

                candidates.discard(bi)
                candidates.discard(bj)
                candidates.discard(bk)

                if candidates:
                    cand_list = list(candidates)
                    rng.shuffle(cand_list)
                    for bl in cand_list[:10]:
                        if bl in block_set:
                            continue

                        for b in [bi, bj, bk]:
                            block_set.remove(b)
                            for ts in block_tsubs[b]:
                                cover_count[ts] -= 1

                        block_set.add(bl)
                        for ts in block_tsubs[bl]:
                            cover_count[ts] += 1

                        improved = True
                        break

                    if improved:
                        break

        return list(block_set)

    best_blocks = None
    best_len = float('inf')

    num_trials = 80
    for i in range(num_trials):
        if time.time() > deadline - 40:
            break

        blocks = greedy_cover(i)
        if not blocks:
            continue

        blocks = local_search(blocks, i, max_rounds=8)

        if time.time() < deadline - 25:
            blocks = two_block_swap_search(blocks, i + 1000, max_rounds=4)

        if time.time() < deadline - 15:
            blocks = three_block_swap_search(blocks, i + 2000, max_rounds=2)

        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)

    # Final polish on best solution
    if best_blocks is not None and time.time() < deadline - 15:
        best_blocks = local_search(best_blocks, 99999, max_rounds=15)
        if time.time() < deadline - 10:
            best_blocks = two_block_swap_search(best_blocks, 88888, max_rounds=6)
        if time.time() < deadline - 5:
            best_blocks = three_block_swap_search(best_blocks, 77777, max_rounds=3)

    if best_blocks is None:
        best_blocks = greedy_cover(0)

    return [all_blocks[i] for i in best_blocks]
