import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin before 120s kill

    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)

    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Precompute coverage: for each block, which t-subsets it covers
    # tsub_mask[b] = bitset of t-subsets covered by block b
    # block_tsubs[b] = list of t-subset indices covered by block b
    tsub_mask = [0] * n_blocks
    block_tsubs = [[] for _ in range(n_blocks)]

    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            mask |= (1 << idx)
            block_tsubs[b_idx].append(idx)
        tsub_mask[b_idx] = mask

    full_tsub_mask = (1 << n_tsubsets) - 1

    # For each t-subset, which blocks cover it (as a list for fast access)
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    for b_idx in range(n_blocks):
        for ts_idx in block_tsubs[b_idx]:
            tsub_blocks[ts_idx].append(b_idx)

    def greedy_cover(seed, noise_level=0.0):
        """
        Greedy set cover with optional random noise for diversification.
        """
        rng = random.Random(seed)
        uncovered_mask = full_tsub_mask
        selected = []
        remaining = set(range(n_blocks))

        while uncovered_mask and time.time() < deadline - 3.0:
            best_block = -1
            best_gain = 0
            best_score = -1.0

            # Sample a subset of blocks if too many remain
            if len(remaining) > 5000:
                candidates = rng.sample(list(remaining), 5000)
            else:
                candidates = list(remaining)

            for b_idx in candidates:
                gain = (tsub_mask[b_idx] & uncovered_mask).bit_count()
                if gain == 0:
                    continue
                # Add noise for diversification
                score = gain + noise_level * rng.random()
                if score > best_score:
                    best_score = score
                    best_gain = gain
                    best_block = b_idx

            if best_block == -1:
                break

            selected.append(best_block)
            remaining.discard(best_block)
            uncovered_mask &= ~tsub_mask[best_block]

        return selected

    def local_search(blocks, seed, max_iters=200):
        """
        Local search: try to remove blocks, then try 1-for-1 swaps to enable removals.
        """
        rng = random.Random(seed)
        if not blocks:
            return blocks

        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1

        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 2.0:
            improved = False
            iters += 1

            # Phase 1: Remove redundant blocks
            to_remove = []
            for b in list(block_set):
                redundant = True
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b)

            if to_remove:
                for b in to_remove:
                    block_set.discard(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                improved = True
                continue

            # Phase 2: Try 1-for-1 swaps
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            # Focus on blocks with most unique subsets (harder to remove)
            block_uniq = []
            for b in current_blocks:
                uniq = 0
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        uniq += 1
                block_uniq.append((uniq, b))
            block_uniq.sort(reverse=True)

            for uniq, b in block_uniq[:30]:  # Try top 30 blocks
                if time.time() > deadline - 2.0:
                    break
                if b not in block_set:
                    continue
                if uniq == 0:
                    continue

                # Find unique subsets for b
                unique_tsubs = []
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)

                if not unique_tsubs:
                    continue

                # Find candidate blocks covering all unique_tsubs
                # Use intersection of tsub_blocks lists
                cand_set = set(tsub_blocks[unique_tsubs[0]])
                for ts_idx in unique_tsubs[1:]:
                    cand_set &= set(tsub_blocks[ts_idx])
                    if not cand_set:
                        break

                if not cand_set:
                    continue

                cand_set.discard(b)
                candidates = list(cand_set)
                if not candidates:
                    continue

                # Prefer candidates not already in block_set
                new_cands = [c for c in candidates if c not in block_set]
                if new_cands:
                    candidates = new_cands

                rng.shuffle(candidates)
                for r in candidates[:5]:
                    if r in block_set:
                        continue

                    # Perform swap
                    block_set.discard(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                    block_set.add(r)
                    for ts_idx in block_tsubs[r]:
                        cover_count[ts_idx] += 1

                    improved = True
                    break

                if improved:
                    break

        return list(block_set)

    def two_for_one_search(blocks, seed, max_checks=50):
        """
        Try to replace two blocks with one.
        """
        rng = random.Random(seed)
        if len(blocks) < 2:
            return blocks

        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            for _ in range(max_checks):
                if time.time() > deadline - 2.0 or len(block_set) < 2:
                    break

                i, j = rng.sample(range(len(current_blocks)), 2)
                b1, b2 = current_blocks[i], current_blocks[j]
                if b1 not in block_set or b2 not in block_set or b1 == b2:
                    continue

                # Find required mask: t-subsets that would be uncovered if we remove b1 and b2
                required_mask = 0
                for ts_idx in block_tsubs[b1]:
                    if cover_count[ts_idx] <= 1:
                        required_mask |= (1 << ts_idx)
                for ts_idx in block_tsubs[b2]:
                    if cover_count[ts_idx] <= 1:
                        required_mask |= (1 << ts_idx)

                if required_mask == 0:
                    continue

                # Find blocks covering all required t-subsets
                # Get list of set bits in required_mask
                req_list = []
                m = required_mask
                while m:
                    lsb = m & (-m)
                    req_list.append(lsb.bit_length() - 1)
                    m ^= lsb

                if not req_list:
                    continue

                # Intersection of tsub_blocks for all required t-subsets
                cand_set = set(tsub_blocks[req_list[0]])
                for ts_idx in req_list[1:]:
                    cand_set &= set(tsub_blocks[ts_idx])
                    if not cand_set:
                        break

                if not cand_set:
                    continue

                cand_set.discard(b1)
                cand_set.discard(b2)
                candidates = list(cand_set)
                if not candidates:
                    continue

                rng.shuffle(candidates)
                r = candidates[0]

                # Perform 2-for-1 swap
                block_set.discard(b1)
                for ts_idx in block_tsubs[b1]:
                    cover_count[ts_idx] -= 1
                block_set.discard(b2)
                for ts_idx in block_tsubs[b2]:
                    cover_count[ts_idx] -= 1
                block_set.add(r)
                for ts_idx in block_tsubs[r]:
                    cover_count[ts_idx] += 1

                improved = True
                break

        return list(block_set)

    def final_cleanup(blocks):
        """
        Final pass to remove any remaining redundant blocks.
        """
        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1

        changed = True
        while changed:
            changed = False
            to_remove = []
            for b in list(block_set):
                redundant = True
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b)
            for b in to_remove:
                block_set.discard(b)
                for ts_idx in block_tsubs[b]:
                    cover_count[ts_idx] -= 1
                changed = True

        return list(block_set)

    best_solution = None
    best_size = float('inf')

    # Multiple restarts with varying noise
    num_restarts = 12
    for i in range(num_restarts):
        if time.time() > deadline - 15.0:
            break

        seed = 42 + i * 137
        noise = 0.0 if i < 4 else (i % 5) * 0.2  # Vary noise level

        # Greedy
        greedy_sol = greedy_cover(seed, noise_level=noise)
        if not greedy_sol:
            continue

        # Local search
        ls_sol = local_search(greedy_sol, seed, max_iters=150)

        # 2-for-1 search
        t2o_sol = two_for_one_search(ls_sol, seed, max_checks=40)

        # Final cleanup
        final_sol = final_cleanup(t2o_sol)

        if len(final_sol) < best_size:
            best_size = len(final_sol)
            best_solution = final_sol.copy()

    # Fallback
    if best_solution is None:
        best_solution = final_cleanup(greedy_cover(0, noise_level=0.0))

    return [list(all_blocks[idx]) for idx in best_solution]
