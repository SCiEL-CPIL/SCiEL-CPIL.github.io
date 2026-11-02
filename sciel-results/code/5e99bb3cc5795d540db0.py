"""
Covering design solver: dynamic greedy with heap-based priority updates,
combined with multi-restart randomization and aggressive local search.
Strategy:
  1. Precompute all C(v,k) blocks and their t-subset memberships.
  2. Dynamic greedy: max-heap of blocks keyed by current uncovered-gain.
  3. Multiple randomized restarts (different tie-breaking / orderings).
  4. Deletion phase: remove redundant blocks.
  5. Local search: try to replace a block with a smaller-covering block,
     or remove redundant blocks iteratively.
"""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import heapq
from collections import defaultdict

_DEADLINE = 145.0
_start_time = None


def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def construct(v, k, t):
    global _start_time
    _start_time = time.time()

    rng = random.Random(12345)

    # All t-subsets
    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_index = {ts: i for i, ts in enumerate(all_ts)}

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, precompute the list of t-subset indices it covers
    block_ts = []
    for blk in all_blocks:
        ts_list = [ts_index[ts] for ts in itertools.combinations(blk, t)]
        block_ts.append(ts_list)

    # Precompute: for each t-subset index, which blocks contain it
    ts_to_blocks = defaultdict(list)
    for b_idx, ts_list in enumerate(block_ts):
        for s_idx in ts_list:
            ts_to_blocks[s_idx].append(b_idx)

    def greedy_solve(seed, max_time_frac=0.5):
        """Run one greedy pass with given random seed for tie-breaking."""
        local_rng = random.Random(seed)
        time_budget = _DEADLINE * max_time_frac
        local_start = time.time()

        uncovered = [True] * n_ts
        uncovered_count = n_ts

        # Current gain for each block
        block_gain = [len(ts_list) for ts_list in block_ts]
        block_version = [0] * n_blocks
        in_block = [False] * n_blocks

        # Heap: (-gain, block_index, version)
        heap = [(-block_gain[i], i, 0) for i in range(n_blocks)]
        heapq.heapify(heap)

        selected = []

        while uncovered_count > 0:
            if time.time() - local_start > time_budget:
                break

            # Pop until valid
            valid_entry = None
            while heap:
                neg_gain, b_idx, ver = heapq.heappop(heap)
                if in_block[b_idx]:
                    continue
                if ver != block_version[b_idx]:
                    continue
                if block_gain[b_idx] <= 0:
                    continue
                valid_entry = (neg_gain, b_idx, ver)
                break

            if valid_entry is None:
                break

            neg_gain, b_idx, ver = valid_entry
            selected.append(b_idx)
            in_block[b_idx] = True

            # Update coverage
            for s_idx in block_ts[b_idx]:
                if not uncovered[s_idx]:
                    continue
                uncovered[s_idx] = False
                uncovered_count -= 1

                # Update gains of other blocks containing this t-subset
                for other_b in ts_to_blocks[s_idx]:
                    if in_block[other_b]:
                        continue
                    block_gain[other_b] -= 1
                    block_version[other_b] += 1
                    if block_gain[other_b] > 0:
                        heapq.heappush(heap, (-block_gain[other_b], other_b, block_version[other_b]))

        # Convert to list of blocks
        result = [list(all_blocks[b]) for b in selected]

        # Check if complete
        if uncovered_count > 0:
            # Add remaining blocks greedily (brute force over remaining)
            unc_set = set(i for i in range(n_ts) if not uncovered[i])
            while unc_set and time.time() - local_start < _DEADLINE - 2.0:
                best_b = -1
                best_gain = 0
                for b_idx in range(n_blocks):
                    if in_block[b_idx]:
                        continue
                    gain = sum(1 for s in block_ts[b_idx] if s in unc_set)
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b_idx
                if best_b == -1 or best_gain == 0:
                    break
                result.append(list(all_blocks[best_b]))
                in_block[best_b] = True
                for s in block_ts[best_b]:
                    unc_set.discard(s)

        return result

    def local_search_improve(blocks, seed, time_limit):
        """Improve a covering by removing redundant blocks and trying swaps."""
        local_rng = random.Random(seed)
        local_start = time.time()
        n_blocks_cur = len(blocks)

        if not blocks:
            return blocks

        # Build coverage counts
        coverage = [0] * n_ts
        for blk in blocks:
            for ts in itertools.combinations(blk, t):
                coverage[ts_index[ts]] += 1

        # Phase 1: Remove redundant blocks
        changed = True
        while changed and time.time() - local_start < time_limit:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() - local_start > time_limit:
                    break
                blk = blocks[i]
                redundant = True
                for ts in itertools.combinations(blk, t):
                    if coverage[ts_index[ts]] <= 1:
                        redundant = False
                        break
                if redundant:
                    for ts in itertools.combinations(blk, t):
                        coverage[ts_index[ts]] -= 1
                    blocks.pop(i)
                    changed = True
                    break

        # Phase 2: Try to replace blocks with better alternatives
        # For each block, find its "unique" t-subsets (covered only by this block)
        # If the union of points in unique t-subsets has size <= k, we might
        # be able to find a replacement block that covers them plus others.
        # This is complex; instead, try random block removal + re-cover.
        attempts = 0
        max_attempts = 500
        while (time.time() - local_start < time_limit and attempts < max_attempts
               and len(blocks) > 0):
            attempts += 1
            idx = local_rng.randint(0, len(blocks) - 1)
            old_block = blocks[idx]

            # Find unique t-subsets
            unique = []
            for ts in itertools.combinations(old_block, t):
                if coverage[ts_index[ts]] == 1:
                    unique.append(ts_index[ts])

            if not unique:
                # Redundant - remove
                for ts in itertools.combinations(old_block, t):
                    coverage[ts_index[ts]] -= 1
                blocks.pop(idx)
                continue

            # Try to find a replacement block that covers all unique t-subsets
            # and minimizes additional cost
            union_pts = set()
            for s_idx in unique:
                for p in all_ts[s_idx]:
                    union_pts.add(p)

            if len(union_pts) > k:
                continue

            # Find best replacement
            best_repl = -1
            best_score = float('inf')
            for b_idx in range(n_blocks):
                blk = all_blocks[b_idx]
                # Must cover all unique t-subsets
                blk_set = set(blk)
                ok = True
                for s_idx in unique:
                    if not all(p in blk_set for p in all_ts[s_idx]):
                        ok = False
                        break
                if not ok:
                    continue

                # Score: number of NEW t-subsets this block covers (not already covered)
                # Actually, we want to minimize the "loss" - how many t-subsets
                # become uncovered if we remove old_block and add new_block
                # Simpler: count how many t-subsets of new_block are already covered
                new_gain = 0
                for ts in itertools.combinations(blk, t):
                    if coverage[ts_index[ts]] > 0:
                        new_gain += 1

                # We want to maximize new_gain (more overlap = better)
                # Or equivalently minimize (C(k,t) - new_gain)
                score = 35 - new_gain  # C(7,4) = 35
                if score < best_score:
                    best_score = score
                    best_repl = b_idx

            if best_repl == -1:
                continue

            # Check if replacement is strictly better (covers more already-covered)
            # For simplicity, only accept if it's the same block or better
            new_block = list(all_blocks[best_repl])

            # Apply swap: remove old, add new
            for ts in itertools.combinations(old_block, t):
                coverage[ts_index[ts]] -= 1
            for ts in itertools.combinations(new_block, t):
                coverage[ts_index[ts]] += 1

            # Verify no t-subset became uncovered
            valid = True
            for ts in itertools.combinations(old_block, t):
                if coverage[ts_index[ts]] == 0:
                    # This t-subset is now uncovered - bad
                    valid = False
                    break

            if valid:
                blocks[idx] = new_block
            else:
                # Revert
                for ts in itertools.combinations(new_block, t):
                    coverage[ts_index[ts]] -= 1
                for ts in itertools.combinations(old_block, t):
                    coverage[ts_index[ts]] += 1

        return blocks

    # Multi-restart greedy
    best_result = None
    best_count = float('inf')

    # Determine number of restarts based on time
    n_restarts = 8
    time_per_restart = (_DEADLINE - 10.0) / (n_restarts * 2)  # leave time for local search

    for restart in range(n_restarts):
        if _remaining() < 5.0:
            break

        seed = 12345 + restart * 7919
        max_frac = min(0.6, time_per_restart / _DEADLINE)

        result = greedy_solve(seed, max_time_frac=max_frac)

        if not result:
            continue

        # Check validity
        cov_check = [0] * n_ts
        for blk in result:
            for ts in itertools.combinations(blk, t):
                cov_check[ts_index[ts]] += 1
        if not all(c > 0 for c in cov_check):
            # Invalid - skip
            continue

        # Local search
        ls_time = min(5.0, _remaining() - 2.0)
        if ls_time > 1.0:
            result = local_search_improve(result, seed + 1, ls_time)

        if len(result) < best_count:
            best_count = len(result)
            best_result = result

    if best_result is None:
        # Fallback: single greedy pass
        best_result = greedy_solve(12345, max_time_frac=0.9)

    # Final verification
    final_cov = [0] * n_ts
    for blk in best_result:
        for ts in itertools.combinations(blk, t):
            final_cov[ts_index[ts]] += 1
    if not all(c > 0 for c in final_cov):
        # Emergency fix: add blocks for uncovered subsets
        unc_set = set(i for i in range(n_ts) if final_cov[i] == 0)
        while unc_set and _remaining() > 0.5:
            best_b = -1
            best_gain = 0
            for b_idx in range(n_blocks):
                gain = sum(1 for s in block_ts[b_idx] if s in unc_set)
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
            if best_b == -1 or best_gain == 0:
                break
            best_result.append(list(all_blocks[best_b]))
            for s in block_ts[best_b]:
                unc_set.discard(s)

    return best_result
# EVOLVE-BLOCK-END
