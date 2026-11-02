"""Covering design C(19,8,5) - precomputed incidence + randomized greedy + local search."""
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Precompute all t-subsets
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {tset: i for i, tset in enumerate(all_tsets)}

    # Precompute all k-blocks
    all_kblocks = list(itertools.combinations(range(v), k))
    num_kblocks = len(all_kblocks)
    block_to_idx = {tuple(b): i for i, b in enumerate(all_kblocks)}

    # Incidence: block_to_ts[b] = list of t-subset indices covered by block b
    # ts_to_blocks[ts] = list of block indices covering t-subset ts
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_tsets)]
    for j, block in enumerate(all_kblocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = tset_to_idx[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)

    # Each block covers exactly C(k, t) t-subsets
    ts_per_block = len(block_to_ts[0]) if block_to_ts else 0

    def run_greedy(seed_val, deterministic=False):
        """Randomized or deterministic greedy set cover."""
        if not deterministic:
            local_rng = random.Random(seed_val)
        else:
            local_rng = None

        gain = np.full(num_kblocks, ts_per_block, dtype=np.int32)
        is_uncovered = np.full(num_tsets, True, dtype=np.bool_)
        is_selected = np.full(num_kblocks, False, dtype=np.bool_)
        sol = []

        while True:
            if time.time() > deadline - 3.0:
                return None

            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break

            if deterministic:
                b_idx = int(np.argmax(gain))
            else:
                # Randomly pick among max-gain unselected blocks
                candidates = np.where((gain == max_gain) & (~is_selected))[0]
                if len(candidates) == 0:
                    break
                b_idx = int(candidates[local_rng.randint(0, len(candidates) - 1)])

            sol.append(b_idx)
            is_selected[b_idx] = True
            gain[b_idx] = -1

            # Update gains for all blocks sharing t-subsets with b_idx
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1

        return sol

    def polish_blocks(block_indices):
        """Remove redundant blocks (those not uniquely covering any t-subset)."""
        if not block_indices:
            return block_indices

        cover_count = np.zeros(num_tsets, dtype=np.int32)
        for b in block_indices:
            for ts_idx in block_to_ts[b]:
                cover_count[ts_idx] += 1

        i = len(block_indices) - 1
        while i >= 0:
            if time.time() > deadline - 1.0:
                break
            b = block_indices[i]
            can_remove = True
            for ts_idx in block_to_ts[b]:
                if cover_count[ts_idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                for ts_idx in block_to_ts[b]:
                    cover_count[ts_idx] -= 1
                block_indices.pop(i)
            else:
                i -= 1
        return block_indices

    def local_search_swap(block_indices):
        """1-exchange: try replacing one block with another to maintain/improve coverage."""
        if not block_indices:
            return block_indices, False

        cover_count = np.zeros(num_tsets, dtype=np.int32)
        for b in block_indices:
            for ts_idx in block_to_ts[b]:
                cover_count[ts_idx] += 1

        improved = False
        max_iterations = len(block_indices) * 2
        iteration = 0
        while iteration < max_iterations:
            if time.time() > deadline - 1.0:
                break
            iteration += 1

            for i in range(len(block_indices)):
                if time.time() > deadline - 1.0:
                    break
                b = block_indices[i]

                # Find t-subsets uniquely covered by block b
                unique_ts = []
                for ts_idx in block_to_ts[b]:
                    if cover_count[ts_idx] == 1:
                        unique_ts.append(ts_idx)

                if not unique_ts:
                    # This block is redundant; just remove it
                    for ts_idx in block_to_ts[b]:
                        cover_count[ts_idx] -= 1
                    block_indices.pop(i)
                    improved = True
                    break

                # Find candidate blocks that cover ALL unique_ts
                candidates = set(ts_to_blocks[unique_ts[0]])
                for ts_idx in unique_ts[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break

                if not candidates:
                    continue

                candidates.discard(b)

                # Pick a candidate that maximizes new coverage (minimizes redundancy)
                best_c = None
                best_score = -1
                for c in list(candidates):
                    # Score: number of t-subsets in c that are currently covered count > 1
                    # (i.e., adding c creates less redundancy)
                    score = 0
                    for ts_idx in block_to_ts[c]:
                        if cover_count[ts_idx] >= 2:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_c = c

                if best_c is not None:
                    # Replace b with best_c
                    for ts_idx in block_to_ts[b]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_ts[best_c]:
                        cover_count[ts_idx] += 1
                    block_indices[i] = best_c
                    improved = True
                    break

            if improved:
                break  # Restart loop after a change

        return block_indices, improved

    best_blocks = None
    best_len = float('inf')

    # Initial deterministic greedy
    sol = run_greedy(0, deterministic=True)
    if sol is not None:
        sol = polish_blocks(sol)
        sol, _ = local_search_swap(sol)
        sol = polish_blocks(sol)
        best_blocks = sol[:]
        best_len = len(sol)

    # Randomized greedy restarts
    num_restarts = 300
    for i in range(num_restarts):
        if time.time() > deadline - 3.0:
            break

        sol = run_greedy(i + 1)
        if sol is None:
            break

        sol = polish_blocks(sol)
        if len(sol) < best_len:
            sol, _ = local_search_swap(sol)
            sol = polish_blocks(sol)
            if len(sol) < best_len:
                best_len = len(sol)
                best_blocks = sol[:]
        elif len(sol) <= best_len + 5:
            # Close enough, try local search
            sol, _ = local_search_swap(sol)
            sol = polish_blocks(sol)
            if len(sol) < best_len:
                best_len = len(sol)
                best_blocks = sol[:]

    # Final polish on best solution
    if best_blocks is not None:
        best_blocks = polish_blocks(best_blocks)
        best_blocks, _ = local_search_swap(best_blocks)
        best_blocks = polish_blocks(best_blocks)

    if best_blocks is None:
        # Fallback: return empty (shouldn't happen)
        return []

    result = [list(all_kblocks[b]) for b in best_blocks]
    return result
