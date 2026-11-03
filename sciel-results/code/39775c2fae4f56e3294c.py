import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140

    if v != 19 or k != 7 or t != 4:
        return _greedy_fallback(v, k, t, deadline)

    rng = random.Random(42)
    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_id = {tsub: i for i, tsub in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    all_blocks = []
    block_tsub_ids = []
    for block_tuple in itertools.combinations(points, k):
        block_fs = frozenset(block_tuple)
        tsubs_in_block = [tsub_to_id[tsub] for tsub in itertools.combinations(block_tuple, t)]
        all_blocks.append(block_fs)
        block_tsub_ids.append(tsubs_in_block)

    num_blocks_total = len(all_blocks)

    # Precompute: for each tsub_id, list of block indices that contain it
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for bi, tids in enumerate(block_tsub_ids):
        for tid in tids:
            tsub_to_blocks[tid].append(bi)

    best_blocks = None
    best_len = float('inf')

    # Multiple greedy restarts with different shuffles
    seeds = [42, 1, 7, 13, 99, 123, 456, 789, 1024, 2048, 3333, 7777, 12345, 54321, 99999]

    for seed in seeds:
        if time.time() > deadline - 5:
            break

        rng2 = random.Random(seed)

        # Shuffle block order for diverse greedy
        order = list(range(num_blocks_total))
        rng2.shuffle(order)

        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        blocks = []
        used = set()
        num_uncovered = num_tsubs

        while num_uncovered > 0 and time.time() < deadline - 5:
            # Find the block (in shuffled order) with max gain
            best_gain = -1
            best_idx = -1
            for bi in order:
                if bi in used:
                    continue
                gain = 0
                for tid in block_tsub_ids[bi]:
                    if cover_count[tid] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = bi

            if best_idx < 0 or best_gain <= 0:
                break

            used.add(best_idx)
            blocks.append(all_blocks[best_idx])
            for tid in block_tsub_ids[best_idx]:
                cover_count[tid] += 1
            num_uncovered = int(np.sum(cover_count == 0))

        # Deletion phase
        blocks = _remove_redundant(blocks, block_tsub_ids, tsub_to_blocks, deadline)

        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]

    # Local search: swap-based improvement
    if best_blocks is not None and time.time() < deadline - 10:
        best_blocks = _local_search_swap(best_blocks, all_blocks, block_tsub_ids, tsub_to_blocks,
                                         num_tsubs, num_blocks_total, rng, deadline)
        best_blocks = _remove_redundant(best_blocks, block_tsub_ids, tsub_to_blocks, deadline)
        best_len = len(best_blocks)

    # Final validity check and repair
    final_cover = np.zeros(num_tsubs, dtype=np.int32)
    for b in best_blocks:
        for tid in block_tsub_ids[all_blocks.index(b)]:
            final_cover[tid] += 1

    # Actually, let's use the index-based approach for repair
    # Rebuild with indices
    block_idx_to_fs = {fs: i for i, fs in enumerate(all_blocks)}
    final_cover = np.zeros(num_tsubs, dtype=np.int32)
    for b in best_blocks:
        bi = block_idx_to_fs[b]
        for tid in block_tsub_ids[bi]:
            final_cover[tid] += 1

    if np.any(final_cover == 0) and time.time() < deadline - 1:
        uncovered = set(np.where(final_cover == 0)[0])
        current_set = set(best_blocks)
        while uncovered and time.time() < deadline - 1:
            best_gain = -1
            best_bi = -1
            for bi in range(num_blocks_total):
                if all_blocks[bi] in current_set:
                    continue
                gain = 0
                for tid in block_tsub_ids[bi]:
                    if tid in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
            if best_bi < 0 or best_gain <= 0:
                break
            best_blocks.append(all_blocks[best_bi])
            current_set.add(all_blocks[best_bi])
            for tid in block_tsub_ids[best_bi]:
                final_cover[tid] += 1
                if tid in uncovered:
                    uncovered.discard(tid)

    return [sorted(b) for b in best_blocks]


def _remove_redundant(blocks, block_tsub_ids, tsub_to_blocks, deadline):
    """Remove blocks that don't uniquely cover any t-subset."""
    if not blocks:
        return blocks

    # Map each block in our solution to its index in all_blocks
    fs_to_idx = {}
    for i, b in enumerate(blocks):
        fs_to_idx[b] = i

    # Count coverage: for each tsub, how many blocks in our solution cover it
    # We need the global block index for each solution block
    # Let's use a different approach: build coverage counts directly

    # For each tsub_id, count how many of our blocks cover it
    # We need to map our blocks to their global indices
    # Actually, let's just track by the block objects

    # Build: tsub_id -> count of blocks covering it
    tsub_count = {}
    block_tids = []
    for b in blocks:
        # Find tids for this block - we need global index
        # Let's pass the global index mapping differently
        pass

    # Simpler: work with the tids directly
    # For each block in solution, we need its tsub ids
    # Let's restructure: pass a list of tids for each block in solution

    # Actually, let me just use a set-based approach
    # For each tsub, track which solution blocks cover it
    # Use tsub_id -> list of solution block indices

    # First, map solution blocks to global indices
    global_idx_map = {}
    for i, b in enumerate(blocks):
        global_idx_map[b] = i

    # We need tids for each solution block
    # Let's compute them from the frozenset
    # This is expensive but blocks is small (~150)
    points = list(range(19))
    tsub_set = set()
    for r in range(1, 5):
        for combo in itertools.combinations(points, r):
            pass
    # Actually let's just precompute all 4-subsets
    all_tsubs_list = list(itertools.combinations(range(19), 4))
    tsub_to_global_id = {ts: i for i, ts in enumerate(all_tsubs_list)}

    sol_tids = []
    for b in blocks:
        tids = [tsub_to_global_id[ts] for ts in itertools.combinations(sorted(b), 4)]
        sol_tids.append(tids)

    # Now do the deletion
    tsub_to_sol_blocks = {}
    for i, tids in enumerate(sol_tids):
        for tid in tids:
            if tid not in tsub_to_sol_blocks:
                tsub_to_sol_blocks[tid] = []
            tsub_to_sol_blocks[tid].append(i)

    improved = True
    current_blocks = blocks[:]
    current_sol_tids = sol_tids[:]

    while improved and time.time() < deadline - 2:
        improved = False
        tsub_to_sol_blocks = {}
        for i, tids in enumerate(current_sol_tids):
            for tid in tids:
                if tid not in tsub_to_sol_blocks:
                    tsub_to_sol_blocks[tid] = []
                tsub_to_sol_blocks[tid].append(i)

        removable = []
        for i in range(len(current_blocks)):
            safe = True
            for tid in current_sol_tids[i]:
                if len(tsub_to_sol_blocks[tid]) == 1:
                    safe = False
                    break
            if safe:
                removable.append(i)

        if removable:
            remove_set = set(removable)
            current_blocks = [b for i, b in enumerate(current_blocks) if i not in remove_set]
            current_sol_tids = [t for i, t in enumerate(current_sol_tids) if i not in remove_set]
            improved = True

    return current_blocks


def _local_search_swap(blocks, all_blocks, block_tsub_ids, tsub_to_blocks,
                       num_tsubs, num_blocks_total, rng, deadline):
    """Try to swap blocks to improve the solution."""
    if not blocks:
        return blocks

    # Build coverage count
    cover = np.zeros(num_tsubs, dtype=np.int32)
    for b in blocks:
        # Find global index
        # This is O(n) per block, but blocks is small
        pass

    # Let's use a dict-based approach
    fs_set = set(blocks)

    # Build coverage
    for b in blocks:
        for i, ab in enumerate(all_blocks):
            if ab == b:
                for tid in block_tsub_ids[i]:
                    cover[tid] += 1
                break

    # For each block in solution, compute its unique coverage (tsubs only it covers)
    # For each candidate replacement, compute net change

    current = blocks[:]
    for iteration in range(200):
        if time.time() > deadline - 5:
            break

        idx = rng.randint(0, len(current) - 1)
        old_block = current[idx]

        # Find old block's global index
        old_gi = -1
        for i, ab in enumerate(all_blocks):
            if ab == old_block:
                old_gi = i
                break
        if old_gi < 0:
            continue

        # Temporarily remove old block
        for tid in block_tsub_ids[old_gi]:
            cover[tid] -= 1

        # Find best replacement
        best_repl = None
        best_net = 0

        for _ in range(30):
            cand_gi = rng.randint(0, num_blocks_total - 1)
            cand = all_blocks[cand_gi]
            if cand in fs_set:
                continue

            # Compute net gain
            cand_gain = 0
            for tid in block_tsub_ids[cand_gi]:
                if cover[tid] == 0:
                    cand_gain += 1

            # Old loss: tsubs that become uncovered
            old_loss = 0
            for tid in block_tsub_ids[old_gi]:
                if cover[tid] == 0:
                    old_loss += 1

            net = cand_gain - old_loss
            if net > best_net:
                best_net = net
                best_repl = cand_gi

        # Restore old block
        for tid in block_tsub_ids[old_gi]:
            cover[tid] += 1

        if best_repl is not None and best_net > 0:
            # Actually apply the swap
            # Remove old
            for tid in block_tsub_ids[old_gi]:
                cover[tid] -= 1
            # Add new
            for tid in block_tsub_ids[best_repl]:
                cover[tid] += 1
            current[idx] = all_blocks[best_repl]
            fs_set = set(current)

    return current


def _greedy_fallback(v, k, t, deadline):
    """Simple greedy fallback for non-target parameters."""
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_id = {tsub: i for i, tsub in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    all_blocks = []
    block_tsubs = {}
    for block_tuple in itertools.combinations(range(v), k):
        block_fs = frozenset(block_tuple)
        tsubs_in_block = [tsub_to_id[tsub] for tsub in itertools.combinations(block_tuple, t)]
        all_blocks.append(block_fs)
        block_tsubs[block_fs] = tsubs_in_block

    cover_count = np.zeros(num_tsubs, dtype=np.int32)
    blocks = []

    while True:
        num_uncovered = int(np.sum(cover_count == 0))
        if num_uncovered == 0 or time.time() > deadline:
            break

        best_block = None
        best_gain = -1
        for b in all_blocks:
            gain = sum(1 for tid in block_tsubs[b] if cover_count[tid] == 0)
            if gain > best_gain:
                best_gain = gain
                best_block = b

        if best_block is None or best_gain <= 0:
            break

        blocks.append(best_block)
        for tid in block_tsubs[best_block]:
            cover_count[tid] += 1

    return [sorted(b) for b in blocks]
