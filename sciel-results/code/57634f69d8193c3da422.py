import time
import numpy as np
import random
import itertools

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute all t-subsets and their indices
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    num_t = len(all_t_subsets)
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}

    # For each point, list of t-subsets containing it (for quick lookup)
    point_tsubs = [[] for _ in range(v)]
    for ts in all_t_subsets:
        for p in ts:
            point_tsubs[p].append(t_to_idx[ts])

    # Build reverse index: for each t-subset, which blocks contain it
    # C(19,7) = 50388 blocks, each covers C(7,4)=35 t-subsets
    # Total entries ~ 1.76M
    # Store as list of numpy arrays
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    block_tsubs = []  # for each block, array of t-subset indices it covers
    for block in all_blocks:
        subs = list(itertools.combinations(block, t))
        block_tsubs.append(np.array([t_to_idx[s] for s in subs], dtype=np.int32))

    # Reverse index: t_subset_idx -> list of block indices covering it
    # This is expensive but very useful for repair
    # t_subset_to_blocks[i] = list of block indices that contain t_subset i
    t_to_blocks = [[] for _ in range(num_t)]
    for bi, tsubs in enumerate(block_tsubs):
        for ts_idx in tsubs:
            t_to_blocks[ts_idx].append(bi)

    # Convert to arrays for faster access
    t_to_blocks = [np.array(lst, dtype=np.int32) for lst in t_to_blocks]

    # ===================== PHASE 1: PACKING-FIRST CORE =====================
    # Build a core of blocks maximizing distinct 4-subset coverage
    # Greedy: at each step, pick the block that covers the most uncovered t-subsets
    # But with a twist: prefer blocks with less overlap with already-selected blocks
    # (i.e., maximize marginal new coverage)
    
    uncovered = np.ones(num_t, dtype=np.int32)  # 1 = uncovered
    core = []
    cover_counts = np.zeros(num_t, dtype=np.int32)

    while True:
        if time.time() > deadline - 50.0:
            break
        if np.all(uncovered == 0):
            break

        # Find block with maximum marginal coverage
        # marginal[b] = sum of uncovered[t] for t in block_tsubs[b]
        best_block = -1
        best_gain = 0
        for bi in range(num_blocks):
            tsubs = block_tsubs[bi]
            gain = np.sum(uncovered[tsubs])
            if gain > best_gain:
                best_gain = gain
                best_block = bi
            elif gain == best_gain and best_block != -1:
                # Tie-break: prefer less overlap (lower sum of cover_counts)
                overlap_new = np.sum(cover_counts[tsubs])
                overlap_old = np.sum(cover_counts[block_tsubs[best_block]])
                if overlap_new < overlap_old:
                    best_block = bi

        if best_block == -1 or best_gain == 0:
            break

        core.append(best_block)
        cover_counts[block_tsubs[best_block]] += 1
        uncovered[block_tsubs[best_block]] = 0

    core_size = len(core)

    # ===================== PHASE 2: TARGETED REPAIR =====================
    # If core doesn't cover everything, repair with targeted additions
    current_cover = set(core)
    # Rebuild cover_counts
    cover_counts = np.zeros(num_t, dtype=np.int32)
    for bi in core:
        cover_counts[block_tsubs[bi]] += 1

    # Repair: repeatedly pick uncovered t-subset, add a block covering it
    # Prefer blocks that also cover other uncovered subsets
    repair_count = 0
    while True:
        if time.time() > deadline - 40.0:
            break
        uncovered_mask = (cover_counts == 0)
        if not np.any(uncovered_mask):
            break
        unc_idx = np.where(uncovered_mask)[0]
        if len(unc_idx) == 0:
            break

        # Pick an uncovered t-subset (random for diversity)
        u = unc_idx[random.randint(0, len(unc_idx) - 1)]

        # Find best block covering u that also covers most other uncovered
        candidates = t_to_blocks[u]
        if len(candidates) == 0:
            break
        best_b = -1
        best_score = -1
        for bi in candidates:
            if bi in current_cover:
                continue
            tsubs = block_tsubs[bi]
            # Score: number of uncovered t-subsets this block covers
            score = np.sum(cover_counts[tsubs] == 0)
            # Small penalty for overlap
            overlap = np.sum(cover_counts[tsubs] > 0)
            score = score - 0.01 * overlap
            if score > best_score:
                best_score = score
                best_b = bi

        if best_b == -1:
            # All candidates already in cover; pick any
            for bi in candidates:
                if bi not in current_cover:
                    best_b = bi
                    break
            if best_b == -1:
                best_b = candidates[0]

        current_cover.add(best_b)
        cover_counts[block_tsubs[best_b]] += 1
        repair_count += 1

    # ===================== PHASE 3: REDUNDANCY REMOVAL =====================
    def reduce_cover(block_list):
        if not block_list:
            return []
        indices = list(block_list)
        cc = np.zeros(num_t, dtype=np.int32)
        for bi in indices:
            cc[block_tsubs[bi]] += 1
        changed = True
        while changed:
            changed = False
            to_remove = []
            for i, bi in enumerate(indices):
                tsubs = block_tsubs[bi]
                if np.all(cc[tsubs] > 1):
                    to_remove.append(i)
            if to_remove:
                remove_set = set(to_remove)
                for i in to_remove:
                    cc[block_tsubs[indices[i]]] -= 1
                indices = [b for i, b in enumerate(indices) if i not in remove_set]
                changed = True
        return indices

    current_solution = reduce_cover(list(current_cover))
    best_solution = list(current_solution)
    best_size = len(best_solution)

    # ===================== PHASE 4: LOCAL SEARCH =====================
    # 1-for-0 removals and 1-for-1 swaps
    if time.time() < deadline - 10.0:
        sol_set = set(best_solution)
        cc = np.zeros(num_t, dtype=np.int32)
        for bi in best_solution:
            cc[block_tsubs[bi]] += 1
        sol_list = list(best_solution)
        cur_size = len(sol_list)

        max_iters = 5000
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            if cur_size == 0:
                break

            # Try to remove a redundant block
            # Pick random block
            i = random.randint(0, cur_size - 1)
            bi = sol_list[i]
            tsubs = block_tsubs[bi]
            critical = tsubs[cc[tsubs] == 1]

            if len(critical) == 0:
                # Remove
                sol_list.pop(i)
                sol_set.remove(bi)
                cc[tsubs] -= 1
                cur_size -= 1
                if cur_size < best_size:
                    best_solution = list(sol_list)
                    best_size = cur_size
                continue

            # Try 1-for-1 swap: find block covering all critical
            found = False
            # Sample some candidates from the critical t-subsets
            c = critical[random.randint(0, len(critical) - 1)]
            candidates = t_to_blocks[c]
            shuffled = candidates.copy()
            np.random.shuffle(shuffled)
            for cand in shuffled[:30]:
                cand = int(cand)
                if cand in sol_set:
                    continue
                cand_tsubs = block_tsubs[cand]
                if np.all(np.isin(critical, cand_tsubs)):
                    # Perform swap
                    sol_list[i] = cand
                    sol_set.remove(bi)
                    sol_set.add(cand)
                    cc[tsubs] -= 1
                    cc[cand_tsubs] += 1
                    found = True
                    if cur_size < best_size:
                        best_solution = list(sol_list)
                        best_size = cur_size
                    break

            if not found:
                # Occasionally perturb: add a random block to escape local optima
                if random.random() < 0.1 and cur_size < best_size + 5:
                    cand = random.randint(0, num_blocks - 1)
                    if cand not in sol_set:
                        sol_list.append(cand)
                        sol_set.add(cand)
                        cc[block_tsubs[cand]] += 1
                        cur_size += 1

    # Final reduction
    best_solution = reduce_cover(best_solution)

    # Convert to output
    blocks = [list(all_blocks[bi]) for bi in best_solution]
    return blocks
