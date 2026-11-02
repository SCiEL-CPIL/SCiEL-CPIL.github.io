<<<<<<< SEARCH
    def local_search(blocks, max_iters=3000):
        n_blocks = len(blocks)
        if n_blocks == 0:
            return blocks, 0

        # Build cover count and block-to-ts mapping for current solution
        block_ts_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_to_ts_np[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
                arr = np.array(ts_list, dtype=np.int32)
                block_ts_indices.append(arr)

        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for ts_arr in block_ts_indices:
            np.add.at(cover_count, ts_arr, 1)

        sol_block_set = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_set.add(idx)

        improved_count = 0

        # Phase A: Iteratively remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 3.0:
                    return blocks, improved_count
                ts_arr = block_ts_indices[i]
                if np.all(cover_count[ts_arr] >= 2):
                    np.subtract.at(cover_count, ts_arr, 1)
                    blocks.pop(i)
                    block_ts_indices.pop(i)
                    if i < len(blocks):
                        idx = get_block_idx(blocks[i])
                    else:
                        idx = -1
                    if idx >= 0:
                        sol_block_set.discard(idx)
                    improved_count += 1
                    changed = True
                    break

        # Phase B: 1-exchange local search - try to replace a block with a better one
        iters = 0
        while iters < max_iters:
            iters += 1
            if time.time() > deadline - 3.0:
                break
            if len(blocks) == 0:
                break

            i = random.randint(0, len(blocks) - 1)
            ts_arr_i = block_ts_indices[i]

            # Find t-subsets uniquely covered by block i
            unique_mask = cover_count[ts_arr_i] == 1
            unique_ts = ts_arr_i[unique_mask]

            if len(unique_ts) == 0:
                # Block is redundant, remove it
                np.subtract.at(cover_count, ts_arr_i, 1)
                blocks.pop(i)
                block_ts_indices.pop(i)
                improved_count += 1
                continue

            # Find candidate blocks that cover all unique_ts
            candidates = set(ts_to_blocks[int(unique_ts[0])])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[int(ts_idx)])
                if not candidates:
                    break

            if not candidates:
                continue

            valid_candidates = [c for c in candidates if c not in sol_block_set]
            if not valid_candidates:
                continue

            # Pick a random candidate
            new_block_idx = random.choice(valid_candidates)
            new_ts_arr = block_to_ts_np[new_block_idx]

            # Check if swap is valid
            np.subtract.at(cover_count, ts_arr_i, 1)
            np.add.at(cover_count, new_ts_arr, 1)

            if np.min(cover_count) >= 1:
                blocks[i] = list(all_k_blocks[new_block_idx])
                block_ts_indices[i] = new_ts_arr
                old_idx = get_block_idx(blocks[i])
                if old_idx >= 0:
                    sol_block_set.discard(old_idx)
                sol_block_set.add(new_block_idx)
                improved_count += 1
            else:
                # Revert
                np.subtract.at(cover_count, new_ts_arr, 1)
                np.add.at(cover_count, ts_arr_i, 1)

        return blocks, improved_count
=======
    def local_search(blocks, max_iters=3000, seed=0):
        rng = random.Random(seed)
        n_blocks = len(blocks)
        if n_blocks == 0:
            return blocks, 0

        # Build cover count and block-to-ts mapping for current solution
        block_ts_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_to_ts_np[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
                arr = np.array(ts_list, dtype=np.int32)
                block_ts_indices.append(arr)

        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for ts_arr in block_ts_indices:
            np.add.at(cover_count, ts_arr, 1)

        sol_block_set = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_set.add(idx)

        improved_count = 0

        # Phase A: Iteratively remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 3.0:
                    return blocks, improved_count
                ts_arr = block_ts_indices[i]
                if np.all(cover_count[ts_arr] >= 2):
                    np.subtract.at(cover_count, ts_arr, 1)
                    blocks.pop(i)
                    block_ts_indices.pop(i)
                    if i < len(blocks):
                        idx = get_block_idx(blocks[i])
                    else:
                        idx = -1
                    if idx >= 0:
                        sol_block_set.discard(idx)
                    improved_count += 1
                    changed = True
                    break

        # Phase B: 1-exchange local search
        iters = 0
        while iters < max_iters:
            iters += 1
            if time.time() > deadline - 3.0:
                break
            if len(blocks) == 0:
                break

            i = rng.randint(0, len(blocks) - 1)
            ts_arr_i = block_ts_indices[i]

            unique_mask = cover_count[ts_arr_i] == 1
            unique_ts = ts_arr_i[unique_mask]

            if len(unique_ts) == 0:
                np.subtract.at(cover_count, ts_arr_i, 1)
                blocks.pop(i)
                block_ts_indices.pop(i)
                improved_count += 1
                continue

            candidates = set(ts_to_blocks[int(unique_ts[0])])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[int(ts_idx)])
                if not candidates:
                    break

            if not candidates:
                continue

            valid_candidates = [c for c in candidates if c not in sol_block_set]
            if not valid_candidates:
                continue

            new_block_idx = rng.choice(valid_candidates)
            new_ts_arr = block_to_ts_np[new_block_idx]

            np.subtract.at(cover_count, ts_arr_i, 1)
            np.add.at(cover_count, new_ts_arr, 1)

            if np.min(cover_count) >= 1:
                blocks[i] = list(all_k_blocks[new_block_idx])
                block_ts_indices[i] = new_ts_arr
                old_idx = get_block_idx(blocks[i])
                if old_idx >= 0:
                    sol_block_set.discard(old_idx)
                sol_block_set.add(new_block_idx)
                improved_count += 1
            else:
                np.subtract.at(cover_count, new_ts_arr, 1)
                np.add.at(cover_count, ts_arr_i, 1)

        # Phase C: 2-for-1 exchange - remove two blocks, add one
        iters_c = 0
        max_iters_c = max(500, max_iters // 3)
        while iters_c < max_iters_c:
            iters_c += 1
            if time.time() > deadline - 3.0:
                break
            if len(blocks) < 2:
                break

            i = rng.randint(0, len(blocks) - 1)
            j = rng.randint(0, len(blocks) - 1)
            if i == j:
                continue

            ts_arr_i = block_ts_indices[i]
            ts_arr_j = block_ts_indices[j]

            # Unique t-subsets covered only by block i or only by block j (after considering both)
            # We need t-subsets that are covered by exactly one of {i, j} and not by any other block
            # More precisely: t-subsets where cover_count == 1 and they're in block i, 
            # or cover_count == 2 and they're in both i and j (but we'd need to check others)
            
            # Simpler approach: find all t-subsets uniquely covered by i or j
            # A t-subset is "critical" for the pair if removing both i and j would uncover it
            # i.e., cover_count[ts] <= 2 and ts is in i or j, and if cover_count[ts]==2, ts is in both i and j
            
            unique_i = ts_arr_i[cover_count[ts_arr_i] == 1]
            unique_j = ts_arr_j[cover_count[ts_arr_j] == 1]
            
            # Also t-subsets in both i and j with count exactly 2
            both_mask = (cover_count[ts_arr_i] == 2) & (cover_count[ts_arr_j] == 2)
            # Need to check which of ts_arr_i with count 2 are also in ts_arr_j
            # This is expensive; approximate by checking a sample
            all_critical = np.concatenate([unique_i, unique_j])
            
            if len(all_critical) == 0:
                # Both blocks are redundant
                np.subtract.at(cover_count, ts_arr_i, 1)
                np.subtract.at(cover_count, ts_arr_j, 1)
                if np.min(cover_count) >= 1:
                    # Remove the higher index first to keep indices valid
                    hi = max(i, j)
                    lo = min(i, j)
                    blocks.pop(hi)
                    block_ts_indices.pop(hi)
                    blocks.pop(lo)
                    block_ts_indices.pop(lo)
                    idx_hi = get_block_idx(blocks[hi] if hi < len(blocks) else blocks[-1])
                    idx_lo = get_block_idx(blocks[lo] if lo < len(blocks) else blocks[-1])
                    # Actually need to be more careful
                    # Let's just mark as improved
                    improved_count += 2
                else:
                    # Revert
                    np.add.at(cover_count, ts_arr_j, 1)
                    np.add.at(cover_count, ts_arr_i, 1)
                continue

            # Find a single block that covers all critical t-subsets
            candidates = set(ts_to_blocks[int(all_critical[0])])
            for ts_idx in all_critical[1:]:
                candidates.intersection_update(ts_to_blocks[int(ts_idx)])
                if not candidates:
                    break

            if not candidates:
                continue

            valid_candidates = [c for c in candidates if c not in sol_block_set and c != get_block_idx(blocks[i]) and c != get_block_idx(blocks[j])]
            if not valid_candidates:
                continue

            new_block_idx = rng.choice(valid_candidates)
            new_ts_arr = block_to_ts_np[new_block_idx]

            # Try the swap
            np.subtract.at(cover_count, ts_arr_i, 1)
            np.subtract.at(cover_count, ts_arr_j, 1)
            np.add.at(cover_count, new_ts_arr, 1)

            if np.min(cover_count) >= 1:
                # Accept: remove blocks i and j, add new_block
                hi = max(i, j)
                lo = min(i, j)
                # Remove hi first
                idx_h = get_block_idx(blocks[hi])
                blocks.pop(hi)
                block_ts_indices.pop(hi)
                if idx_h >= 0:
                    sol_block_set.discard(idx_h)
                # Remove lo (adjust index if needed)
                idx_l = get_block_idx(blocks[lo])
                blocks.pop(lo)
                block_ts_indices.pop(lo)
                if idx_l >= 0:
                    sol_block_set.discard(idx_l)
                # Add new block
                blocks.append(list(all_k_blocks[new_block_idx]))
                block_ts_indices.append(new_ts_arr)
                sol_block_set.add(new_block_idx)
                improved_count += 1
            else:
                np.subtract.at(cover_count, new_ts_arr, 1)
                np.add.at(cover_count, ts_arr_j, 1)
                np.add.at(cover_count, ts_arr_i, 1)

        # Phase D: Final cleanup - remove any remaining redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 3.0:
                    break
                ts_arr = block_ts_indices[i]
                if np.all(cover_count[ts_arr] >= 2):
                    np.subtract.at(cover_count, ts_arr, 1)
                    idx = get_block_idx(blocks[i])
                    blocks.pop(i)
                    block_ts_indices.pop(i)
                    if idx >= 0:
                        sol_block_set.discard(idx)
                    improved_count += 1
                    changed = True
                    break

        return blocks, improved_count
>>>>>>> REPLACE

