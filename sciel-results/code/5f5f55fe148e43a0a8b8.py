<<<<<<< SEARCH
def _local_search(blocks, coverage, rng, time_budget):
    """Local search: removal, swap, and 2-swap passes."""
    blocks = list(blocks)
    n_ks, n_t = coverage.shape

    def compute_union(exclude_indices=None):
        if not blocks:
            return np.zeros(n_t, dtype=np.uint8)
        union = np.zeros(n_t, dtype=np.uint8)
        for idx, b in enumerate(blocks):
            if exclude_indices is not None and idx in exclude_indices:
                continue
            union |= coverage[b]
        return union

    improved = True
    while improved and _elapsed() < time_budget:
        improved = False

        # --- Removal pass ---
        while _elapsed() < time_budget:
            removed_any = False
            i = 0
            while i < len(blocks) and _elapsed() < time_budget:
                union = compute_union(exclude_indices={i})
                if union.all():
                    blocks.pop(i)
                    removed_any = True
                    improved = True
                    continue
                i += 1
            if not removed_any:
                break

        # --- Swap pass ---
        n_swaps = 30
        for _ in range(n_swaps):
            if _elapsed() >= time_budget:
                break
            i = rng.randrange(len(blocks))
            j = rng.randrange(n_ks)
            if j in blocks:
                continue
            old = blocks[i]
            blocks[i] = j
            # Try to remove any block
            removed = False
            for idx in range(len(blocks)):
                union = compute_union(exclude_indices={idx})
                if union.all():
                    blocks.pop(idx)
                    removed = True
                    improved = True
                    break
            if not removed:
                blocks[i] = old
            if removed:
                # Additional removal pass
                while _elapsed() < time_budget:
                    removed_any = False
                    ii = 0
                    while ii < len(blocks) and _elapsed() < time_budget:
                        union = compute_union(exclude_indices={ii})
                        if union.all():
                            blocks.pop(ii)
                            removed_any = True
                            improved = True
                            continue
                        ii += 1
                    if not removed_any:
                        break

        # --- 2-Swap pass: replace two blocks with two new blocks if it reduces count ---
        if len(blocks) >= 2 and _elapsed() < time_budget:
            n_2swaps = 10
            for _ in range(n_2swaps):
                if _elapsed() >= time_budget:
                    break
                i1 = rng.randrange(len(blocks))
                i2 = rng.randrange(len(blocks))
                if i1 == i2:
                    continue
                j1 = rng.randrange(n_ks)
                j2 = rng.randrange(n_ks)
                if j1 in blocks or j2 in blocks or j1 == j2:
                    continue
                old1, old2 = blocks[i1], blocks[i2]
                blocks[i1] = j1
                blocks[i2] = j2
                # Try to remove any two blocks
                removed_count = 0
                for idx in range(len(blocks)):
                    if removed_count >= 2:
                        break
                    union = compute_union(exclude_indices={idx})
                    if union.all():
                        blocks.pop(idx)
                        removed_count += 1
                        improved = True
                if removed_count < 2:
                    # Revert
                    blocks[i1] = old1
                    blocks[i2] = old2
                    # Need to restore any removed blocks - this is complex, skip for now
                    # Actually we can't easily revert, so only accept if removed_count == 2
                    # But we already popped... Let's just not do 2-swap to keep it simple
                    pass

        # --- Repair if needed ---
        if not _is_fully_covered(blocks, coverage):
            uncovered = np.ones(n_t, dtype=np.uint8)
            for b in blocks:
                uncovered[coverage[b] > 0] = 0
            gains = coverage.dot(uncovered)
            if gains.max() > 0:
                blocks.append(int(gains.argmax()))

    return blocks
=======
def _local_search(blocks, coverage, rng, time_budget):
    """Local search with incremental coverage counts for fast removal/swap checks."""
    blocks = list(blocks)
    n_ks, n_t = coverage.shape
    used = set(blocks)

    # Build coverage count: cov[t] = number of blocks covering t-subset t
    cov = np.zeros(n_t, dtype=np.int32)
    for b in blocks:
        cov += coverage[b]

    def try_remove(idx):
        """Try to remove block at index idx. Returns True if removed."""
        b = blocks[idx]
        new_cov = cov - coverage[b]
        if (new_cov > 0).all():
            cov -= coverage[b]
            blocks.pop(idx)
            used.discard(b)
            return True
        return False

    improved = True
    while improved and _elapsed() < time_budget:
        improved = False

        # --- Removal pass (fast with incremental cov) ---
        while _elapsed() < time_budget:
            removed_any = False
            i = 0
            while i < len(blocks) and _elapsed() < time_budget:
                if try_remove(i):
                    removed_any = True
                    improved = True
                    continue
                i += 1
            if not removed_any:
                break

        # --- Swap pass: replace block i with candidate j, then try removals ---
        n_swaps = 50
        for _ in range(n_swaps):
            if _elapsed() >= time_budget:
                break
            if len(blocks) == 0:
                break
            i = rng.randrange(len(blocks))
            j = rng.randrange(n_ks)
            if j in used:
                continue
            old = blocks[i]
            # Tentatively swap
            cov -= coverage[old]
            cov += coverage[j]
            blocks[i] = j
            used.discard(old)
            used.add(j)

            # Try to remove any block (preferentially the one we just added or others)
            removed = False
            # First try removing the newly added block (if redundant)
            for idx in range(len(blocks)):
                b_idx = blocks[idx]
                new_cov = cov - coverage[b_idx]
                if (new_cov > 0).all():
                    cov -= coverage[b_idx]
                    blocks.pop(idx)
                    used.discard(b_idx)
                    removed = True
                    improved = True
                    break
            if not removed:
                # Revert
                cov -= coverage[j]
                cov += coverage[old]
                blocks[i] = old
                used.discard(j)
                used.add(old)
            else:
                # Additional removal pass
                while _elapsed() < time_budget:
                    removed_any = False
                    ii = 0
                    while ii < len(blocks) and _elapsed() < time_budget:
                        if try_remove(ii):
                            removed_any = True
                            improved = True
                            continue
                        ii += 1
                    if not removed_any:
                        break

        # --- Frequency-guided swap: target low-contribution blocks for replacement ---
        if len(blocks) > 1 and _elapsed() < time_budget:
            n_freq_swaps = 20
            for _ in range(n_freq_swaps):
                if _elapsed() >= time_budget:
                    break
                # Compute marginal contribution of each block
                # marginal[i] = number of t-subsets ONLY covered by block i
                marginals = []
                for i, b in enumerate(blocks):
                    only_cov = (cov == 1) & (coverage[b] > 0)
                    marginals.append(int(only_cov.sum()))
                # Sort by marginal contribution (ascending) - target weakest blocks
                order = sorted(range(len(blocks)), key=lambda x: marginals[x])
                # Pick a weak block to replace
                i = order[rng.randrange(min(3, len(order)))]
                j = rng.randrange(n_ks)
                if j in used:
                    continue
                old = blocks[i]
                cov -= coverage[old]
                cov += coverage[j]
                blocks[i] = j
                used.discard(old)
                used.add(j)

                # Try removals
                removed = False
                for idx in range(len(blocks)):
                    b_idx = blocks[idx]
                    new_cov = cov - coverage[b_idx]
                    if (new_cov > 0).all():
                        cov -= coverage[b_idx]
                        blocks.pop(idx)
                        used.discard(b_idx)
                        removed = True
                        improved = True
                        break
                if not removed:
                    cov -= coverage[j]
                    cov += coverage[old]
                    blocks[i] = old
                    used.discard(j)
                    used.add(old)
                else:
                    while _elapsed() < time_budget:
                        removed_any = False
                        ii = 0
                        while ii < len(blocks) and _elapsed() < time_budget:
                            if try_remove(ii):
                                removed_any = True
                                improved = True
                                continue
                            ii += 1
                        if not removed_any:
                            break

        # --- Repair if needed ---
        if not (cov > 0).all():
            uncovered = (cov == 0)
            gains = coverage.dot(uncovered.astype(np.uint8))
            if gains.max() > 0:
                best_i = int(gains.argmax())
                blocks.append(best_i)
                used.add(best_i)
                cov += coverage[best_i]

    return blocks
>>>>>>> REPLACE

