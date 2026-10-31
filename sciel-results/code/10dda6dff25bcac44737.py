<<<<<<< SEARCH
def _coverage_of_blocks(blocks, coverage):
    """Union of coverage sets of the given block indices."""
    cov = set()
    for b in blocks:
        cov |= coverage[b]
    return cov


def _is_covered(blocks, coverage, n_t):
    cov = set()
    for b in blocks:
        cov |= coverage[b]
    return len(cov) == n_t


def _local_search(blocks, coverage, n_t, k_subsets, rng, time_budget):
    """Removal + swap + add local search to minimize |blocks| while staying a cover."""
    blocks = list(blocks)
    n = len(blocks)

    def union_cov(exclude=None):
        cov = set()
        for idx, b in enumerate(blocks):
            if exclude is not None and idx == exclude:
                continue
            cov |= coverage[b]
        return cov

    improved = True
    while improved and _elapsed() < time_budget:
        improved = False
        # --- Removal pass ---
        i = 0
        while i < len(blocks) and _elapsed() < time_budget:
            b = blocks[i]
            # Check if removing b still covers everything
            cov = set()
            for idx, bb in enumerate(blocks):
                if idx == i:
                    continue
                cov |= coverage[bb]
            if len(cov) == n_t:
                blocks.pop(i)
                improved = True
                continue
            i += 1
        # --- Swap pass: replace block i with some j not in blocks if it improves (same size, but reduces redundancy) ---
        # Actually swap keeps size; we want to reduce size via removal. Swap helps find better removals.
        # Do a few random swap attempts to escape local minima, then removal again.
        for _ in range(30):
            if _elapsed() >= time_budget:
                break
            i = rng.randrange(len(blocks))
            j = rng.randrange(len(k_subsets))
            if j in blocks:
                continue
            # Try swapping: does the new set still cover? If yes, keep (same size) — only keep if it enables later removals.
            # We'll tentatively swap and see if removal then possible.
            old = blocks[i]
            blocks[i] = j
            # quick removal check
            removed = False
            for idx, bb in enumerate(blocks):
                cov = set()
                for idx2, bb2 in enumerate(blocks):
                    if idx2 == idx:
                        continue
                    cov |= coverage[bb2]
                if len(cov) == n_t:
                    blocks.pop(idx)
                    removed = True
                    break
            if removed:
                improved = True
            else:
                blocks[i] = old  # revert
        # --- Add pass: if not covered (shouldn't happen), add best block ---
        if len(blocks) > 0:
            cov = set()
            for b in blocks:
                cov |= coverage[b]
            if len(cov) < n_t:
                # add the block covering most missing
                missing = set(range(n_t)) - cov
                best_i = -1
                best_g = 0
                for idx, c in enumerate(coverage):
                    g = len(c & missing)
                    if g > best_g:
                        best_g = g
                        best_i = idx
                if best_i >= 0:
                    blocks.append(best_i)
    return blocks
=======
def _union_coverage(blocks, coverage):
    """Union of coverage bitmasks of the given block indices."""
    if not blocks:
        return 0
    result = 0
    for b in blocks:
        result |= coverage[b]
    return result


def _is_covered(blocks, coverage, n_t):
    cov = _union_coverage(blocks, coverage)
    return _popcount(cov) == n_t


def _local_search(blocks, coverage, n_t, k_subsets, rng, time_budget):
    """Removal + swap + add local search to minimize |blocks| while staying a cover."""
    blocks = list(blocks)
    n_k = len(k_subsets)
    
    def union_cov(exclude_idx=None):
        cov = 0
        for idx, b in enumerate(blocks):
            if exclude_idx is not None and idx == exclude_idx:
                continue
            cov |= coverage[b]
        return cov

    improved = True
    while improved and _elapsed() < time_budget:
        improved = False
        # --- Removal pass ---
        i = 0
        while i < len(blocks) and _elapsed() < time_budget:
            cov_without = union_cov(exclude_idx=i)
            if _popcount(cov_without) == n_t:
                blocks.pop(i)
                improved = True
                continue
            i += 1
        
        # --- Swap pass: try random swaps that enable removals ---
        for _ in range(50):
            if _elapsed() >= time_budget:
                break
            i = rng.randrange(len(blocks))
            j = rng.randrange(n_k)
            if j in blocks:
                continue
            old = blocks[i]
            blocks[i] = j
            # Check if any block can now be removed
            removed = False
            for idx in range(len(blocks)):
                cov_without = union_cov(exclude_idx=idx)
                if _popcount(cov_without) == n_t:
                    blocks.pop(idx)
                    removed = True
                    break
            if removed:
                improved = True
            else:
                blocks[i] = old  # revert
        
        # --- Add pass: repair if not covered ---
        cov = _union_coverage(blocks, coverage)
        if _popcount(cov) < n_t:
            missing = ((1 << n_t) - 1) & ~cov
            best_i = -1
            best_g = 0
            for idx, c in enumerate(coverage):
                g = _popcount(c & missing)
                if g > best_g:
                    best_g = g
                    best_i = idx
            if best_i >= 0:
                blocks.append(best_i)
    return blocks
>>>>>>> REPLACE

