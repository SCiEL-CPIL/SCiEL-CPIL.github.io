<<<<<<< SEARCH
def _local_search(blocks, coverage, n_t, k_subsets, rng, time_budget):
    """Removal + swap local search with Counter-based coverage tracking."""
    blocks = list(blocks)
    
    # Build coverage counter
    cov_count = Counter()
    for b in blocks:
        for idx in coverage[b]:
            cov_count[idx] += 1
    
    def is_valid():
        return all(cov_count[idx] > 0 for idx in range(n_t))
    
    def is_redundant(b):
        for idx in coverage[b]:
            if cov_count[idx] == 1:
                return False
        return True
    
    def remove_block(b):
        blocks.remove(b)
        for idx in coverage[b]:
            cov_count[idx] -= 1
    
    def add_block(b):
        blocks.append(b)
        for idx in coverage[b]:
            cov_count[idx] += 1
    
    improved = True
    while improved and _elapsed() < time_budget:
        improved = False
        
        # --- Removal pass ---
        i = 0
        while i < len(blocks) and _elapsed() < time_budget:
            b = blocks[i]
            if is_redundant(b):
                remove_block(b)
                improved = True
                continue
            i += 1
        
        # --- Swap pass: try to replace blocks with better ones ---
        for _ in range(50):
            if _elapsed() >= time_budget:
                break
            i = rng.randrange(len(blocks))
            old_b = blocks[i]
            
            # Find critical t-subsets for old_b
            critical = []
            for idx in coverage[old_b]:
                if cov_count[idx] == 1:
                    critical.append(idx)
            
            if not critical:
                # old_b is redundant, should have been removed
                remove_block(old_b)
                improved = True
                continue
            
            # Find candidates that cover all critical t-subsets
            candidates = None
            for idx in critical:
                # Find all blocks covering this t-subset
                covering_blocks = [j for j in range(len(k_subsets)) if idx in coverage[j]]
                if candidates is None:
                    candidates = set(covering_blocks)
                else:
                    candidates.intersection_update(covering_blocks)
            
            if candidates:
                candidates -= set(blocks)
                if candidates:
                    # Pick a random candidate
                    new_b = rng.choice(list(candidates))
                    # Tentatively swap
                    blocks[i] = new_b
                    # Update coverage
                    for idx in coverage[old_b]:
                        cov_count[idx] -= 1
                    for idx in coverage[new_b]:
                        cov_count[idx] += 1
                    
                    # Check if valid and if removal is now possible
                    if is_valid():
                        # Try to remove redundant blocks
                        removed_any = False
                        for j in range(len(blocks)):
                            if is_redundant(blocks[j]):
                                remove_block(blocks[j])
                                removed_any = True
                        if removed_any:
                            improved = True
                        else:
                            # Keep the swap even if no removal (might help later)
                            pass
                    else:
                        # Revert
                        blocks[i] = old_b
                        for idx in coverage[new_b]:
                            cov_count[idx] -= 1
                        for idx in coverage[old_b]:
                            cov_count[idx] += 1
        
        # --- Add pass: if not covered, add best block ---
        if not is_valid():
            missing = [idx for idx in range(n_t) if cov_count[idx] == 0]
            if missing:
                best_i = -1
                best_g = 0
                for idx, c in enumerate(coverage):
                    g = len(c & set(missing))
                    if g > best_g:
                        best_g = g
                        best_i = idx
                if best_i >= 0:
                    add_block(best_i)
    
    return blocks
=======
def _local_search(blocks, coverage, n_t, k_subsets, reverse, rng, time_budget):
    """Removal + swap local search with Counter-based coverage tracking and reverse index."""
    blocks = list(blocks)
    n_k = len(k_subsets)
    
    # Build coverage counter
    cov_count = Counter()
    for b in blocks:
        for idx in coverage[b]:
            cov_count[idx] += 1
    
    def is_valid():
        return all(cov_count[idx] > 0 for idx in range(n_t))
    
    def is_redundant(b):
        for idx in coverage[b]:
            if cov_count[idx] == 1:
                return False
        return True
    
    def remove_block(b):
        blocks.remove(b)
        for idx in coverage[b]:
            cov_count[idx] -= 1
    
    def add_block(b):
        blocks.append(b)
        for idx in coverage[b]:
            cov_count[idx] += 1
    
    improved = True
    while improved and _elapsed() < time_budget:
        improved = False
        
        # --- Removal pass ---
        i = 0
        while i < len(blocks) and _elapsed() < time_budget:
            b = blocks[i]
            if is_redundant(b):
                remove_block(b)
                improved = True
                continue
            i += 1
        
        # --- Swap pass: try to replace blocks with better ones ---
        # Number of swap attempts scales with time remaining
        remaining = max(0.0, time_budget - _elapsed())
        n_swaps = min(200, max(10, int(remaining * 40)))
        
        for _ in range(n_swaps):
            if _elapsed() >= time_budget:
                break
            i = rng.randrange(len(blocks))
            old_b = blocks[i]
            
            # Find critical t-subsets for old_b
            critical = []
            for idx in coverage[old_b]:
                if cov_count[idx] == 1:
                    critical.append(idx)
            
            if not critical:
                # old_b is redundant, should have been removed
                remove_block(old_b)
                improved = True
                continue
            
            # Find candidates that cover all critical t-subsets using reverse index
            candidates = reverse[critical[0]]
            for idx in critical[1:]:
                candidates = candidates & reverse[idx]
            
            if candidates:
                candidates -= set(blocks)
                if candidates:
                    # Pick a random candidate
                    new_b = rng.choice(tuple(candidates))
                    # Tentatively swap
                    blocks[i] = new_b
                    # Update coverage
                    for idx in coverage[old_b]:
                        cov_count[idx] -= 1
                    for idx in coverage[new_b]:
                        cov_count[idx] += 1
                    
                    # Check if valid and if removal is now possible
                    if is_valid():
                        # Try to remove redundant blocks
                        removed_any = False
                        for j in range(len(blocks)):
                            if is_redundant(blocks[j]):
                                remove_block(blocks[j])
                                removed_any = True
                        if removed_any:
                            improved = True
                    else:
                        # Revert
                        blocks[i] = old_b
                        for idx in coverage[new_b]:
                            cov_count[idx] -= 1
                        for idx in coverage[old_b]:
                            cov_count[idx] += 1
        
        # --- Add pass: if not covered, add best block ---
        if not is_valid():
            missing = [idx for idx in range(n_t) if cov_count[idx] == 0]
            if missing:
                missing_set = set(missing)
                best_i = -1
                best_g = 0
                for idx, c in enumerate(coverage):
                    g = len(c & missing_set)
                    if g > best_g:
                        best_g = g
                        best_i = idx
                if best_i >= 0:
                    add_block(best_i)
    
    return blocks
>>>>>>> REPLACE

