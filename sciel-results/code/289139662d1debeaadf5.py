import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 120s

    # Precomputation
    # 1. Enumerate all t-subsets and assign indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)

    # 2. Enumerate all k-subsets and compute their coverage bitmask
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)

    # masks[i] is an integer bitmask where bit j is 1 if k_subset i covers t_subset j
    masks = [0] * num_k_subsets
    # Also store the list of t-subset indices covered by each k-subset for faster count updates
    mask_indices = [[] for _ in range(num_k_subsets)]
    for i, ks in enumerate(k_subsets):
        mask = 0
        indices = []
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
            indices.append(idx)
        masks[i] = mask
        mask_indices[i] = indices

    best_blocks = None
    best_count = float('inf')

    # Precompute global count of how many k-subsets cover each t-subset
    # This is constant for the problem instance.
    global_counts = [0] * num_t_subsets
    for i in range(num_k_subsets):
        for idx in mask_indices[i]:
            global_counts[idx] += 1

    # Run multiple random restarts
    num_restarts = 50000  # We will limit by time

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        uncovered = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        block_indices = []

        while uncovered:
            if time.time() > deadline:
                break

            max_gain = 0
            candidates = []

            # Calculate gain for all unused blocks
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                gain = (masks[i] & uncovered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    candidates = [i]
                elif gain == max_gain and gain > 0:
                    candidates.append(i)

            if max_gain == 0:
                break

            # Tie-breaking: Pick the candidate that covers the "rarest" uncovered t-subsets.
            # Rarity is approximated by global_counts (fewer total options = rarer).
            # We want to cover t-subsets with low global_counts.
            # Score = sum of (1.0 / global_counts[idx]) for idx in newly covered.
            
            if len(candidates) > 1:
                best_candidate = None
                best_score = -1
                for i in candidates:
                    score = 0
                    # Only consider newly covered bits
                    new_covered = masks[i] & uncovered
                    # Iterate set bits
                    m = new_covered
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        score += 1.0 / global_counts[idx]
                        m ^= lsb
                    if score > best_score:
                        best_score = score
                        best_candidate = i
                chosen_idx = best_candidate
            else:
                chosen_idx = candidates[0]

            current_blocks.append(k_subsets[chosen_idx])
            block_indices.append(chosen_idx)
            used_blocks.add(chosen_idx)
            uncovered &= ~masks[chosen_idx]

        if uncovered != 0:
            continue

        # Local Search with Swaps
        local_indices = list(block_indices)
        
        # Initialize coverage counts
        counts = [0] * num_t_subsets
        for bi in local_indices:
            for idx in mask_indices[bi]:
                counts[idx] += 1

        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Remove redundant blocks
            for pos in range(len(local_indices)):
                if time.time() > deadline:
                    break
                bi = local_indices[pos]
                is_redundant = True
                for idx in mask_indices[bi]:
                    if counts[idx] < 2:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove bi
                    for idx in mask_indices[bi]:
                        counts[idx] -= 1
                    local_indices.pop(pos)
                    improved = True
                    break # Restart loop as indices changed

            if improved:
                continue

            # Phase 2: 1-for-1 Swaps
            n_blocks = len(local_indices)
            for pos in range(n_blocks):
                if time.time() > deadline:
                    break
                bi = local_indices[pos]
                
                # Simulate removing bi
                uncovered_by_removal = []
                for idx in mask_indices[bi]:
                    if counts[idx] == 1:
                        uncovered_by_removal.append(idx)
                
                if not uncovered_by_removal:
                    # This block is redundant, remove it
                    for idx in mask_indices[bi]:
                        counts[idx] -= 1
                    local_indices.pop(pos)
                    improved = True
                    break

                # We need a block B that covers ALL of uncovered_by_removal
                req_mask = 0
                for idx in uncovered_by_removal:
                    req_mask |= (1 << idx)
                
                found_swap = False
                for bj in range(num_k_subsets):
                    if bj in local_indices:
                        continue
                    # Check if bj covers all required indices
                    if (masks[bj] & req_mask) == req_mask:
                        # Perform swap
                        for idx in mask_indices[bi]:
                            counts[idx] -= 1
                        for idx in mask_indices[bj]:
                            counts[idx] += 1
                        local_indices[pos] = bj
                        found_swap = True
                        improved = True
                        break
                
                if found_swap:
                    break # Restart loop

        if len(local_indices) < best_count:
            best_count = len(local_indices)
            best_blocks = [list(k_subsets[bi]) for bi in local_indices]

    if best_blocks is None:
        # Fallback to simple greedy if randomization failed (unlikely)
        uncovered = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        while uncovered:
            max_gain = 0
            best_block_idx = -1
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                gain = (masks[i] & uncovered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_block_idx = i
            if max_gain == 0:
                break
            current_blocks.append(k_subsets[best_block_idx])
            used_blocks.add(best_block_idx)
            uncovered &= ~masks[best_block_idx]
        best_blocks = [list(b) for b in current_blocks]

    return best_blocks
