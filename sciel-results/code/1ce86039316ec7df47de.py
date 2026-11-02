import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    def _is_valid(blocks):
        """Check if blocks cover all t-subsets."""
        if not blocks:
            return False
        covered = set()
        for block in blocks:
            for sub in itertools.combinations(block, t):
                covered.add(sub)
        return len(covered) == len(list(itertools.combinations(range(v), t)))

    # 1. Precompute all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_index_map = {s: i for i, s in enumerate(t_subsets)}
    num_t = len(t_subsets)

    # 2. Precompute all k-blocks and their coverage masks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)

    # Precompute t-subsets within each block for fast redundancy check
    block_tsubs = []
    block_masks = [0] * num_k
    for idx, block in enumerate(k_blocks):
        mask = 0
        tsubs_in_block = []
        for combo in itertools.combinations(block, t):
            i = t_index_map[combo]
            mask |= (1 << i)
            tsubs_in_block.append(i)
        block_masks[idx] = mask
        block_tsubs.append(tsubs_in_block)

    full_mask = (1 << num_t) - 1

    def _greedy_pass(seed, strategy='max_gain', hard_deadline=None):
        """One greedy set cover pass with different strategies."""
        if hard_deadline is None:
            hard_deadline = deadline
            
        if time.time() > hard_deadline:
            return None

        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        selected_set = set()
        active_indices = list(range(num_k))

        while uncovered:
            if time.time() > hard_deadline:
                return None

            best_idx = -1
            best_count = -1

            if strategy == 'max_gain':
                # Standard max gain with random tie-breaking
                best_indices = []
                # Iterate only over active indices
                for i in active_indices:
                    gain = (uncovered & block_masks[i]).bit_count()
                    if gain > best_count:
                        best_count = gain
                        best_indices = [i]
                    elif gain == best_count and gain > 0:
                        best_indices.append(i)
                if best_indices:
                    if len(best_indices) > 1:
                        best_idx = rng.choice(best_indices)
                    else:
                        best_idx = best_indices[0]
            elif strategy == 'randomized':
                # Randomly weight gains to escape local optima
                weighted_best = -1
                for i in active_indices:
                    gain = (uncovered & block_masks[i]).bit_count()
                    if gain == 0:
                        continue
                    # Add random perturbation
                    w = gain + rng.random() * 0.5
                    if w > weighted_best:
                        weighted_best = w
                        best_idx = i
            else:  # 'first_random'
                # Pick randomly from top-10 gain blocks
                gains = []
                for i in active_indices:
                    gain = (uncovered & block_masks[i]).bit_count()
                    if gain > 0:
                        gains.append((gain, i))
                if not gains:
                    break
                gains.sort(key=lambda x: -x[0])
                top_n = min(10, len(gains))
                best_idx = rng.choice(gains[:top_n])[1]
                best_count = gains[0][0]

            if best_idx == -1 or best_count <= 0:
                break

            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]
            # Remove selected index from active list for efficiency
            # active_indices.remove(best_idx) is O(N). 
            # For 75k items, this is slow. 
            # We can just leave it and check `if i in selected_set` but we removed that check.
            # Let's just filter active_indices occasionally or use a set for selected.
            # Actually, since we are iterating active_indices, we should remove it.
            # To avoid O(N) remove, we can swap with last and pop.
            # But order doesn't matter.
            try:
                idx_pos = active_indices.index(best_idx)
                active_indices[idx_pos] = active_indices[-1]
                active_indices.pop()
            except ValueError:
                pass

        return selected

    def _remove_redundant_fast(selected_indices):
        """Remove redundant blocks using t-subset counts (faster than mask OR)."""
        if not selected_indices:
            return selected_indices

        # Count how many selected blocks cover each t-subset
        tsub_count = Counter()
        for idx in selected_indices:
            for ts in block_tsubs[idx]:
                tsub_count[ts] += 1

        # Keep only blocks that cover at least one t-subset uniquely
        kept = []
        for idx in selected_indices:
            if time.time() > deadline:
                kept.append(idx)
                continue
            has_unique = False
            for ts in block_tsubs[idx]:
                if tsub_count[ts] == 1:
                    has_unique = True
                    break
            if has_unique:
                kept.append(idx)

        return kept

    def _local_search_polish(selected_indices, max_swaps=50):
        """Try to improve by swapping blocks in/out."""
        if time.time() > deadline:
            return selected_indices

        current = list(selected_indices)
        current_mask = 0
        for idx in current:
            current_mask |= block_masks[idx]

        # Verify validity
        if current_mask != full_mask:
            return current

        improved = True
        iterations = 0
        while improved and iterations < max_swaps and time.time() < deadline:
            improved = False
            iterations += 1

            # Try removing each block and see if we can add a better one
            for i in range(len(current)):
                if time.time() > deadline - 2.0:
                    return current

                # Remove block i
                test_mask = current_mask & ~block_masks[current[i]]
                # Check if still valid
                if test_mask != full_mask:
                    continue

                # Try to find a block that covers the missing parts with fewer blocks
                missing = full_mask & ~test_mask
                # Find block with max coverage of missing
                best_add = -1
                best_gain = 0
                for j in range(num_k):
                    if j in current and j != current[i]:
                        continue
                    gain = (missing & block_masks[j]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_add = j

                if best_add != -1 and best_gain > 0:
                    # Check if adding best_add covers all missing
                    new_mask = test_mask | block_masks[best_add]
                    if new_mask == full_mask:
                        # We can swap: remove i, add best_add
                        # But we need same count or less; here it's same count
                        # Only do if it helps future swaps
                        # For now, skip same-count swaps unless forced
                        pass

            # Try adding a block and removing two others (net -1)
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    if time.time() > deadline - 2.0:
                        return current

                    test_mask = current_mask & ~block_masks[current[i]] & ~block_masks[current[j]]
                    if test_mask == full_mask:
                        # Both i and j are redundant!
                        new_current = []
                        for idx in current:
                            if idx != current[i] and idx != current[j]:
                                new_current.append(idx)
                        current = new_current
                        current_mask = 0
                        for idx in current:
                            current_mask |= block_masks[idx]
                        improved = True
                        break
                if improved:
                    break

        return current

    best_blocks = None
    best_len = float('inf')

    # 1. Get a baseline valid solution quickly
    # Use a generous deadline for the first pass to ensure it completes
    first_deadline = start_time + 30.0
    selected = _greedy_pass(0, 'max_gain', hard_deadline=first_deadline)
    if selected is not None:
        selected = _remove_redundant_fast(selected)
        if _is_valid([k_blocks[i] for i in selected]):
            best_blocks = [list(k_blocks[i]) for i in selected]
            best_len = len(selected)

    # 2. Try to improve with multiple strategies
    strategies = ['max_gain', 'randomized', 'first_random']
    attempt = 1
    while time.time() < deadline - 10.0:  # Leave 10s for final check
        strategy = strategies[attempt % len(strategies)]
        seed = attempt * 7 + 42
        selected = _greedy_pass(seed, strategy)
        if selected is None:
            # If we run out of time in greedy, skip to next or break?
            # If we have a valid best_blocks, we can break.
            if best_blocks is not None:
                break
            else:
                # No valid solution yet, try again with different seed?
                # Or break and use fallback.
                break

        # Remove redundant blocks (fast method)
        selected = _remove_redundant_fast(selected)
        
        # Verify validity before polishing (polish might be buggy or slow)
        if not _is_valid([k_blocks[i] for i in selected]):
            continue

        # Local search polish
        selected_polished = _local_search_polish(selected)
        
        # Verify validity after polish
        if _is_valid([k_blocks[i] for i in selected_polished]):
            if len(selected_polished) < best_len:
                best_len = len(selected_polished)
                best_blocks = [list(k_blocks[i]) for i in selected_polished]

        # Early stop if we've found a very good solution
        if best_len <= 335:
            break

        attempt += 1

    # Fallback if no solution found
    if best_blocks is None:
        # Try one more time with no time limit (should be fast)
        selected = _greedy_pass(123, 'max_gain', hard_deadline=None)
        if selected is not None:
            selected = _remove_redundant_fast(selected)
            if _is_valid([k_blocks[i] for i in selected]):
                best_blocks = [list(k_blocks[i] for i in selected)]
        
        if best_blocks is None:
            # Last resort: Return all blocks (valid but large)
            # Or a simple valid construction
            best_blocks = [list(block) for block in k_blocks]

    return best_blocks
