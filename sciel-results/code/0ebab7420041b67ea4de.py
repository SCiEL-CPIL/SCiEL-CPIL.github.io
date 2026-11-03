import itertools
import random
import time


def _build_block_data(v, k, t):
    """Precompute all blocks and their triple masks."""
    all_triples = list(itertools.combinations(range(v), t))
    n_triples = len(all_triples)
    triple_index = {tr: i for i, tr in enumerate(all_triples)}
    triple_bit = {tr: (1 << i) for i, tr in enumerate(all_triples)}
    
    block_list = []
    block_masks = []
    block_triples = []
    
    for block in itertools.combinations(range(v), k):
        mask = 0
        trs = []
        for sub in itertools.combinations(block, t):
            b = triple_bit[sub]
            mask |= b
            trs.append(triple_index[sub])
        block_list.append(block)
        block_masks.append(mask)
        block_triples.append(trs)
    
    full_mask = (1 << n_triples) - 1
    return block_list, block_masks, block_triples, n_triples, triple_index, full_mask


def _greedy_cover(block_list, block_masks, block_triples, n_triples, full_mask, rng=None, randomize=False):
    """Greedy set cover. Returns list of block indices."""
    uncovered = full_mask
    selected = []
    used = set()
    
    while uncovered:
        best_gain = 0
        best_candidates = []
        for i in range(len(block_list)):
            if i in used:
                continue
            gain = (uncovered & block_masks[i]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_candidates = [i]
            elif gain == best_gain and gain > 0:
                best_candidates.append(i)
        
        if not best_candidates:
            break
        
        if randomize and rng and len(best_candidates) > 1:
            # Pick from top candidates with some randomness
            chosen = rng.choice(best_candidates)
        else:
            chosen = best_candidates[0]
        
        selected.append(chosen)
        used.add(chosen)
        uncovered &= ~block_masks[chosen]
    
    return selected


def _compute_coverage(selected, block_triples, n_triples):
    """Compute coverage count for each triple."""
    counts = [0] * n_triples
    for idx in selected:
        for tr_idx in block_triples[idx]:
            counts[tr_idx] += 1
    return counts


def _local_search(selected, block_list, block_masks, block_triples, n_triples, full_mask, rng, time_limit):
    """Local search: try removals and swaps to reduce block count."""
    selected = list(selected)
    n_blocks_total = len(block_list)
    start_time = time.time()
    
    # Compute initial coverage
    counts = _compute_coverage(selected, block_triples, n_triples)
    selected_set = set(selected)
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        
        # Try removals: find a block that can be removed
        for pos in range(len(selected)):
            if time.time() > time_limit:
                break
            idx = selected[pos]
            # Check if all triples in this block are covered by others
            can_remove = True
            for tr_idx in block_triples[idx]:
                if counts[tr_idx] <= 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove it
                for tr_idx in block_triples[idx]:
                    counts[tr_idx] -= 1
                selected.pop(pos)
                selected_set.discard(idx)
                improved = True
                break  # Restart scan after removal
        
        if not improved:
            # Try swaps: replace one block with another
            for pos in range(len(selected)):
                if time.time() > time_limit:
                    break
                old_idx = selected[pos]
                # Find a block not in selected that covers all "essential" triples of old_idx
                # and doesn't add too many new requirements
                old_triples = block_triples[old_idx]
                
                # A triple is "essential" for old_idx if counts[tr] == 1 (only old_idx covers it)
                # If any triple has count == 1, we need the replacement to cover it too.
                essential_triples = []
                for tr_idx in old_triples:
                    if counts[tr_idx] == 1:
                        essential_triples.append(tr_idx)
                
                if not essential_triples:
                    # All triples in old_idx are covered by others, so we can remove it
                    # (This should have been caught above, but just in case)
                    for tr_idx in old_triples:
                        counts[tr_idx] -= 1
                    selected.pop(pos)
                    selected_set.discard(old_idx)
                    improved = True
                    break
                
                # Try to find a replacement
                best_new_idx = -1
                best_new_gain = -1  # We want to maximize the number of triples covered by new block that are currently uncovered by old block's removal
                
                for new_idx in range(n_blocks_total):
                    if new_idx in selected_set:
                        continue
                    # Check if new block covers all essential triples
                    new_triples_set = set(block_triples[new_idx])
                    all_essential_covered = all(tr in new_triples_set for tr in essential_triples)
                    if not all_essential_covered:
                        continue
                    
                    # This is a valid swap candidate
                    # Compute the "gain": how many triples would be covered by new block
                    # that are NOT covered by old block (net new coverage)
                    # Actually, we just need to verify the swap maintains coverage.
                    # Since essential triples are covered, and other triples in old_idx have count >= 2,
                    # removing old_idx is safe for them.
                    # The new block adds coverage for its triples.
                    # We want to prefer blocks that cover more currently-undercovered triples?
                    # But all triples are covered (we're maintaining a cover).
                    # So any valid swap maintains coverage.
                    # We can just take the first valid one, or randomize.
                    
                    best_new_idx = new_idx
                    break  # Take first valid for speed, or randomize
                
                if best_new_idx != -1:
                    # Perform swap
                    for tr_idx in block_triples[old_idx]:
                        counts[tr_idx] -= 1
                    for tr_idx in block_triples[best_new_idx]:
                        counts[tr_idx] += 1
                    selected[pos] = best_new_idx
                    selected_set.discard(old_idx)
                    selected_set.add(best_new_idx)
                    improved = True
                    break
            
            if improved:
                break
        
        # If no improvement in a full pass, try randomized swaps
        if not improved and time.time() < time_limit:
            # Try random swaps for a few iterations
            for _ in range(50):
                if time.time() > time_limit:
                    break
                pos = rng.randrange(len(selected))
                old_idx = selected[pos]
                old_triples = block_triples[old_idx]
                
                essential_triples = []
                for tr_idx in old_triples:
                    if counts[tr_idx] == 1:
                        essential_triples.append(tr_idx)
                
                if not essential_triples:
                    for tr_idx in old_triples:
                        counts[tr_idx] -= 1
                    selected.pop(pos)
                    selected_set.discard(old_idx)
                    improved = True
                    break
                
                # Try random candidates
                tried = 0
                while tried < 20:
                    new_idx = rng.randrange(n_blocks_total)
                    if new_idx in selected_set:
                        tried += 1
                        continue
                    new_triples_set = set(block_triples[new_idx])
                    if all(tr in new_triples_set for tr in essential_triples):
                        # Valid swap
                        for tr_idx in block_triples[old_idx]:
                            counts[tr_idx] -= 1
                        for tr_idx in block_triples[new_idx]:
                            counts[tr_idx] += 1
                        selected[pos] = new_idx
                        selected_set.discard(old_idx)
                        selected_set.add(new_idx)
                        improved = True
                        break
                    tried += 1
                
                if improved:
                    break
    
    return selected


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute
    block_list, block_masks, block_triples, n_triples, triple_index, full_mask = _build_block_data(v, k, t)
    n_blocks_total = len(block_list)
    
    best_blocks = None
    best_count = float('inf')
    
    rng = random.Random(42)
    
    # Phase 1: Greedy covers with different randomizations
    n_greedy_restarts = 10
    for r in range(n_greedy_restarts):
        if time.time() > deadline - 30.0:
            break
        
        randomize = (r > 0)
        selected = _greedy_cover(block_list, block_masks, block_triples, n_triples, full_mask, rng, randomize)
        
        if len(selected) < best_count:
            best_count = len(selected)
            best_blocks = [block_list[i] for i in selected]
        
        # Phase 2: Local search on this cover
        if time.time() < deadline - 10.0:
            local_time_limit = deadline - 10.0
            improved_selected = _local_search(selected, block_list, block_masks, block_triples, n_triples, full_mask, rng, local_time_limit)
            
            if len(improved_selected) < best_count:
                best_count = len(improved_selected)
                best_blocks = [block_list[i] for i in improved_selected]
        
        # Update rng seed for diversity
        rng.seed(42 + r * 1000)
    
    # Phase 3: If time remains, do more local search on the best solution
    if best_blocks and time.time() < deadline - 5.0:
        # Reconstruct indices from best_blocks
        block_to_idx = {tuple(b): i for i, b in enumerate(block_list)}
        best_indices = [block_to_idx[tuple(b)] for b in best_blocks]
        
        local_time_limit = deadline - 5.0
        improved = _local_search(best_indices, block_list, block_masks, block_triples, n_triples, full_mask, rng, local_time_limit)
        
        if len(improved) < best_count:
            best_count = len(improved)
            best_blocks = [block_list[i] for i in improved]
    
    if best_blocks:
        return [list(b) for b in best_blocks]
    
    # Fallback
    return []
