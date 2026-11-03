"""Covering design C(v,k,t) via bitmask greedy + aggressive local search."""
import itertools
import time


def construct(v, k, t):
    start = time.time()
    deadline = start + 110.0  # Safety margin under 120s limit

    def _deadline_ok():
        return time.time() < deadline

    # 1. Precomputation
    all_blocks = [tuple(c) for c in itertools.combinations(range(v), k)]
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    sub_index = {s: i for i, s in enumerate(all_t_subsets)}
    num_subsets = len(all_t_subsets)
    
    # Precompute for each block: list of t-subset indices it covers
    # This is used for counting coverage in local search
    block_subsets = []
    for i in range(num_blocks):
        block = all_blocks[i]
        subs = [sub_index[ts] for ts in itertools.combinations(block, t)]
        block_subsets.append(subs)
    
    # Also precompute the bitmask for each block for fast greedy
    block_masks = [0] * num_blocks
    for i in range(num_blocks):
        mask = 0
        for s_idx in block_subsets[i]:
            mask |= (1 << s_idx)
        block_masks[i] = mask

    all_mask = (1 << num_subsets) - 1

    # 2. Greedy Set Cover (with indices)
    covered = 0
    selected_indices = []
    selected_set = set()
    
    while covered != all_mask:
        if not _deadline_ok():
            break
        
        best_block_idx = -1
        best_gain = 0
        
        # Find the block that covers the most *uncovered* subsets
        # Optimization: skip blocks that are already selected? 
        # No, we don't select duplicates. But we can skip blocks that have 0 gain.
        # The main cost is the bit_count.
        
        for i in range(num_blocks):
            if i in selected_set:
                continue
            gain = (block_masks[i] & (all_mask ^ covered)).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
        
        if best_block_idx == -1 or best_gain == 0:
            break
            
        selected_indices.append(best_block_idx)
        selected_set.add(best_block_idx)
        covered |= block_masks[best_block_idx]

    if not _deadline_ok():
        return [list(all_blocks[i]) for i in selected_indices]

    # 3. Local Search: Deletion and Swap
    # State: selected_set, subset_counts
    
    # Initialize counts
    subset_counts = [0] * num_subsets
    for idx in selected_set:
        for s_idx in block_subsets[idx]:
            subset_counts[s_idx] += 1

    # Helper: check if a block is redundant
    # A block is redundant if all its subsets have count > 1
    def _is_redundant(block_idx):
        for s_idx in block_subsets[block_idx]:
            if subset_counts[s_idx] == 1:
                return False
        return True

    # Helper: try to find a redundant block
    def _find_redundant():
        for idx in list(selected_set):
            if _is_redundant(idx):
                return idx
        return -1

    # Helper: try a swap
    # Pick a block b, find its "critical" subsets (count == 1)
    # Find a new block b' that covers all critical subsets
    # If found, swap b -> b'. This maintains coverage.
    # The hope is that b' is less "redundant" or allows future deletions.
    # Actually, just swapping doesn't reduce the number of blocks.
    # We need a swap that leads to a deletion.
    
    # Let's try a "1-for-1" swap that might enable a deletion.
    # Or, simpler: just try to find a block that covers the same subsets 
    # as a redundant block but is "better" in some way?
    # No, if it's redundant, we just delete it.
    
    # The power of local search in set cover usually comes from:
    # 1. Deletion (if possible)
    # 2. Swap (replace one block with another that covers the unique parts)
    #    If we replace b with b', and b' covers some subsets that were ONLY covered by b,
    #    AND b' covers some subsets that were covered by multiple blocks,
    #    we haven't changed the count.
    
    # However, if we can find a sequence of swaps that allows a deletion, that's good.
    # Let's stick to a simple heuristic:
    # 1. Delete redundant blocks.
    # 2. If no deletions, try a random swap to "shuffle" the solution, hoping to break symmetry/local minima.
    
    improved = True
    while improved and _deadline_ok():
        improved = False
        
        # Phase 1: Deletion
        while _deadline_ok():
            redundant_idx = _find_redundant()
            if redundant_idx == -1:
                break
            # Remove it
            selected_set.remove(redundant_idx)
            for s_idx in block_subsets[redundant_idx]:
                subset_counts[s_idx] -= 1
            improved = True
        
        if not _deadline_ok():
            break
        
        # Phase 2: Swap
        # We try a few random swaps.
        # Pick a random block b from selected_set.
        # Find its "critical" subsets: C = {s in subsets(b) | count(s) == 1}
        # If C is empty, b is redundant (should have been caught).
        # Find candidates: blocks b' such that C is a subset of subsets(b').
        # If candidates exist, pick one (e.g., one that maximizes coverage of "common" subsets?)
        # And swap.
        
        # To be efficient, we can't iterate all blocks for every swap attempt.
        # We limit the number of swap attempts per "epoch".
        
        for _ in range(20):  # Limit swap attempts
            if not _deadline_ok():
                break
            
            if len(selected_set) == 0:
                break
                
            # Pick a random block
            b_idx = next(iter(selected_set))
            
            # Find critical subsets
            critical = []
            for s_idx in block_subsets[b_idx]:
                if subset_counts[s_idx] == 1:
                    critical.append(s_idx)
            
            if not critical:
                # Should have been deleted
                continue
            
            # Find a candidate block that covers all critical subsets
            # This is expensive. We can precompute for each subset, which blocks cover it?
            # sub_to_blocks is huge.
            # Instead, let's just iterate through all blocks? No, too slow.
            
            # Alternative:
            # We can just try to find a block that covers the *intersection* of critical subsets?
            # No, it must cover *each* critical subset.
            
            # Let's restrict: only try swap if we have time and maybe only for a few random blocks.
            # Actually, a better swap heuristic:
            # Try to replace b with a block b' that covers the "most" of b's subsets,
            # but specifically must cover the critical ones.
            
            # Let's try a simple random walk:
            # Pick a random subset s from critical.
            # Pick a random block b' that covers s.
            # Check if b' covers all critical subsets.
            
            s_rand = critical[0]
            
            # Which blocks cover s_rand?
            # We can iterate all blocks and check if s_rand is in block_subsets[i].
            # This is O(N_blocks * t).
            # With 75k blocks, this is ~75k * 50 = 3.7M operations. Might be ok for a few tries.
            
            found_swap = False
            for i in range(num_blocks):
                if i == b_idx or i in selected_set:
                    continue
                
                # Check if i covers s_rand
                # Efficient check: is s_rand in block_subsets[i]?
                # block_subsets[i] is a list.
                # We can precompute a set for each block? Or use bitmask?
                # Using bitmask: (block_masks[i] >> s_rand) & 1
                if not (block_masks[i] >> s_rand) & 1:
                    continue
                
                # Check if i covers ALL critical subsets
                valid = True
                for c in critical:
                    if not (block_masks[i] >> c) & 1:
                        valid = False
                        break
                
                if valid:
                    # We found a swap!
                    # Remove b_idx, add i
                    selected_set.remove(b_idx)
                    for s in block_subsets[b_idx]:
                        subset_counts[s] -= 1
                    
                    selected_set.add(i)
                    for s in block_subsets[i]:
                        subset_counts[s] += 1
                    
                    improved = True
                    found_swap = True
                    break
            
            if found_swap:
                # A swap happened. It might enable a deletion.
                # The outer while loop will re-check for deletions.
                break

    # Final verification and return
    # Ensure all subsets are covered
    final_covered = 0
    for idx in selected_set:
        final_covered |= block_masks[idx]
    
    if final_covered != all_mask:
        # Fallback to greedy result
        return [list(all_blocks[i]) for i in selected_indices]
    
    return [list(all_blocks[i]) for i in selected_set]
