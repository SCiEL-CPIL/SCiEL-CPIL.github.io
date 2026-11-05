import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    # Use 140s to be safe
    deadline = start_time + 140.0

    # 1. Precomputation
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    n_subsets = len(all_subsets)

    all_blocks = list(itertools.combinations(range(v), k))
    # Map block tuple to index for quick lookup if needed
    n_blocks = len(all_blocks)

    # Precompute block coverage: list of subset indices for each block
    block_cover = []
    for b in all_blocks:
        cover = [subset_to_idx[s] for s in itertools.combinations(b, t)]
        block_cover.append(cover)

    # Precompute which blocks contain which subset
    # blocks_containing[s_idx] = list of block indices
    blocks_containing = [[] for _ in range(n_subsets)]
    for i, b in enumerate(all_blocks):
        for s_idx in block_cover[i]:
            blocks_containing[s_idx].append(i)
    
    # Precompute degree of each subset (number of blocks containing it)
    # Used for "hardest subset" heuristic
    subset_degree = [len(lst) for lst in blocks_containing]

    # 2. Greedy Initialization
    # We maintain:
    # uncovered: list of booleans
    # cover_count: list of ints
    # active_blocks: set of block indices
    
    uncovered = [1] * n_subsets
    cover_count = [0] * n_subsets
    active_blocks = set()
    
    # Heuristic: Pick uncovered subset with minimum degree (hardest to cover)
    # To make this fast, we can just pick random initially, or scan.
    # Scanning 3876 items is fast.
    
    while any(uncovered):
        if time.time() > deadline:
            break
        
        # Find hardest uncovered subset
        min_deg = float('inf')
        target_subset_idx = -1
        for i in range(n_subsets):
            if uncovered[i]:
                if subset_degree[i] < min_deg:
                    min_deg = subset_degree[i]
                    target_subset_idx = i
                    # Can't break early as we want min, but maybe we can sample?
                    # For 3876, full scan is < 1ms.
        
        if target_subset_idx == -1:
            break # Should not happen if any(uncovered)

        # Find best block containing target_subset_idx
        best_block_idx = -1
        best_gain = -1
        candidates = blocks_containing[target_subset_idx]
        
        for b_idx in candidates:
            gain = 0
            for s_idx in block_cover[b_idx]:
                if uncovered[s_idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
        
        if best_block_idx == -1:
            break # No block found? Should not happen.

        # Add block
        active_blocks.add(best_block_idx)
        for s_idx in block_cover[best_block_idx]:
            uncovered[s_idx] = 0
            cover_count[s_idx] += 1

    # Initial solution
    best_size = len(active_blocks)
    best_blocks_set = set(active_blocks)
    
    # 3. Local Search / Reduction
    # Goal: Minimize number of blocks while maintaining coverage.
    # Move 1: Remove block if redundant.
    # Move 2: Swap block if it allows future removals (or just to explore).
    
    # Helper: Check if block b_idx is redundant (can be removed)
    def is_redundant(b_idx):
        for s_idx in block_cover[b_idx]:
            if cover_count[s_idx] == 1:
                return False
        return True

    # Helper: Find a valid replacement for b_idx
    # Valid if all subsets covered ONLY by b_idx are also covered by new_block
    def find_replacement(b_idx):
        # Identify subsets covered only by b_idx
        unique_subsets = [s_idx for s_idx in block_cover[b_idx] if cover_count[s_idx] == 1]
        if not unique_subsets:
            return -1 # It's redundant, handled by removal logic
        
        # Find intersection of blocks containing all unique_subsets
        # Start with the one with smallest list?
        # Just use the first one
        candidates = set(blocks_containing[unique_subsets[0]])
        
        for s_idx in unique_subsets[1:]:
            candidates.intersection_update(blocks_containing[s_idx])
            if not candidates:
                return -1
        
        # Exclude current block
        candidates.discard(b_idx)
        
        if not candidates:
            return -1
        
        # Pick randomly to explore
        return random.choice(list(candidates))

    # Main Loop
    stuck_counter = 0
    max_stuck = 1000 # Reset stuck counter or break if too many failed attempts
    
    while time.time() < deadline:
        # Phase 1: Remove redundant blocks
        removed_any = False
        for b_idx in list(active_blocks):
            if is_redundant(b_idx):
                # Remove
                for s_idx in block_cover[b_idx]:
                    cover_count[s_idx] -= 1
                active_blocks.remove(b_idx)
                removed_any = True
                
                if len(active_blocks) < best_size:
                    best_size = len(active_blocks)
                    best_blocks_set = set(active_blocks)
        
        if removed_any:
            stuck_counter = 0
            continue

        # Phase 2: Try swaps
        # Pick a random block to try replacing
        if not active_blocks:
            break
            
        b_idx = random.choice(list(active_blocks))
        new_b_idx = find_replacement(b_idx)
        
        if new_b_idx != -1:
            # Perform swap
            # Decrement old
            for s_idx in block_cover[b_idx]:
                cover_count[s_idx] -= 1
            # Increment new
            for s_idx in block_cover[new_b_idx]:
                cover_count[s_idx] += 1
            
            active_blocks.remove(b_idx)
            active_blocks.add(new_b_idx)
            
            stuck_counter = 0
        else:
            stuck_counter += 1
            if stuck_counter > max_stuck:
                # Try to restart or break?
                # Breaking might be better to save time, but we might be stuck in a bad local min.
                # We can try to "shake" by doing a random valid swap if available?
                # We just tried random.
                # Let's try iterating through all blocks to find ANY valid swap.
                found_swap = False
                for b_idx_try in active_blocks:
                    new_b_idx_try = find_replacement(b_idx_try)
                    if new_b_idx_try != -1:
                        # Perform swap
                        for s_idx in block_cover[b_idx_try]:
                            cover_count[s_idx] -= 1
                        for s_idx in block_cover[new_b_idx_try]:
                            cover_count[s_idx] += 1
                        active_blocks.remove(b_idx_try)
                        active_blocks.add(new_b_idx_try)
                        found_swap = True
                        stuck_counter = 0
                        break
                
                if not found_swap:
                    # Completely stuck. No single swap maintains validity.
                    # We might need a multi-swap, but that's complex.
                    # Break to avoid wasting time.
                    break

    # Convert set of indices back to list of blocks
    result_blocks = [all_blocks[i] for i in best_blocks_set]
    return result_blocks
