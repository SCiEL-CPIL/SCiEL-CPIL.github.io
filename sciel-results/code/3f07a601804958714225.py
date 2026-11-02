import itertools
import random
import time
import math

def construct(v, k, t):
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = []
    block_t_indices = []
    
    for block in itertools.combinations(range(v), k):
        all_k_subsets.append(block)
        t_indices = [t_subset_to_index[ts] for ts in itertools.combinations(block, t)]
        block_t_indices.append(t_indices)
        
    num_k_subsets = len(all_k_subsets)
    
    # For faster repair, we might want to know, for each t-subset, which blocks cover it.
    # But storing this is large. Let's rely on scanning or a smaller index.
    # Given v=19, num_t_subsets = C(19,4) = 3876.
    # num_k_subsets = C(19,7) = 50388.
    # We can build a list of lists: t_coverers[i] = list of block indices covering t_subset i.
    t_coverers = [[] for _ in range(num_t_subsets)]
    for b_idx, t_indices in enumerate(block_t_indices):
        for t_idx in t_indices:
            t_coverers[t_idx].append(b_idx)

    def greedy_init(seed):
        rng = random.Random(seed)
        uncovered = set(range(num_t_subsets))
        current_blocks = []
        
        while uncovered:
            best_block_idx = -1
            max_covered = 0
            # To add randomness, we can pick from top K
            # For simplicity, standard greedy first, maybe add noise later
            for i in range(num_k_subsets):
                count = 0
                for idx in block_t_indices[i]:
                    if idx in uncovered:
                        count += 1
                if count > max_covered:
                    max_covered = count
                    best_block_idx = i
            
            if best_block_idx == -1:
                break # Should not happen
                
            current_blocks.append(best_block_idx)
            
            for idx in block_t_indices[best_block_idx]:
                uncovered.discard(idx)
                
        return current_blocks

    def local_search(blocks, rng):
        # blocks: list of block indices
        # Returns improved blocks list
        
        current_blocks = blocks[:]
        
        # Build coverage counts
        coverage_counts = [0] * num_t_subsets
        for b_idx in current_blocks:
            for t_idx in block_t_indices[b_idx]:
                coverage_counts[t_idx] += 1
        
        improved = True
        while improved:
            improved = False
            
            # Pass 1: Remove redundant blocks
            # A block is redundant if all its t-subsets are covered by >= 1 other block
            # i.e., coverage_counts[t_idx] > 1 for all t_idx in block
            new_blocks = []
            for b_idx in current_blocks:
                is_redundant = True
                for t_idx in block_t_indices[b_idx]:
                    if coverage_counts[t_idx] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove it
                    for t_idx in block_t_indices[b_idx]:
                        coverage_counts[t_idx] -= 1
                    improved = True
                else:
                    new_blocks.append(b_idx)
            current_blocks = new_blocks
            
            # Pass 2: Randomized Remove & Repair
            # Try a few random removals
            for _ in range(50): # Limit iterations per pass
                if not current_blocks:
                    break
                b_idx_pos = rng.randint(0, len(current_blocks) - 1)
                b_idx_old = current_blocks[b_idx_pos]
                
                # Remove
                for t_idx in block_t_indices[b_idx_old]:
                    coverage_counts[t_idx] -= 1
                
                # Find uncovered
                uncovered = [i for i in range(num_t_subsets) if coverage_counts[i] == 0]
                
                if not uncovered:
                    # Successfully removed
                    current_blocks.pop(b_idx_pos)
                    improved = True
                    break # Restart passes
                else:
                    # Try to repair
                    # Option A: Find 1 block covering all uncovered
                    # Option B: Greedily add blocks
                    
                    # Check if 1 block can cover all
                    # The intersection of t_coverers for all uncovered must be non-empty
                    # Or simpler: iterate blocks? Too slow.
                    # Let's try greedy repair
                    old_size = len(current_blocks)
                    repair_blocks = []
                    temp_uncovered = set(uncovered)
                    
                    while temp_uncovered:
                        best_add_idx = -1
                        max_cover = 0
                        # Sample candidates to speed up?
                        # Candidates must cover at least one in temp_uncovered
                        # We can pick a random uncovered t_idx and look at its coverers
                        sample_t_idx = next(iter(temp_uncovered))
                        candidates = t_coverers[sample_t_idx]
                        
                        for c_idx in candidates:
                            if c_idx in current_blocks: # Already have it? No, we removed one, maybe it's in there?
                                # If c_idx is in current_blocks, it's already counted.
                                # We are adding NEW blocks.
                                pass
                            count = 0
                            for t_idx in block_t_indices[c_idx]:
                                if t_idx in temp_uncovered:
                                    count += 1
                            if count > max_cover:
                                max_cover = count
                                best_add_idx = c_idx
                                
                        if best_add_idx == -1:
                            break # No repair possible? Should not happen.
                            
                        repair_blocks.append(best_add_idx)
                        for t_idx in block_t_indices[best_add_idx]:
                            if t_idx in temp_uncovered:
                                temp_uncovered.discard(t_idx)
                        
                    new_size = len(current_blocks) + len(repair_blocks)
                    
                    if new_size < old_size:
                        # Accept repair
                        for t_idx in block_t_indices[b_idx_old]:
                            # We already decremented. 
                            # We need to add back the repair blocks' coverage.
                            pass # Logic below handles final state
                        # Actually, we need to update coverage_counts for the repair blocks
                        # And remove b_idx_old from list
                        current_blocks.pop(b_idx_pos)
                        for r_idx in repair_blocks:
                            current_blocks.append(r_idx)
                            for t_idx in block_t_indices[r_idx]:
                                coverage_counts[t_idx] += 1
                        improved = True
                        break # Restart passes
                    else:
                        # Revert removal
                        for t_idx in block_t_indices[b_idx_old]:
                            coverage_counts[t_idx] += 1
                        # current_blocks unchanged (we didn't pop yet in this branch logic? 
                        # Wait, I popped in the success branch. In fail branch, I didn't pop.
                        # But I decremented coverage_counts. I need to increment back.
                        # Correct.

        return current_blocks

    best_blocks = None
    best_size = float('inf')
    
    start_time = time.time()
    time_limit = 110.0
    seed = 0
    
    while (time.time() - start_time) < time_limit:
        # Greedy Init
        init_blocks = greedy_init(seed)
        
        # Local Search
        rng = random.Random(seed)
        ls_blocks = local_search(init_blocks, rng)
        
        if len(ls_blocks) < best_size:
            best_size = len(ls_blocks)
            best_blocks = ls_blocks[:]
            
        seed += 1
        
    # Convert to final format
    final_blocks = [list(all_k_subsets[i]) for i in best_blocks]
    return final_blocks
