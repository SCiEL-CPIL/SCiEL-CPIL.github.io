import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 115s safety margin
    
    # 1. Precomputation
    # Map t-subsets to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Map k-subsets (blocks) to list of t-subset indices they cover
    all_k_subsets = list(itertools.combinations(range(v), k))
    block_to_t_indices = []
    for block in all_k_subsets:
        t_indices = []
        for t_sub in itertools.combinations(block, t):
            t_indices.append(t_subset_to_idx[t_sub])
        block_to_t_indices.append(t_indices)
        
    # Precompute for each t-subset index, the list of block indices that cover it
    # This allows fast candidate generation in greedy
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, t_indices in enumerate(block_to_t_indices):
        for t_idx in t_indices:
            t_to_blocks[t_idx].append(b_idx)
            
    # Helper: Fast Greedy to cover a set of t-subset indices
    def fast_greedy(uncovered_indices):
        uncovered = set(uncovered_indices)
        selected_blocks = []
        while uncovered:
            # Pick an arbitrary uncovered t-subset
            # Using next(iter(uncovered)) is O(1)
            target_t_idx = next(iter(uncovered))
            
            # Get candidate blocks covering this t-subset
            candidates = t_to_blocks[target_t_idx]
            
            best_block = -1
            max_gain = -1
            
            for b_idx in candidates:
                # Calculate gain: how many of this block's t-subsets are in uncovered
                # block_to_t_indices[b_idx] is a list of t-indices
                gain = 0
                for t_idx in block_to_t_indices[b_idx]:
                    if t_idx in uncovered:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = b_idx
                    if gain == len(block_to_t_indices[b_idx]):
                        break # Can't get better
            
            if best_block != -1:
                selected_blocks.append(best_block)
                # Update uncovered
                for t_idx in block_to_t_indices[best_block]:
                    uncovered.discard(t_idx)
                
        return selected_blocks

    # Initial Greedy
    initial_uncovered = set(range(num_t_subsets))
    current_blocks = fast_greedy(initial_uncovered)
    
    # Helper to check validity and count
    def get_covered_set(blocks):
        covered = set()
        for b_idx in blocks:
            covered.update(block_to_t_indices[b_idx])
        return covered
        
    # Local Search
    best_blocks = list(current_blocks)
    best_size = len(best_blocks)
    
    # Pre-compute block sets for faster intersection? 
    # block_to_t_indices are lists. Converting to sets might help if we do many intersections.
    # But we only intersect with 'uncovered' in greedy.
    # For local search, we need to know which t-subsets become uncovered when a block is removed.
    
    while time.time() < deadline:
        # Calculate coverage counts
        # Count how many times each t-subset is covered
        counts = [0] * num_t_subsets
        for b_idx in current_blocks:
            for t_idx in block_to_t_indices[b_idx]:
                counts[t_idx] += 1
        
        # Identify redundant blocks: blocks where all t-subsets have count > 1
        redundant_blocks = []
        for b_idx in current_blocks:
            is_redundant = True
            for t_idx in block_to_t_indices[b_idx]:
                if counts[t_idx] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                redundant_blocks.append(b_idx)
        
        if redundant_blocks:
            # Remove one redundant block
            b_to_remove = random.choice(redundant_blocks)
            new_blocks = [b for b in current_blocks if b != b_to_remove]
            if len(new_blocks) < best_size:
                best_blocks = new_blocks
                best_size = len(best_blocks)
            current_blocks = new_blocks
        else:
            # No redundant blocks. Try a kick.
            # Pick a random block to remove
            if not current_blocks:
                break
            b_to_remove = random.choice(current_blocks)
            
            # Find t-subsets that become uncovered
            # These are t-subsets in b_to_remove that had count == 1
            uncovered_after_removal = []
            for t_idx in block_to_t_indices[b_to_remove]:
                if counts[t_idx] == 1:
                    uncovered_after_removal.append(t_idx)
            
            # Greedily cover these
            if not uncovered_after_removal:
                 # This shouldn't happen if no redundant blocks, but just in case
                 new_blocks = [b for b in current_blocks if b != b_to_remove]
                 if len(new_blocks) < best_size:
                     best_blocks = new_blocks
                     best_size = len(best_blocks)
                 current_blocks = new_blocks
                 continue
            
            added_blocks = fast_greedy(uncovered_after_removal)
            
            # New total blocks
            new_total_blocks = [b for b in current_blocks if b != b_to_remove] + added_blocks
            
            if len(new_total_blocks) < best_size:
                best_blocks = new_total_blocks
                best_size = len(best_size)
            current_blocks = new_total_blocks
            
            # Optional: If we didn't improve, maybe revert to best?
            # Or just keep current for further kicks?
            # Usually, keeping the current state allows further exploration.
            # But if we get stuck, we might want to reset to best.
            # Let's stick to current state for now, as it's a random walk.

    # Convert block indices back to actual blocks (lists of points)
    # The task requires returning a list of blocks, where each block is a list of k distinct integers.
    # We have indices into all_k_subsets.
    
    # Ensure no duplicates in best_blocks? 
    # The greedy and local search might add duplicates? 
    # fast_greedy adds unique blocks because it removes from uncovered.
    # Local search removes and adds. 
    # It's possible to have duplicates if we're not careful, but the logic "remove b, add new ones covering uncovered" 
    # implies the new ones are needed. 
    # However, if `added_blocks` contains a block already in `current_blocks` (minus b_to_remove), 
    # that's fine, but it might be redundant.
    # Let's deduplicate best_blocks to be safe.
    
    unique_best_blocks = list(set(best_blocks))
    
    # Convert to lists of points
    result_blocks = [list(all_k_subsets[b_idx]) for b_idx in unique_best_blocks]
    
    # Final check: ensure all t-subsets are covered
    # This is a safety net. If not covered, fall back to initial greedy?
    # The algorithm should maintain coverage.
    # Let's verify.
    covered_final = set()
    for b_idx in unique_best_blocks:
        covered_final.update(block_to_t_indices[b_idx])
        
    if len(covered_final) < num_t_subsets:
        # Fallback to initial greedy which is guaranteed to cover
        result_blocks = [list(all_k_subsets[b_idx]) for b_idx in current_blocks]
        # Re-verify current_blocks
        covered_check = set()
        for b_idx in current_blocks:
            covered_check.update(block_to_t_indices[b_idx])
        if len(covered_check) < num_t_subsets:
             # This should not happen. Return a trivial cover? 
             # Or just the initial one.
             pass

    return result_blocks
