import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    n_triples = len(all_t_subsets)
    triple_to_idx = {triple: i for i, triple in enumerate(all_t_subsets)}
    
    # Precompute all k-subsets
    all_k_subsets = list(itertools.combinations(all_points, k))
    n_k_subsets = len(all_k_subsets)
    
    # Precompute C(k, t)
    c_k_t = int(len(list(itertools.combinations(range(k), t))))
    
    # cover_matrix[i] contains indices of t-subsets covered by all_k_subsets[i]
    # Shape: (n_k_subsets, c_k_t)
    cover_matrix = np.zeros((n_k_subsets, c_k_t), dtype=np.int32)
    
    # Also build a reverse map: for each triple, which blocks cover it?
    # This is crucial for fast local search repairs.
    # triple_to_blocks[i] is a list of block indices that cover triple i
    # Since memory might be high, we can build it as a list of lists or arrays.
    # For v=24, k=7, t=3: C(24,3)=2024, C(24,7)=346104.
    # Each triple is in C(24-3, 7-3) = C(21,4) = 5985 blocks.
    # Total entries in triple_to_blocks: 2024 * 5985 ~ 12M integers. ~48MB. Acceptable.
    
    # Initialize list of lists
    triple_to_blocks = [[] for _ in range(n_triples)]
    
    for i, block in enumerate(all_k_subsets):
        for j, triple in enumerate(itertools.combinations(block, t)):
            idx = triple_to_idx[triple]
            cover_matrix[i, j] = idx
            triple_to_blocks[idx].append(i)
            
    # Convert to numpy arrays for faster iteration if needed, but lists are fine for random access
    # Actually, for the repair step, we need to iterate over blocks covering a specific triple.
    # Let's keep them as lists of ints.

    best_blocks = None
    best_len = float('inf')
    
    def fast_greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_triples, dtype=bool)
        blocks = []
        
        while uncovered.any():
            # Vectorized gain calculation
            # uncovered[cover_matrix] creates a boolean array of shape (n_k_subsets, c_k_t)
            # sum along axis 1 gives the count of uncovered triples covered by each block
            gains = np.sum(uncovered[cover_matrix], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            chosen_idx = candidates[rng.randint(0, len(candidates) - 1)]
            
            blocks.append(all_k_subsets[chosen_idx])
            uncovered[cover_matrix[chosen_idx]] = False
            
        return blocks

    def local_search_improve(blocks):
        """
        Attempts to reduce the number of blocks by:
        1. Removing redundant blocks.
        2. Swapping a block with another block that covers its unique triples 
           and potentially allows further reductions (or just maintains coverage).
           Specifically: Try to remove block B. If this leaves triples uncovered,
           find a block B' that covers ALL of them. If found, replace B with B'.
           If B' covers more "shared" triples, it might enable future removals.
        """
        if not blocks:
            return blocks
            
        # Map block tuple to index in all_k_subsets for fast lookup
        block_to_k_idx = {block: i for i, block in enumerate(all_k_subsets)}
        
        # Current set of blocks as a set of indices for O(1) lookup
        current_block_indices = set(block_to_k_idx[b] for b in blocks)
        
        def get_coverage_counts(bks):
            """Returns an array of counts for each triple."""
            coverage = np.zeros(n_triples, dtype=np.int32)
            for b in bks:
                idx = block_to_k_idx[b]
                coverage[cover_matrix[idx]] += 1
            return coverage

        # Initial coverage
        coverage = get_coverage_counts(blocks)
        
        improved = True
        # Limit iterations to prevent infinite loops or excessive time
        max_iterations = 20 
        iter_count = 0
        
        while improved and iter_count < max_iterations:
            improved = False
            iter_count += 1
            if time.time() > deadline - 5:
                break
                
            # 1. Try to remove redundant blocks (coverage count == 1 is NOT redundant, count == 0 is impossible if valid)
            # A block is redundant if all its triples are covered by other blocks (count > 1)
            # Actually, if we remove a block, its triples' counts drop by 1.
            # If any drop to 0, it's not redundant.
            
            to_remove = []
            for i, block in enumerate(blocks):
                idx = block_to_k_idx[block]
                # If all triples in this block have coverage > 1, it is redundant
                if np.all(coverage[cover_matrix[idx]] > 1):
                    to_remove.append(i)
            
            if to_remove:
                # Remove all redundant blocks at once
                remove_set = set(to_remove)
                new_blocks = [b for i, b in enumerate(blocks) if i not in remove_set]
                blocks = new_blocks
                current_block_indices = set(block_to_k_idx[b] for b in blocks)
                coverage = get_coverage_counts(blocks)
                improved = True
                continue # Restart loop as coverage changed significantly
                
            # 2. Try swap-based improvement
            # For each block, check if it can be replaced by another block
            # that covers its "unique" triples (those with count == 1)
            
            # Identify blocks that have unique triples
            # If a block has no unique triples, it's redundant (handled above).
            # So every remaining block has at least one unique triple.
            
            for i, block in enumerate(blocks):
                idx = block_to_k_idx[block]
                unique_triples_in_block = cover_matrix[idx][coverage[cover_matrix[idx]] == 1]
                
                if len(unique_triples_in_block) == 0:
                    continue # Should not happen if redundancy removal is working, but safe check
                
                # We need to find a block B' not in current set such that:
                # 1. B' covers all unique_triples_in_block
                # 2. Ideally, B' is "better" (covers more shared triples to allow future swaps/removals)
                #    For now, let's just find ANY valid replacement.
                
                # Strategy: Pick the "hardest" unique triple (one with fewest covering blocks)
                # and look for candidates covering it.
                
                # Find the triple in unique_triples_in_block with the smallest candidate pool
                min_candidates = float('inf')
                target_triple_idx = -1
                for t_idx in unique_triples_in_block:
                    cands = triple_to_blocks[t_idx]
                    # Filter out current blocks? No, we just need to find a candidate.
                    # If the only candidate is the block itself, we can't swap.
                    # Count how many candidates are NOT in current_block_indices?
                    # That's expensive. Let's just pick the triple with fewest total candidates.
                    if len(cands) < min_candidates:
                        min_candidates = len(cands)
                        target_triple_idx = t_idx
                        
                if target_triple_idx == -1:
                    continue
                
                candidates = triple_to_blocks[target_triple_idx]
                
                # Check candidates
                found_swap = False
                for cand_idx in candidates:
                    if cand_idx in current_block_indices:
                        continue
                    
                    # Check if cand_idx covers ALL unique_triples_in_block
                    cand_covered_triples = set(cover_matrix[cand_idx].tolist())
                    if all(t_idx in cand_covered_triples for t_idx in unique_triples_in_block):
                        # Valid swap!
                        # Perform swap
                        old_idx = idx
                        new_idx = cand_idx
                        
                        # Update coverage
                        coverage[cover_matrix[old_idx]] -= 1
                        coverage[cover_matrix[new_idx]] += 1
                        
                        # Update blocks list
                        blocks[i] = all_k_subsets[new_idx]
                        
                        # Update index set
                        current_block_indices.remove(old_idx)
                        current_block_indices.add(new_idx)
                        
                        found_swap = True
                        improved = True
                        break # Restart loop as state changed
                
                if found_swap:
                    break # Restart while loop
                    
        return blocks

    # Main loop
    restart_count = 0
    while time.time() < deadline - 10:
        blocks = fast_greedy_cover(restart_count)
        
        # Apply local search
        if time.time() < deadline - 5:
            blocks = local_search_improve(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
            
        restart_count += 1
        
        # If we are doing well, we might want to spend more time on local search?
        # The current structure restarts greedy. This is good for diversity.
        
    return [list(b) for b in best_blocks] if best_blocks is not None else []
