import itertools
import numpy as np
import random
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a Covering Density heuristic.
    
    Strategy:
    1. Initialize with a dense pool of blocks derived from cyclic structures.
    2. Use greedy set cover to get an initial covering.
    3. Apply fast vectorized local search:
       - Aggressively remove redundant blocks.
       - Swap blocks to enable further removals (minimize block count).
    """
    
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Precompute t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Generate candidate blocks using cyclic shifts of base blocks
    random.seed(42)
    
    # Generate diverse base blocks
    # Heuristic: Base blocks should be spread out.
    # Using cyclic shifts of random samples is a good standard approach.
    # We will generate a larger pool to have more options for swaps.
    base_blocks = []
    
    # Add some structured blocks (e.g., consecutive, or spread)
    # Consecutive: [0,1,2,3,4,5,6]
    base_blocks.append(list(range(k)))
    # Spread: [0, 3, 6, 9, 12, 15, 18]
    step = v // k
    base_blocks.append([i * step for i in range(k)])
    
    # Random base blocks
    for _ in range(200):
        base = sorted(random.sample(range(v), k))
        base_blocks.append(base)
    
    # Generate all cyclic shifts
    candidate_blocks = []
    seen_blocks = set()
    for base in base_blocks:
        for shift in range(v):
            block = tuple(sorted((x + shift) % v for x in base))
            if block not in seen_blocks:
                seen_blocks.add(block)
                candidate_blocks.append(block)
    
    # For each candidate block, compute the t-subset indices it covers
    # Store as a list of numpy arrays for fast access
    block_t_indices = []
    for block in candidate_blocks:
        indices = []
        for t_sub in itertools.combinations(block, t):
            indices.append(t_subset_to_idx[t_sub])
        block_t_indices.append(np.array(indices, dtype=np.int32))
        
    # Map block tuple to index in candidate_blocks for quick lookup
    block_to_idx = {block: i for i, block in enumerate(candidate_blocks)}
    
    # Greedy set cover
    uncovered = np.ones(n_t_subsets, dtype=np.bool_)
    selected_block_indices = []
    available = np.ones(len(candidate_blocks), dtype=np.bool_)
    
    # Precompute a matrix or use sparse logic? 
    # For v=24, k=7, t=3: C(7,3)=35 triples per block.
    # n_t_subsets = C(24,3) = 2024.
    # candidate_blocks ~ 4500.
    # Greedy step: For each available block, count how many uncovered triples it covers.
    
    while uncovered.any():
        if time.time() - start_time > time_limit:
            break
            
        # Vectorized coverage count for uncovered triples
        # We can't easily vectorize across all blocks without a huge matrix (4500 x 2024).
        # 4500 * 2024 * 4 bytes = 36 MB. This is acceptable.
        # Let's create the incidence matrix M where M[i, j] = 1 if block i covers triple j.
        
    # Let's build the incidence matrix once.
    # This allows very fast greedy and swap evaluations.
    incidence = np.zeros((len(candidate_blocks), n_t_subsets), dtype=np.int8)
    for i, indices in enumerate(block_t_indices):
        incidence[i, indices] = 1
        
    # Greedy Set Cover
    uncovered_count = n_t_subsets
    selected = []
    while uncovered_count > 0:
        if time.time() - start_time > time_limit:
            break
            
        # Find block that covers most uncovered triples
        # uncovered is a boolean vector
        # coverage = incidence @ uncovered
        coverage = incidence @ uncovered
        best_idx = np.argmax(coverage)
        if coverage[best_idx] == 0:
            # Should not happen if pool is good enough, but fallback
            # Pick random
            break
            
        selected.append(best_idx)
        uncovered[incidence[best_idx] == 1] = False
        uncovered_count -= coverage[best_idx]
        
    # Convert to list of block tuples
    current_blocks = [candidate_blocks[i] for i in selected]
    
    # Local Search: Minimize number of blocks
    # 1. Compute coverage counts
    # We maintain coverage_counts for the current solution
    coverage_counts = np.zeros(n_t_subsets, dtype=np.int32)
    for b in current_blocks:
        idx = block_to_idx[b]
        coverage_counts[incidence[idx] == 1] += 1
        
    # Helper to check if solution is valid (all counts > 0)
    def is_valid(counts):
        return np.all(counts > 0)
        
    # Helper to remove redundant blocks
    def remove_redundant(blocks, counts):
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                idx = block_to_idx[blocks[i]]
                # Check if all triples in this block have count > 1
                if np.all(counts[incidence[idx] == 1] > 1):
                    counts[incidence[idx] == 1] -= 1
                    blocks.pop(i)
                    changed = True
                    break # Restart loop to check new redundancy
        return blocks, counts

    # Phase 1: Initial Removal
    current_blocks, coverage_counts = remove_redundant(current_blocks, coverage_counts)
    
    # Phase 2: Swap to reduce count
    # For each block, try swapping with a candidate that covers the "unique" triples
    # and then see if we can remove more blocks.
    
    # Precompute for each candidate block which triples it covers (already in incidence)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(len(current_blocks)):
            if time.time() - start_time > time_limit:
                break
            
            block_i = current_blocks[i]
            idx_i = block_to_idx[block_i]
            
            # Temporarily remove block_i
            temp_counts = coverage_counts.copy()
            temp_counts[incidence[idx_i] == 1] -= 1
            
            # Find triples that are now uncovered (count == 0)
            deficient_mask = (temp_counts == 0)
            if not np.any(deficient_mask):
                # If removing block_i doesn't break coverage, it was redundant.
                # This should have been caught by remove_redundant, but just in case.
                coverage_counts = temp_counts
                current_blocks.pop(i)
                improved = True
                break # Restart outer loop
            
            # Find candidates that cover ALL deficient triples
            # A candidate j is valid if incidence[j, deficient_mask] are all 1
            # And it must not be the same block (though swapping to same is useless)
            
            # Efficient check:
            # For each candidate, check if it covers the deficient set.
            # deficient_indices = np.where(deficient_mask)[0]
            # We can check: np.all(incidence[:, deficient_indices] == 1, axis=1)
            
            deficient_indices = np.where(deficient_mask)[0]
            
            # Mask of candidates that cover all deficient triples
            valid_candidates_mask = np.all(incidence[:, deficient_indices] == 1, axis=1)
            
            # Exclude current block index? 
            # If we swap to the same block, no change.
            # valid_candidates_mask[idx_i] = False
            
            valid_indices = np.where(valid_candidates_mask)[0]
            
            # If no valid candidate, skip
            if len(valid_indices) == 0:
                continue
            
            # Try a few random valid candidates
            # We want one that leads to a state where we can remove *more* blocks.
            # Simple heuristic: Pick a random valid candidate, swap, and try to remove redundants.
            # If the final count is lower, keep it.
            
            # To speed up, we can try the first few, or randomize.
            # Let's try up to 5 random valid candidates
            trials = min(5, len(valid_indices))
            chosen_indices = random.sample(list(valid_indices), trials)
            
            best_new_blocks = None
            best_new_counts = None
            best_len = len(current_blocks)
            
            for cand_idx in chosen_indices:
                new_counts = temp_counts.copy()
                new_counts[incidence[cand_idx] == 1] += 1
                
                # Check validity (should be valid by construction of deficient mask)
                # But other triples might have been affected? No, we only added a block.
                # So new_counts is valid.
                
                # Try to remove redundants from this new state
                new_blocks = current_blocks.copy()
                new_blocks[i] = candidate_blocks[cand_idx]
                
                # Optimization: Instead of full remove_redundant loop, 
                # just check if we can remove *any* block.
                # Actually, we must do the full pass to ensure we get the minimal count for this state.
                
                new_blocks, new_counts = remove_redundant(new_blocks, new_counts)
                
                if len(new_blocks) < best_len:
                    best_len = len(new_blocks)
                    best_new_blocks = new_blocks
                    best_new_counts = new_counts
                    
            if best_new_blocks is not None:
                current_blocks = best_new_blocks
                coverage_counts = best_new_counts
                improved = True
                break # Restart outer loop as indices changed
    
    # Final verification
    final_counts = np.zeros(n_t_subsets, dtype=np.int32)
    for b in current_blocks:
        idx = block_to_idx[b]
        final_counts[incidence[idx] == 1] += 1
        
    if np.any(final_counts == 0):
        # Fallback: add random blocks to fix
        uncovered_mask = (final_counts == 0)
        while np.any(uncovered_mask):
            if time.time() - start_time > time_limit:
                break
            # Find best block for uncovered
            coverage = incidence @ uncovered_mask
            best_idx = np.argmax(coverage)
            if coverage[best_idx] == 0:
                break
            current_blocks.append(candidate_blocks[best_idx])
            final_counts[incidence[best_idx] == 1] += 1
            uncovered_mask = (final_counts == 0)

    return [list(b) for b in current_blocks]
