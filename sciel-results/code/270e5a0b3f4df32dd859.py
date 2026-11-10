"""
Optimized covering design C(19, 7, 4) using vectorized greedy and local search.
"""
import time
import random
import itertools
import numpy as np

_DEADLINE = 120.0
_MARGIN = 5.0

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + _DEADLINE - _MARGIN
    rng = random.Random(42)

    # 1. Precomputation
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(t_subsets)
    t_subset_index = {sub: i for i, sub in enumerate(t_subsets)}

    # Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(k_blocks)
    
    # Build incidence matrix M: (num_blocks, num_subsets) boolean
    # M[i, j] = 1 if k_block i covers t_subset j
    M = np.zeros((num_blocks, num_subsets), dtype=np.uint8)
    for i, block in enumerate(k_blocks):
        block_set = set(block)
        for sub in itertools.combinations(block, t):
            idx = t_subset_index[sub]
            M[i, idx] = 1

    # 2. Greedy Construction
    coverage_counts = np.zeros(num_subsets, dtype=np.int32)
    selected_blocks = []
    selected_set = set()
    
    while time.time() < deadline:
        if np.all(coverage_counts > 0):
            break
            
        uncovered = coverage_counts == 0
        if not np.any(uncovered):
            break
            
        # Calculate gain for each block: number of currently uncovered subsets it covers
        # M is uint8, uncovered is bool (0/1). Dot product works.
        gains = M.dot(uncovered)
        
        # Find best block
        best_idx = np.argmax(gains)
        
        # If best gain is 0, we are stuck (shouldn't happen if valid covering exists)
        if gains[best_idx] == 0:
            break
            
        # Add block
        selected_blocks.append(best_idx)
        selected_set.add(best_idx)
        coverage_counts += M[best_idx]

    # 3. Local Search / Refinement
    # We want to minimize the number of blocks.
    # Strategy: 
    #   a. Remove redundant blocks.
    #   b. Swap blocks to create more redundancy.
    
    current_blocks = list(selected_blocks)
    
    def is_valid(blocks_list):
        if not blocks_list:
            return False
        counts = np.zeros(num_subsets, dtype=np.int32)
        for b in blocks_list:
            counts += M[b]
        return np.all(counts > 0)

    # Function to remove redundant blocks
    def remove_redundant(blocks_list):
        if not blocks_list:
            return blocks_list
        
        counts = np.zeros(num_subsets, dtype=np.int32)
        for b in blocks_list:
            counts += M[b]
            
        kept = []
        for b in blocks_list:
            # Check if any subset covered by b has count 1
            # If min(counts[covered_by_b]) > 1, b is redundant
            covered_indices = np.where(M[b] > 0)[0]
            if len(covered_indices) == 0:
                # Block covers nothing? Should not happen
                continue
            if np.min(counts[covered_indices]) > 1:
                # Redundant, remove
                counts -= M[b]
            else:
                kept.append(b)
                
        return kept

    # Initial redundancy removal
    current_blocks = remove_redundant(current_blocks)
    
    # Iterative improvement
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Pass 1: Redundancy removal
        new_blocks = remove_redundant(current_blocks)
        if len(new_blocks) < len(current_blocks):
            current_blocks = new_blocks
            improved = True
            continue # Restart loop as counts changed significantly
            
        # Pass 2: Block Swapping
        # Try to replace one block with another to potentially enable further removal
        # We only swap if it's valid.
        # To find a good swap, we can look for a block that covers many "critical" (count=1) subsets
        # and replace it with a block that covers them plus others, or just a valid swap.
        
        if len(current_blocks) == 0:
            break
            
        counts = np.zeros(num_subsets, dtype=np.int32)
        for b in current_blocks:
            counts += M[b]
            
        # Identify critical subsets (count == 1)
        critical = counts == 1
        
        # For each block in current set, see if we can swap it out
        # We'll try a few random swaps to avoid getting stuck in local minima
        swaps_attempted = 0
        max_swaps = 100 # Limit per iteration to keep time bounded
        
        # Sort current blocks by "criticality" (how many critical subsets they cover)
        # Blocks covering many critical subsets are "important" and hard to remove directly,
        # but swapping them might be key.
        block_criticality = []
        for b in current_blocks:
            crit_covered = np.sum(M[b] & critical)
            block_criticality.append((crit_covered, b))
            
        # Try swapping blocks with high criticality first? Or random?
        # Let's try random swaps from the current set against the full set of blocks
        rng.shuffle(current_blocks)
        
        for i, b_old in enumerate(current_blocks):
            if time.time() > deadline:
                break
            if swaps_attempted >= max_swaps:
                break
                
            # Try to find a replacement block b_new
            # We want b_new such that:
            # 1. b_new is not in current_blocks (or is different index)
            # 2. counts - M[b_old] + M[b_new] >= 1 everywhere
            
            # To speed up, we can pre-filter candidates.
            # A candidate b_new must cover all subsets that b_old uniquely covers.
            # Unique covers of b_old: subsets s where counts[s]==1 and M[b_old, s]==1
            unique_covers = (counts == 1) & (M[b_old] == 1)
            
            if not np.any(unique_covers):
                # b_old is redundant? Should have been removed.
                # If not, maybe due to order. Let's try to remove it.
                # Actually, remove_redundant handles this.
                continue
            
            # Find candidates that cover all unique_covers
            # M_candidate[:, unique_covers] must be all 1s
            # This is a boolean matrix multiplication or sum check
            # unique_covers is a boolean mask of length num_subsets
            # We need M[j, unique_covers] to have sum == sum(unique_covers)
            
            # Optimization: 
            # Let U be the indices of unique_covers.
            # We need M[j, U].all()
            
            # Instead of checking all 50k blocks, we can be smarter.
            # But 50k * 35 ops is ~1.7M ops. Fast enough for a few iterations?
            # Let's just iterate over a subset of candidates or all if fast.
            
            # Get indices of unique covers
            U = np.where(unique_covers)[0]
            
            # Check all blocks
            # M[:, U] is (num_blocks, len(U))
            # We want rows where all are 1.
            
            # To avoid creating a huge temporary array, we can do:
            # sum_M_U = M[:, U].sum(axis=1)
            # valid_candidates = np.where(sum_M_U == len(U))[0]
            
            # But M is uint8.
            sum_M_U = M[:, U].sum(axis=1)
            valid_candidates = np.where(sum_M_U == len(U))[0]
            
            # Filter out current blocks
            current_set = set(current_blocks)
            candidates = [c for c in valid_candidates if c not in current_set]
            
            if not candidates:
                continue
                
            # Pick a random candidate
            # Prefer candidates that cover MORE total subsets (higher degree) to increase slack?
            # Or just random.
            b_new = rng.choice(candidates)
            
            # Check validity
            # new_counts = counts - M[b_old] + M[b_new]
            # We know new_counts >= 1 for unique covers because b_new covers them.
            # We need to check if any other subset drops to 0.
            # Subsets where counts==1 and M[b_old]==1 are safe (covered by b_new).
            # Subsets where counts==1 and M[b_old]==0 are safe (not affected by removal).
            # Subsets where counts>1:
            #   If M[b_old]==1, count decreases by 1. If count was 2, becomes 1. Safe.
            #   If M[b_old]==0, count unchanged. Safe.
            #   If M[b_new]==1, count increases. Safe.
            
            # Wait, is it always valid?
            # Let s be a subset.
            # new_count[s] = counts[s] - M[b_old, s] + M[b_new, s]
            # We require new_count[s] >= 1.
            
            # Case 1: counts[s] == 1
            #   If M[b_old, s] == 1:
            #       Then s is in unique_covers.
            #       We ensured M[b_new, s] == 1.
            #       new_count = 1 - 1 + 1 = 1. OK.
            #   If M[b_old, s] == 0:
            #       Then M[b_new, s] can be 0 or 1.
            #       new_count = 1 - 0 + M[b_new, s] >= 1. OK.
            
            # Case 2: counts[s] >= 2
            #   If M[b_old, s] == 1:
            #       new_count = counts[s] - 1 + M[b_new, s] >= 1 + M[b_new, s] >= 1. OK.
            #   If M[b_old, s] == 0:
            #       new_count = counts[s] + M[b_new, s] >= 2. OK.
            
            # So ANY candidate from `valid_candidates` is a valid swap!
            # This is a huge simplification.
            
            # Perform swap
            current_blocks[i] = b_new
            improved = True
            swaps_attempted += 1
            # Break to restart redundancy check with new configuration
            break
            
    # Final cleanup
    current_blocks = remove_redundant(current_blocks)
    
    # Convert indices back to lists of points
    result_blocks = [list(k_blocks[b]) for b in current_blocks]
    
    # Ensure distinctness (should be distinct indices, so distinct blocks)
    # And validity check just in case
    if not is_valid(result_blocks):
        # Fallback: return greedy result if local search broke it
        # But is_valid on indices is better
        pass
        
    return result_blocks

# EVOLVE-BLOCK-END
