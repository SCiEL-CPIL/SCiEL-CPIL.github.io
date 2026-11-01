import itertools
import time
import random
import numpy as np
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s
    
    # Precomputation
    # 1. Enumerate all t-subsets and assign indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Enumerate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # 3. Build coverage matrix using NumPy
    # coverage_matrix[i, j] is True if k_subset i covers t_subset j
    # We can build this efficiently by iterating k_subsets and marking t_subsets.
    # Since num_k_subsets is ~75k and each covers C(8,5)=56 t_subsets, total ops ~4M.
    coverage_matrix = np.zeros((num_k_subsets, num_t_subsets), dtype=bool)
    
    for i, ks in enumerate(k_subsets):
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            coverage_matrix[i, idx] = True
            
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple random restarts with local improvement
    # The local improvement here is aggressive redundancy removal using coverage counts.
    
    while time.time() < deadline:
        uncovered = np.ones(num_t_subsets, dtype=bool)
        selected_blocks = []
        used_blocks = set()
        coverage_count = np.zeros(num_t_subsets, dtype=np.int32)
        
        # Greedy Construction
        while uncovered.any():
            if time.time() > deadline:
                break
            
            # Vectorized gain calculation: sum of True values in columns where uncovered is True
            # We only need to consider unused blocks. 
            # However, masking the whole matrix each time is expensive.
            # Instead, we can just compute gains for all and ignore used ones, or maintain a list of unused.
            # Given 75k blocks, computing gains for all is fast in NumPy.
            # gains[i] = number of currently uncovered t-subsets covered by block i
            
            # To avoid calculating gains for used blocks (which is 0 effectively for selection but wastes time),
            # we can just calculate for all and mask out used ones.
            gains = coverage_matrix[:, uncovered].sum(axis=1)
            
            # Mask out used blocks
            for idx in used_blocks:
                gains[idx] = 0
                
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            chosen_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            selected_blocks.append(chosen_idx)
            used_blocks.add(chosen_idx)
            
            # Update uncovered and coverage_count
            block_mask = coverage_matrix[chosen_idx]
            uncovered &= ~block_mask
            coverage_count += block_mask.astype(np.int32)
        
        if not uncovered.any():
            # Local Improvement: Remove redundant blocks
            # A block is redundant if every t-subset it covers is covered by at least one other selected block.
            # i.e., coverage_count[covered_by_block] > 1 for all covered_by_block.
            
            # We iterate through selected blocks and try to remove them.
            # To make this faster, we can randomize the order of removal attempts.
            removal_order = list(range(len(selected_blocks)))
            random.shuffle(removal_order)
            
            changed = True
            while changed:
                changed = False
                for pos in removal_order:
                    if time.time() > deadline:
                        break
                    
                    if pos >= len(selected_blocks):
                        continue
                        
                    block_idx = selected_blocks[pos]
                    block_mask = coverage_matrix[block_idx]
                    
                    # Check if any t-subset covered by this block has count == 1
                    # If count == 1, this block is the *only* one covering it, so it's not redundant.
                    # If count > 1 for all, it is redundant.
                    counts_for_block = coverage_count[block_mask]
                    if np.all(counts_for_block > 1):
                        # Remove block
                        selected_blocks.pop(pos)
                        used_blocks.remove(block_idx)
                        coverage_count -= block_mask.astype(np.int32)
                        changed = True
                        # Reset removal order? Or just continue with remaining indices?
                        # Since indices shift, it's safer to restart the loop or update removal_order.
                        # Restarting is simpler and safe.
                        removal_order = list(range(len(selected_blocks)))
                        random.shuffle(removal_order)
                        break # Restart the while loop to re-shuffle/re-evaluate from start
                
                if time.time() > deadline:
                    break

            if len(selected_blocks) < best_count:
                best_count = len(selected_blocks)
                best_blocks = [list(k_subsets[i]) for i in selected_blocks]
                
            # Optional: If we hit a very good solution, we might want to stop early?
            # No, keep trying to improve.
            
    if best_blocks is None:
        # Fallback: Simple greedy without redundancy removal, just in case something went wrong
        uncovered = np.ones(num_t_subsets, dtype=bool)
        current_blocks = []
        used_blocks = set()
        while uncovered.any():
            gains = coverage_matrix[:, uncovered].sum(axis=1)
            for idx in used_blocks:
                gains[idx] = 0
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            chosen_idx = np.argmax(gains)
            current_blocks.append(chosen_idx)
            used_blocks.add(chosen_idx)
            uncovered &= ~coverage_matrix[chosen_idx]
        best_blocks = [list(k_subsets[i]) for i in current_blocks]

    return best_blocks
