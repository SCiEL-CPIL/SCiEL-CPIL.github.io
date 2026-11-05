import time
import numpy as np

N = 18
TOTAL_INPUTS = 1 << N
TIME_LIMIT = 85.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Precompute sorted masks
    # Sorted means non-decreasing: 0...01...1
    sorted_masks = set()
    for k in range(N + 1):
        # k ones at the end (high indices)
        mask = ((1 << k) - 1) << (N - k)
        sorted_masks.add(mask)
    
    is_sorted = np.zeros(TOTAL_INPUTS, dtype=bool)
    for m in sorted_masks:
        is_sorted[m] = True
        
    # Initialize states
    states = np.arange(TOTAL_INPUTS, dtype=np.int64)
    
    network = []
    
    while True:
        if time.time() - start_time > TIME_LIMIT:
            break
            
        if np.all(is_sorted[states]):
            break
            
        unsorted_mask = ~is_sorted[states]
        unsorted_states = states[unsorted_mask]
        
        best_score = -1
        best_comp = None
        
        # Evaluate all candidates
        for i in range(N):
            for j in range(i + 1, N):
                mask_i = 1 << i
                mask_j = 1 << j
                
                # Apply comparator to unsorted_states
                inv_mask = (unsorted_states & mask_i) & ~(unsorted_states & mask_j)
                new_unsorted = unsorted_states.copy()
                new_unsorted[inv_mask] = unsorted_states[inv_mask] ^ mask_i ^ mask_j
                
                # Count newly sorted
                num_sorted = np.sum(is_sorted[new_unsorted])
                
                if num_sorted > best_score:
                    best_score = num_sorted
                    best_comp = (i, j)
                    
        if best_comp is None:
            break
            
        # If no state became sorted, pick a random comparator to break symmetry
        if best_score == 0:
            import random
            best_comp = random.choice([(i, j) for i in range(N) for j in range(i+1, N)])
            
        i, j = best_comp
        mask_i = 1 << i
        mask_j = 1 << j
        inv_mask = (states & mask_i) & ~(states & mask_j)
        states[inv_mask] = states[inv_mask] ^ mask_i ^ mask_j
        network.append(best_comp)
        
    return network
# EVOLVE-BLOCK-END
