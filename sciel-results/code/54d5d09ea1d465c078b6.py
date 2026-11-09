import numpy as np
import time

N = 18
TOTAL_INPUTS = 1 << N

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # 1. Precompute bits for all inputs
    # Shape: (2^18, 18), dtype bool
    # bits[x, i] is the i-th bit of x (0 is LSB)
    # We assume wire 0 corresponds to bit 0 (LSB).
    # Sorted means bits are non-decreasing: b0 <= b1 <= ... <= b17.
    
    x = np.arange(TOTAL_INPUTS, dtype=np.uint32)
    shifts = np.arange(N, dtype=np.uint32)
    # (262144, 1) >> (1, 18)
    bits = ((x[:, None] >> shifts[None, :]) & 1).astype(bool)
    
    # 2. Initialize active mask
    # An input is active if it is not sorted.
    # Sorted if bits[i] <= bits[i+1] for all i.
    sorted_mask = np.all(bits[:, :-1] <= bits[:, 1:], axis=1)
    active_mask = ~sorted_mask
    
    network = []
    
    # List of all possible comparators (i, j) with i < j
    comp_i = []
    comp_j = []
    for i in range(N):
        for j in range(i+1, N):
            comp_i.append(i)
            comp_j.append(j)
    comp_i = np.array(comp_i)
    comp_j = np.array(comp_j)
    
    while np.any(active_mask):
        if time.time() > deadline:
            break
            
        # Calculate gain for all comparators
        # gain[k] = sum(active_mask & bits[:, comp_i[k]] & ~bits[:, comp_j[k]])
        
        gains = np.zeros(len(comp_i), dtype=np.int32)
        
        # Loop over comparators
        for k in range(len(comp_i)):
            i = comp_i[k]
            j = comp_j[k]
            # Count active inputs where bit i is 1 and bit j is 0
            gains[k] = np.sum(active_mask & bits[:, i] & ~bits[:, j])
            
        # Find the comparator with the maximum gain
        best_k = np.argmax(gains)
        best_gain = gains[best_k]
        
        if best_gain == 0:
            break
            
        i = comp_i[best_k]
        j = comp_j[best_k]
        
        network.append((int(i), int(j)))
        
        # Apply comparator (i, j)
        affected_mask = active_mask & bits[:, i] & ~bits[:, j]
        affected_indices = np.where(affected_mask)[0]
        
        if len(affected_indices) > 0:
            # Swap bits i and j for affected indices
            temp_i = bits[affected_indices, i].copy()
            temp_j = bits[affected_indices, j].copy()
            
            bits[affected_indices, i] = temp_j
            bits[affected_indices, j] = temp_i
            
            # Check if any affected inputs are now sorted
            is_now_sorted = np.all(bits[affected_indices, :-1] <= bits[affected_indices, 1:], axis=1)
            
            # Update active_mask
            to_remove = affected_indices[is_now_sorted]
            if len(to_remove) > 0:
                active_mask[to_remove] = False
                
    return network
