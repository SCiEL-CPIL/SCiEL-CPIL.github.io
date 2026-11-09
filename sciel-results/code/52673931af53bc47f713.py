import numpy as np
import random
import time

N = 17

def check_network(comparators):
    """
    Check if the given network sorts all 2^N boolean inputs.
    Returns True if valid, False otherwise.
    """
    # Efficient verification using bit manipulation
    inputs = (np.arange(2**N)[:, None] // 2**np.arange(N-1, -1, -1)) % 2
    
    for i, j in comparators:
        if i > j:
            i, j = j, i
        min_val = np.minimum(inputs[:, i], inputs[:, j])
        max_val = np.maximum(inputs[:, i], inputs[:, j])
        inputs[:, i] = min_val
        inputs[:, j] = max_val
        
    diffs = np.diff(inputs, axis=1)
    return np.all(diffs >= 0)

def generate_bitonic_sort(n):
    """
    Generate a Bitonic Sort network for n wires (n must be power of 2).
    Returns a list of (i, j) pairs.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    n_half = n // 2
    # Recursively sort the two halves
    # Note: Bitonic sort sorts the whole array, but the recursive step
    # sorts the first half and the second half independently?
    # No, Bitonic Sort:
    # 1. Sort first half (0..n/2-1) and second half (n/2..n-1) recursively.
    # 2. Merge them.
    
    comps_left = generate_bitonic_sort(n_half)
    comps_right = [(i + n_half, j + n_half) for i, j in generate_bitonic_sort(n_half)]
    
    # Merge step
    # The merge step for Bitonic Sort involves comparing elements
    # in a specific pattern.
    # For n=4, merge (0,1,2,3) where 0,1 is sorted and 2,3 is sorted (reverse?).
    # Actually, Bitonic Sort creates a bitonic sequence.
    # The standard recursive definition:
    # BitonicSort(A, n):
    #   if n > 1:
    #     BitonicSort(A, n/2)
    #     BitonicSort(A+n/2, n/2)
    #     Merge(A, n)
    
    # Merge(A, n):
    #   for k in range(n//2, 0, -1):
    #     for i in range(n - k):
    #       compare (i, i+k)
    
    comps_merge = []
    k = n_half
    while k > 0:
        for i in range(n - k):
            comps_merge.append((i, i + k))
        k //= 2
        
    return comps_left + comps_right + comps_merge

def construct():
    # Generate Bitonic Sort for 16 wires
    comps_16 = generate_bitonic_sort(16)
    
    # Extend to 17 wires
    # Add wire 16.
    # To sort 17 wires, we can add comparators to merge wire 16 into the sorted 16.
    # A simple way: Compare 16 with 0, 1, ..., 15.
    # This ensures that the smallest value ends up in the correct position?
    # If we do (0, 16), (1, 16), ... (15, 16):
    # If 16 is small, it moves to 0?
    # (0,16): min->0, max->16.
    # (1,16): min->1, max->16.
    # If the value at 16 is the global min, it will move to 0, then stay at 0?
    # No, (1,16) compares 1 and 16. If 16 has the min, it goes to 1.
    # So the min moves to the last index compared.
    # This effectively inserts the element at the end.
    # To insert into a sorted array, we should compare in reverse?
    # Or just add them and let the search optimize.
    
    # Let's add comparators (16, i) for i in 0..15.
    # This is a "merge" step.
    comps_17 = comps_16[:]
    for i in range(16):
        comps_17.append((16, i))
        
    # Initial network
    current_network = comps_17[:]
    best_network = current_network[:]
    
    start_time = time.time()
    timeout = 55
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    
    while time.time() - start_time < timeout:
        if random.random() < T:
            # Perturbation
            candidate = current_network[:]
            
            # Randomly modify a comparator
            if len(candidate) > 0:
                idx = random.randint(0, len(candidate) - 1)
                i, j = candidate[idx]
                
                # Try to change i or j
                if random.random() < 0.5:
                    new_i = random.randint(0, N - 1)
                    if new_i != j:
                        candidate[idx] = (min(new_i, j), max(new_i, j))
                else:
                    new_j = random.randint(0, N - 1)
                    if new_j != i:
                        candidate[idx] = (min(i, new_j), max(i, new_j))
            
            # Check validity
            if check_network(candidate):
                # Accept valid move
                current_network = candidate
                if len(current_network) < len(best_network):
                    best_network = current_network[:]
                # Reduce temperature
                T *= cooling
            else:
                # Reject
                T *= cooling
        else:
            # Try to remove a comparator
            if len(current_network) > 0:
                idx = random.randint(0, len(current_network) - 1)
                candidate = current_network[:idx] + current_network[idx+1:]
                if check_network(candidate):
                    current_network = candidate
                    if len(current_network) < len(best_network):
                        best_network = current_network[:]
                    T = max(T, 0.5) # Reheat slightly

    return best_network
