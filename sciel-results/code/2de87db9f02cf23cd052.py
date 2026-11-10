import numpy as np
import time

def _max_ap_discrepancy(x):
    """
    Calculate the maximum absolute sum of any arithmetic progression in x.
    x is a numpy array of -1, 1.
    """
    n = len(x)
    max_disc = 1  # Singletons
    for d in range(1, n // 2 + 1):
        for r in range(d):
            seq = x[r::d]
            L = len(seq)
            if L < 2:
                continue
            # Kadane's algorithm for max and min subarray sum
            max_ending_here = 0
            max_so_far = 0
            min_ending_here = 0
            min_so_far = 0
            for val in seq:
                # Max
                max_ending_here = val if max_ending_here < 0 else max_ending_here + val
                if max_ending_here > max_so_far:
                    max_so_far = max_ending_here
                # Min
                min_ending_here = val if min_ending_here > 0 else min_ending_here + val
                if min_ending_here < min_so_far:
                    min_so_far = min_ending_here
            
            current_max = max(abs(max_so_far), abs(min_so_far))
            if current_max > max_disc:
                max_disc = current_max
    return max_disc

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 256
    best_x = None
    best_val = float('inf')
    
    # 1. Optimized Greedy Initialization
    # sums[d][i] stores the sum of the AP ending at i with step d.
    # We only need to store the current sums to compute the next.
    # Actually, we can just maintain a 2D array of size (n, n) or (n, 1) per step?
    # Let's use a list of arrays or a 2D numpy array.
    # sums[d, i] for d in 1..n, i in 0..n.
    # Memory: 256*256 * 4 bytes = 256KB. Fine.
    
    x = np.zeros(n, dtype=np.int32)
    # sums[d, i] is the sum x[i] + x[i-d] + ...
    # We initialize sums to 0.
    sums = np.zeros((n + 1, n), dtype=np.int32)
    
    for i in range(n):
        best_local_val = float('inf')
        best_sign = 1
        
        for s in [-1, 1]:
            x[i] = s
            local_max = 1
            # Update sums for all d <= i+1
            for d in range(1, i + 1):
                # sum ending at i with step d
                # previous sum was sums[d, i-d]
                # if i-d < 0, previous sum is 0
                prev_sum = sums[d, i - d] if i - d >= 0 else 0
                current_sum = s + prev_sum
                sums[d, i] = current_sum
                if abs(current_sum) > local_max:
                    local_max = abs(current_sum)
            
            if local_max < best_local_val:
                best_local_val = local_max
                best_sign = s
        
        x[i] = best_sign
    
    val = _max_ap_discrepancy(x)
    if val < best_val:
        best_val = val
        best_x = x.copy()
        
    # 2. Simulated Annealing
    current_x = best_x.copy()
    current_val = best_val
    
    T = 2.0
    T_min = 0.01
    cooling = 0.995
    
    while time.time() < deadline:
        i = np.random.randint(n)
        old_sign = current_x[i]
        current_x[i] = -old_sign
        
        new_val = _max_ap_discrepancy(current_x)
        
        if new_val < current_val:
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_x = current_x.copy()
        else:
            delta = new_val - current_val
            if delta > 0:
                # Accept with probability
                # To avoid overflow in exp, if delta/T is large, prob is 0
                if delta / T < 50:
                    prob = np.exp(-delta / T)
                    if np.random.rand() < prob:
                        current_val = new_val
                    else:
                        current_x[i] = old_sign
                else:
                    current_x[i] = old_sign
            else:
                current_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_x = current_x.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_x.tolist()
