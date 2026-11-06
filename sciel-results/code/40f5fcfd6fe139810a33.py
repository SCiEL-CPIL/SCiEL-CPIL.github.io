"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    def time_exceeded():
        return time.time() - start_time >= DEADLINE

    best_k = 1
    best_s = 1

    def find_best_run(phi_arr, N):
        """Find the longest run of consecutive integers with distinct phi values."""
        last_seen = {}
        left = 1
        current_max = 1
        best_k_local = 1
        best_s_local = 1
        
        # Iterate from 1 to N
        # Using a list or array for last_seen might be faster if we map values to indices?
        # Phi values can be up to N. So we can use an array of size N+1 for last_seen.
        # Initialize with -1.
        # This avoids dictionary overhead.
        last_seen_arr = np.full(N + 1, -1, dtype=np.int64)
        
        for right in range(1, N + 1):
            v = phi_arr[right]
            if v <= N: # phi(x) <= x <= N
                if last_seen_arr[v] >= left:
                    left = last_seen_arr[v] + 1
            last_seen_arr[v] = right
            window_len = right - left + 1
            if window_len > current_max:
                current_max = window_len
                best_k_local = window_len
                best_s_local = left
                
        return best_s_local, best_k_local

    # Phase 1: Small range to get a baseline quickly
    N1 = 1000000
    phi_arr1 = np.arange(N1 + 1, dtype=np.int64)
    for i in range(2, N1 + 1):
        if phi_arr1[i] == i:
            phi_arr1[i::i] -= phi_arr1[i::i] // i
            
    s1, k1 = find_best_run(phi_arr1, N1)
    best_s = s1
    best_k = k1
    
    if time_exceeded():
        return {'start': best_s, 'length': best_k}

    # Phase 2: Medium range
    if time_remaining() > 30.0:
        N2 = 10000000
        phi_arr2 = np.arange(N2 + 1, dtype=np.int64)
        for i in range(2, N2 + 1):
            if phi_arr2[i] == i:
                phi_arr2[i::i] -= phi_arr2[i::i] // i
        
        s2, k2 = find_best_run(phi_arr2, N2)
        if k2 > best_k:
            best_s = s2
            best_k = k2
            
    if time_exceeded():
        return {'start': best_s, 'length': best_k}

    # Phase 3: Large range
    if time_remaining() > 80.0:
        N3 = 100000000
        phi_arr3 = np.arange(N3 + 1, dtype=np.int64)
        for i in range(2, N3 + 1):
            if phi_arr3[i] == i:
                phi_arr3[i::i] -= phi_arr3[i::i] // i
        
        s3, k3 = find_best_run(phi_arr3, N3)
        if k3 > best_k:
            best_s = s3
            best_k = k3

    # Ensure constraints
    if best_s + best_k - 1 > 10**12:
        best_k = 10**12 - best_s + 1
    if best_k > 200:
        best_k = 200
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1

    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
