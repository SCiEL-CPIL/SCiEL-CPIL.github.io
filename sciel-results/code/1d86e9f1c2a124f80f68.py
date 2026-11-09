"""Program to find longest run of consecutive integers with distinct totients."""
import numpy as np

def construct():
    # Compute phi for n = 1 to N using a sieve
    # Then find the longest run of distinct phi values using two-pointer
    
    N = 5_000_000
    phi_arr = np.arange(N + 1, dtype=np.int64)
    
    # Totient sieve
    for i in range(2, N + 1):
        if phi_arr[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi_arr[j] -= phi_arr[j] // i
    
    # Two-pointer to find longest window with all distinct phi values
    # phi values are in [1, N], so use an array for last_seen
    last_seen = np.full(N + 1, -1, dtype=np.int32)
    left = 1  # start from n=1
    best_k = 1
    best_s = 1
    
    for right in range(1, N + 1):
        v = phi_arr[right]
        if last_seen[v] >= left:
            left = last_seen[v] + 1
        last_seen[v] = right
        k = right - left + 1
        if k > best_k:
            best_k = k
            best_s = left
    
    return {'start': best_s, 'length': best_k}
