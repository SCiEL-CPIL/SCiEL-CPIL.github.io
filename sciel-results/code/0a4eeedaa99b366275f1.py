"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    def compute_phi_sieve(n):
        """Compute phi for all numbers from 1 to n using sieve."""
        phi = list(range(n + 1))
        for i in range(2, n + 1):
            if phi[i] == i:  # i is prime
                for j in range(i, n + 1, i):
                    phi[j] -= phi[j] // i
        return phi
    
    def scan_for_best_run(phi_vals, max_k=200):
        """
        Scan phi_vals for the longest run of distinct values.
        Uses a dictionary to track last seen position for O(N) scanning.
        Returns (best_k, best_s)
        """
        best_k = 1
        best_s = 1
        n = len(phi_vals)
        
        # last_seen[v] = index of last occurrence of value v
        # Use sliding window with two pointers
        last_seen = {}
        left = 0
        for right in range(n):
            v = phi_vals[right]
            if v in last_seen:
                # Move left pointer past the previous occurrence
                if last_seen[v] >= left:
                    left = last_seen[v] + 1
            last_seen[v] = right
            
            current_k = right - left + 1
            if current_k > best_k:
                best_k = current_k
                best_s = left
                if best_k >= max_k:
                    break
        
        return best_k, best_s
    
    best_k = 1
    best_s = 1
    
    # Strategy: compute phi for increasingly larger ranges
    # Keep the best result found so far
    # Stop when time runs out
    
    # Try a sequence of range sizes, starting moderate and expanding
    range_sizes = [200000, 500000, 1000000, 2000000, 5000000, 10000000]
    
    for range_size in range_sizes:
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            break
        
        # Conservative time budgets for each range size
        if range_size >= 10000000 and elapsed > deadline - 60:
            break
        if range_size >= 5000000 and elapsed > deadline - 40:
            break
        if range_size >= 2000000 and elapsed > deadline - 25:
            break
        if range_size >= 1000000 and elapsed > deadline - 15:
            break
        
        # Compute phi for this range
        phi_vals = compute_phi_sieve(range_size)
        
        # Scan for longest run using efficient O(N) sliding window method
        k_found, s_found = scan_for_best_run(phi_vals, max_k=200)
        
        if k_found > best_k:
            best_k = k_found
            best_s = s_found
        
        if best_k >= 200:
            break
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
