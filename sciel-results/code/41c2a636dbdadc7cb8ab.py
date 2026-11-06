import time
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    def time_remaining():
        return DEADLINE - (time.time() - start_time)
    
    def compute_phi_sieve_numpy(limit):
        """
        Compute phi for all numbers up to limit using a vectorized sieve.
        Returns an array phi where phi[i] = phi(i) for i in 0..limit.
        """
        phi = np.arange(limit + 1, dtype=np.int64)
        # Standard Euler totient sieve
        for i in range(2, limit + 1):
            if phi[i] == i:  # i is prime
                # Update multiples of i
                # phi[j] *= (1 - 1/i) for j multiple of i
                # Vectorized: phi[j] -= phi[j] // i
                indices = np.arange(i, limit + 1, i)
                phi[indices] -= phi[indices] // i
        return phi

    def find_longest_distinct_run(phi_vals):
        """
        Find the longest contiguous subsequence of phi_vals (starting from index 1)
        with all distinct values.
        phi_vals: array where phi_vals[i] = phi(i).
        Returns (max_length, start_index) where start_index is the 1-based integer
        corresponding to the start of the run.
        """
        n = len(phi_vals)
        if n < 2:
            return 1, 1
            
        # We care about indices 1 to n-1 in phi_vals (corresponding to integers 1 to n-1)
        # phi_vals[0] is phi(0)=0, usually ignored for s>=1
        
        last_seen = {}
        left = 1  # Left pointer, 1-based index in phi_vals (which is the integer value)
        max_len = 0
        max_start = 1
        
        for right in range(1, n):
            val = phi_vals[right]
            # Check if this value has been seen in the current window [left, right]
            if val in last_seen and last_seen[val] >= left:
                left = last_seen[val] + 1
            last_seen[val] = right
            
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
                max_start = left
                
        return max_len, max_start

    best_k = 1
    best_s = 1
    
    # Strategy: Try increasing sieve limits.
    # 10^7 is very fast (<1s).
    # 5*10^7 is moderate (~5s).
    # 10^8 is risky (~10-20s depending on hardware).
    # We stop if time runs out.
    
    # Initial small limit to get a baseline quickly
    limits = [1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000, 200_000_000]
    
    for limit in limits:
        # Need enough time for the sieve and the scan
        # Sieve is O(N log log N), scan is O(N)
        # Rough estimate: 10^8 takes ~5-10s. 2*10^8 takes ~10-20s.
        # If we have < 15s left, don't start a big one.
        if time_remaining() < 15.0:
            break
            
        try:
            # Compute phi
            phi_vals = compute_phi_sieve_numpy(limit)
            
            # Find longest run
            k, s = find_longest_distinct_run(phi_vals)
            
            # Cap k at 200 as per task constraints
            # If the run is longer than 200, we can just take the first 200 elements of that run.
            # The start 's' found is the start of the maximal run.
            # If k > 200, the run [s, s+k-1] is valid.
            # We need to return a length <= 200.
            # So we can return length = min(k, 200) and start = s.
            # Wait, if k > 200, does s + 200 - 1 <= limit?
            # Yes, because s + k - 1 <= limit and k > 200 implies s + 200 - 1 < s + k - 1 <= limit.
            
            effective_k = min(k, 200)
            
            if effective_k > best_k:
                best_k = effective_k
                best_s = s
                
        except MemoryError:
            break
        except Exception:
            # In case of any other error, try the next smaller limit or break?
            # Usually MemoryError is the main risk.
            # If it's a bug, we might want to break to avoid infinite loops or repeated failures.
            # But for now, let's just break to be safe.
            break

    # Final safety checks
    if best_k < 1:
        best_k = 1
        best_s = 1
        
    if best_k > 200:
        best_k = 200
        
    # Ensure s + k - 1 <= 10^12 (trivially true for our limits <= 2*10^8)
    # Ensure s >= 1 (guaranteed by find_longest_distinct_run)
    
    return {'start': best_s, 'length': best_k}
