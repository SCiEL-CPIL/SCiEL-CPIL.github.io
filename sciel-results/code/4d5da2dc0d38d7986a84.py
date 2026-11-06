import time
import math
import numpy as np
from sympy import factorint, totient, isprime

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
        for i in range(2, limit + 1):
            if phi[i] == i:  # i is prime
                indices = np.arange(i, limit + 1, i)
                phi[indices] -= phi[indices] // i
        return phi

    def find_longest_distinct_run(phi_vals, offset=0):
        """
        Find the longest contiguous subsequence of phi_vals with all distinct values.
        phi_vals: array where phi_vals[i] = phi(offset + i)
        offset: the integer value corresponding to phi_vals[0]
        Returns (max_length, start_index_in_phi_vals)
        """
        n = len(phi_vals)
        if n < 2:
            return 1, 0
            
        last_seen = {}
        left = 0
        max_len = 0
        max_start = 0
        
        for right in range(n):
            val = phi_vals[right]
            if val in last_seen and last_seen[val] >= left:
                left = last_seen[val] + 1
            last_seen[val] = right
            
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
                max_start = left
                
        return max_len, max_start

    def verify_run(s, k, phi_cache=None):
        """
        Verify that phi(s), ..., phi(s+k-1) are pairwise distinct.
        Uses sympy for exact factorization if needed.
        phi_cache: optional dict mapping int -> int for phi values already computed.
        """
        if k <= 1:
            return True
            
        seen = set()
        for i in range(k):
            n = s + i
            if phi_cache is not None and n in phi_cache:
                val = phi_cache[n]
            else:
                # Use sympy for exact computation
                val = totient(n)
                if phi_cache is not None:
                    phi_cache[n] = val
                    
            if val in seen:
                return False
            seen.add(val)
            
        return True

    best_k = 1
    best_s = 1
    phi_cache = {}
    
    # Phase 1: Sieve-based search for smaller ranges
    # Try increasing limits, but keep them manageable
    sieve_limits = [1000000, 5000000, 10000000]
    
    for limit in sieve_limits:
        if time_remaining() < 30.0:  # Leave time for verification and larger searches
            break
            
        try:
            phi_vals = compute_phi_sieve_numpy(limit)
            # phi_vals[0] = phi(0), phi_vals[1] = phi(1), etc.
            # We care about indices 1..limit
            if limit >= 1:
                k, rel_start = find_longest_distinct_run(phi_vals[1:], offset=1)
                s = rel_start + 1  # Convert to actual number
                
                # Cap k at 200
                if k > 200:
                    k = 200
                
                if k > best_k:
                    # Verify the run using sympy to ensure correctness
                    if verify_run(s, k, phi_cache):
                        best_k = k
                        best_s = s
                    # If verification fails, we keep the old best
        except Exception:
            break
            
    # Phase 2: Targeted search in larger ranges using probabilistic heuristics
    # The longest runs of distinct totients tend to occur at certain locations.
    # Based on known results, long runs are found near multiples of primorials or in gaps.
    # We'll search specific candidate regions in the range [10^7, 10^12]
    
    # Candidate start points based on known patterns:
    # Long runs often start near numbers with many small prime factors or in prime gaps
    candidate_starts = []
    
    # Add some heuristic candidates
    # 1. Near primorials
    primorials = [2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870, 6469693230, 200862915270, 7420738134810]
    for p in primorials:
        if p <= 10**12:
            # Check around the primorial
            for delta in [-10, -5, -1, 0, 1, 5, 10]:
                s = p + delta
                if 1 <= s <= 10**12 - 200:
                    candidate_starts.append(s)
    
    # 2. Random sampling in large ranges
    if time_remaining() > 60.0:
        # Sample random starting points in [10^8, 10^12]
        np.random.seed(42)
        for _ in range(50):
            s = np.random.randint(10**8, 10**12 - 200)
            candidate_starts.append(int(s))
    
    # 3. Also try to extend the best run found in Phase 1
    if best_k > 1:
        # Try extending the current best run
        candidate_starts.append(best_s - 1)
        candidate_starts.append(best_s + best_k)
    
    # Remove duplicates and sort
    candidate_starts = list(set(candidate_starts))
    candidate_starts.sort()
    
    # Phase 3: Verify and refine candidate runs
    # For each candidate, compute phi values and check for distinctness
    # We'll do a local search around each candidate
    
    for s in candidate_starts:
        if time_remaining() < 15.0:
            break
            
        if s < 1 or s > 10**12 - 200:
            continue
            
        # First, try to find a run starting at or near s
        # We'll compute phi values for a window around s
        
        # Compute phi for a window [s-50, s+250] to allow for adjustment
        window_start = max(1, s - 50)
        window_end = min(10**12, s + 250)
        window_size = window_end - window_start + 1
        
        if window_size > 300:
            window_size = 300
            window_end = window_start + window_size - 1
        
        # Compute phi values for this window using sympy (slower but exact)
        # This is only for small windows, so it's acceptable
        phi_window = {}
        for i in range(window_start, window_end + 1):
            if i not in phi_cache:
                phi_cache[i] = totient(i)
            phi_window[i] = phi_cache[i]
        
        # Now find the longest distinct run in this window
        # We can use the sliding window approach on the sorted keys
        keys = sorted(phi_window.keys())
        if len(keys) < 2:
            continue
            
        # Check consecutive integers only
        # Build an array of phi values for consecutive integers
        consecutive_phi = []
        for i in range(window_start, window_end + 1):
            if i in phi_window:
                consecutive_phi.append(phi_window[i])
            else:
                # Should not happen if window is contiguous
                break
        
        if len(consecutive_phi) < 2:
            continue
            
        # Find longest distinct run
        k_local, rel_start = find_longest_distinct_run(np.array(consecutive_phi), offset=window_start)
        s_local = rel_start + window_start
        
        if k_local > 200:
            k_local = 200
            
        if k_local > best_k:
            # Verify the run
            if verify_run(s_local, k_local, phi_cache):
                best_k = k_local
                best_s = s_local
            elif k_local == best_k:
                # Same length, but verify this one too
                if verify_run(s_local, k_local, phi_cache):
                    best_s = s_local
    
    # Phase 4: If we have time, try to extend the best run found
    if best_k > 1 and time_remaining() > 20.0:
        # Try to extend the run to the left
        s_left = best_s - 1
        while s_left >= 1 and time_remaining() > 10.0:
            if s_left in phi_cache:
                val = phi_cache[s_left]
            else:
                val = totient(s_left)
                phi_cache[s_left] = val
            
            # Check if val is already in the run
            run_vals = set()
            for i in range(best_k):
                n = best_s + i
                if n in phi_cache:
                    run_vals.add(phi_cache[n])
                else:
                    run_vals.add(totient(n))
            
            if val not in run_vals:
                # Can extend
                if best_k + 1 <= 200:
                    best_k += 1
                    best_s = s_left
                    s_left -= 1
                else:
                    break
            else:
                break
        
        # Try to extend the run to the right
        s_right = best_s + best_k
        while s_right <= 10**12 and time_remaining() > 10.0:
            if s_right in phi_cache:
                val = phi_cache[s_right]
            else:
                val = totient(s_right)
                phi_cache[s_right] = val
            
            # Check if val is already in the run
            run_vals = set()
            for i in range(best_k):
                n = best_s + i
                if n in phi_cache:
                    run_vals.add(phi_cache[n])
                else:
                    run_vals.add(totient(n))
            
            if val not in run_vals:
                # Can extend
                if best_k + 1 <= 200:
                    best_k += 1
                else:
                    break
                s_right += 1
            else:
                break
    
    # Final safety checks
    if best_k < 1:
        best_k = 1
        best_s = 1
        
    if best_k > 200:
        best_k = 200
        
    if best_s < 1:
        best_s = 1
        
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
        if best_k < 1:
            best_k = 1
            best_s = 1
            
    return {'start': best_s, 'length': best_k}
