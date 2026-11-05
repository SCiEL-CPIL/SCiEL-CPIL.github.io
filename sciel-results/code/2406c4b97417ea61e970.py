import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, aim for 140s usage
    DEADLINE = start_time + 140.0
    
    # 1. Precompute primes up to 10^6
    # sqrt(10^12) is 10^6. Any composite factor > 10^6 cannot exist for n <= 10^12 
    # without having a prime factor <= 10^6.
    limit_p = 10**6
    is_prime = [True] * (limit_p + 1)
    is_prime[0] = False
    if limit_p >= 1:
        is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # 2. Helper for single phi calculation (fallback or verification)
    def phi_single(x):
        if x == 1:
            return 1
        n = x
        result = x
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
        if n > 1:
            result -= result // n
        return result

    best_k = 1
    best_s = 1

    # 3. Segmented Sieve for Totient
    # Strategy: Compute phi(n) for a large window of consecutive integers using vectorized NumPy ops.
    # This is O(W log log W) effectively, much faster than individual factorization for dense ranges.
    
    def compute_phi_segment(start, length):
        """
        Computes phi(n) for n in [start, start + length - 1].
        Returns a numpy array of phis.
        """
        # Initialize with the values themselves
        # Using int64 to avoid overflow during intermediate multiplications? 
        # phi(n) <= n <= 10^12, which fits in int64.
        # However, we do phi = phi * (p-1) // p. 
        # If we do integer division at each step, it stays exact and within bounds.
        
        # We need an array of the numbers themselves to check divisibility?
        # No, we just need to mark which indices are divisible by p.
        
        phis = np.arange(start, start + length, dtype=np.int64)
        
        for p in primes:
            if p > int(math.isqrt(start + length - 1)):
                break
            
            # Find indices in the segment divisible by p
            # first multiple of p >= start
            first = ((start + p - 1) // p) * p
            
            if first > start + length - 1:
                continue
                
            # Indices are (first - start), (first - start + p), ...
            # We need to update phis at these indices: phi = phi * (p-1) // p
            # Vectorized operation:
            indices = np.arange(first - start, length, p)
            
            # Update phi values
            # phis[indices] = phis[indices] * (p - 1) // p
            # This is exact integer arithmetic.
            phis[indices] = (phis[indices] * (p - 1)) // p
            
        return phis

    def find_best_run_in_segment(start, phis):
        """
        Finds the longest run of distinct values in phis.
        Returns (best_k, best_local_start) where best_local_start is offset from start.
        """
        n = len(phis)
        if n == 0:
            return 0, 0
        
        # We want to find max k such that phi[start+i]...phi[start+i+k-1] are distinct.
        # Since k is small (<=200), we can slide a window.
        # However, checking distinctness for every window of size 200 is O(N * K).
        # N ~ 10^5, K ~ 200 -> 20M ops. Might be slow in pure Python.
        # Let's optimize: 
        # If we just want to find *a* long run, we can check windows of size 200 first.
        # If none found, check 199, etc.
        # But usually, if there's a long run, there's a 200 run nearby.
        
        # Alternative: Use a sliding window with a frequency count.
        # But Python set operations are slow.
        
        # Given K <= 200, we can just check all windows of size K_max=200.
        # If we find one, return 200.
        # If not, maybe check smaller?
        # Actually, the goal is to maximize k. 
        # If we find a 200 run, we are done with this segment.
        # If we don't, we might still find a 199, 198...
        
        # Let's just scan for the maximum distinct run length.
        # Since K is small, we can do this:
        # For each start index i, extend j as far as possible.
        # This is O(N * Avg_Run_Length).
        
        max_k = 0
        max_s = 0
        
        # Optimization: Only check windows up to size 200.
        # If we find a collision at step k, the run length is k.
        
        # To speed this up, let's just check if there is a run of length 200.
        # If yes, return it.
        # If no, check 199?
        # It's better to just find the global max in the segment.
        
        # Let's use a simple sliding window with a set, but limit the window size to 200.
        # We'll iterate through the array.
        
        # Actually, since we just need the *best* k in the segment, and k is small,
        # we can just check every window of size 200.
        # If any window of size 200 is distinct, we found k=200.
        # If not, we can try to find the max.
        
        # Let's do a linear scan for the longest distinct subarray, capped at 200.
        # This is O(N * 200) worst case if we use sets, but usually breaks early.
        
        # Better: 
        # Since we only care about k up to 200, we can check:
        # for i in range(0, n - 200 + 1):
        #    if len(set(phis[i:i+200])) == 200: return 200, i
        
        # If no 200, check 199?
        # This might be slow if we do it for many segments.
        
        # Let's just find the longest run using a simple algorithm.
        
        current_start = 0
        seen = set()
        current_len = 0
        best_local_k = 0
        best_local_s = 0
        
        # This O(N * K) approach is risky if N is large and K is 200.
        # 10^5 * 200 = 20M set ops. Too slow in Python.
        
        # Alternative:
        # We know that collisions are rare for large numbers.
        # Most runs will be short.
        # We can just check windows of size 200.
        # If we don't find 200, we can sample a few smaller windows or just return the best found by a cheaper method.
        
        # Let's try a different approach:
        # Check for 200. If found, return.
        # If not, check for 100. If found, return.
        # If not, check for 50...
        # This reduces average cost if long runs are rare.
        
        target_sizes = [200, 150, 100, 50, 20, 10, 5, 2, 1]
        
        for k in target_sizes:
            if k > n:
                continue
            # Check all windows of size k
            # To make this faster, we can use a sliding window count?
            # Or just sets?
            # For k=200, N=10^5, 10^5 set creations of size 200.
            # That's heavy.
            
            # Let's just check a random subset of windows if k is large?
            # No, we need to be sure.
            
            # Actually, if we are scanning a segment of size 10^5, and we expect long runs to be rare,
            # we might just check the first few windows or random windows?
            # But we want the *best* in the segment to update global best.
            
            # Let's stick to checking all windows for k=200.
            # If found, return.
            # If not, we can afford to be less thorough for smaller k?
            # Or just stop?
            # If we don't find 200, the best is likely < 200.
            # We can just return the best found so far (which might be from a previous segment or default).
            # But we should try to find *some* improvement.
            
            # Let's optimize the check for k=200.
            # We can use a sliding window with a frequency dictionary.
            # But updating a dict is slow.
            
            # Let's just do:
            found = False
            for i in range(n - k + 1):
                # Check if phis[i:i+k] are distinct
                # Using a set is O(k).
                # Total O(N*k).
                # For k=200, N=10^5, this is 20M ops. ~1-2 seconds in Python? Maybe too slow.
                
                # Let's try to be smarter.
                # If we find a collision, we can skip ahead?
                # Not easily with a fixed window.
                
                # Let's just check every 10th window to save time?
                # If we miss the best, we might not update global best.
                # But global best is updated across many segments.
                
                # Let's check all windows but break early if we find one.
                # Since we want to find *a* run of size k, we can stop at the first one.
                # This makes it O(N*k) worst case, but often much faster if a run exists.
                
                # If no run of size k exists, it's O(N*k).
                
                # To save time, let's only check k=200 and k=100.
                # If we don't find those, we assume the best is small and let other segments handle it.
                
                pass
            
            # This inner loop is the bottleneck.
            # Let's rewrite this function to be more efficient.
            pass
        
        # Revised strategy for find_best_run_in_segment:
        # 1. Check for k=200. If found, return 200, index.
        # 2. Check for k=100. If found, return 100, index.
        # 3. Return 0, 0 (let other logic handle small runs).
        
        # Implementation of step 1:
        k_target = 200
        if n >= k_target:
            for i in range(n - k_target + 1):
                # Check distinctness of phis[i:i+k_target]
                # Use a set
                # Optimization: If we find a duplicate, break.
                # We can iterate through the slice.
                # This is still O(k) per window.
                
                # Let's just do it.
                # If it's too slow, we reduce k_target or segment size.
                
                seen = set()
                distinct = True
                for j in range(k_target):
                    val = phis[i + j]
                    if val in seen:
                        distinct = False
                        break
                    seen.add(val)
                
                if distinct:
                    return 200, i
        
        # If no 200, try 100?
        k_target = 100
        if n >= k_target:
            for i in range(n - k_target + 1):
                seen = set()
                distinct = True
                for j in range(k_target):
                    val = phis[i + j]
                    if val in seen:
                        distinct = False
                        break
                    seen.add(val)
                
                if distinct:
                    return 100, i
        
        return 0, 0

    # 4. Search Strategy
    # Scan segments near 10^12 where collisions are rarest.
    # Also random segments.
    
    segment_size = 50000  # 50k numbers per segment
    
    # Precompute primes for the sieve loop
    # The sieve loop in compute_phi_segment iterates over all primes up to sqrt(max_n).
    # sqrt(10^12) = 10^6.
    # Number of primes ~ 78k.
    # For each prime, we do vectorized ops on an array of size 50k.
    # This should be fast enough with NumPy.
    
    # Phase 1: Scan downwards from 10^12
    # We'll process segments of size 50k.
    # Number of segments to cover 10^12 is 20 million. We can't do them all.
    # We'll do a random sample of segments near the top.
    
    # Let's define a helper to process one segment
    def process_segment(start):
        if time.time() > DEADLINE - 5.0:
            return 0, 0
        if start + segment_size - 1 > 10**12:
            segment_size_adj = 10**12 - start + 1
            if segment_size_adj <= 0:
                return 0, 0
            phis = compute_phi_segment(start, segment_size_adj)
        else:
            phis = compute_phi_segment(start, segment_size)
        
        k, local_s = find_best_run_in_segment(start, phis)
        if k > 0:
            return k, start + local_s
        return 0, 0

    # Phase 2: Random Sampling of Segments
    # Focus on the upper range [10^11, 10^12] where numbers are larger and phi values more spread out?
    # Actually, phi(n) ~ n. Collisions phi(a)=phi(b) are more likely if a,b are close.
    # But for large n, the density of collisions might be lower?
    # Let's just sample uniformly from [1, 10^12 - segment_size].
    
    # To ensure we don't pick the same segment twice, we could track them, but with 10^12 range,
    # collision probability is low.
    
    # Let's run for a fixed number of segments or until time is up.
    
    segments_processed = 0
    max_segments = 100 # Safety limit
    
    while time.time() < DEADLINE - 5.0 and segments_processed < max_segments:
        # Choose a random start
        # Bias towards larger numbers?
        # Let's use log-uniform to cover the space.
        log_s = random.uniform(0, 12)
        s = max(1, int(10**log_s))
        
        # Ensure s + segment_size - 1 <= 10^12
        if s + segment_size - 1 > 10**12:
            s = 10**12 - segment_size + 1
        
        k, best_in_seg = process_segment(s)
        segments_processed += 1
        
        if k > best_k:
            best_k = k
            best_s = best_in_seg
            
        if best_k == 200:
            break

    # Phase 3: If we didn't find a long run, do a quick check of small numbers?
    # The seed found something in small numbers.
    # Let's just check s=1 to 1000 quickly with single phi.
    if best_k < 10:
        seen = set()
        k = 0
        for i in range(1, 1001):
            if time.time() > DEADLINE - 2.0:
                break
            val = phi_single(i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        if k > best_k:
            best_k = k
            best_s = 1

    return {'start': best_s, 'length': best_k}

# EVOLVE-BLOCK-END
