import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 100.0  # Leave 20s margin for 120s limit
    
    best_k = 1
    best_s = 1
    
    # We will search in segments near 10^12
    # The range of interest is up to 10^12
    max_n = 10**12
    
    # Segment size for the sieve
    # Larger segments allow finding longer runs without restarting, but use more memory.
    # 200,000 is a good balance.
    seg_size = 200_000
    
    # We need primes up to sqrt(max_n) for the sieve.
    # sqrt(10^12) = 10^6.
    # We generate primes up to 10^6 using a simple sieve.
    limit_primes = 10**6
    is_prime = np.ones(limit_primes + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit_primes**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.array([i for i, p in enumerate(is_prime) if p])
    
    # Function to compute phi for a segment [start, start + seg_size)
    def compute_phi_segment(start, size):
        # Initialize phi values with the numbers themselves
        # We use int64 to hold numbers up to 10^12
        phi = np.arange(start, start + size, dtype=np.int64)
        
        # For each prime p, find multiples of p in the segment
        # and update phi[n] = phi[n] / p * (p-1)
        for p in primes:
            if p * p > start + size:
                break
                
            # Find the first multiple of p in [start, start + size)
            # first_multiple = ceil(start / p) * p
            first_multiple = ((start + p - 1) // p) * p
            
            # Iterate through multiples of p
            for m in range(first_multiple, start + size, p):
                idx = m - start
                if phi[idx] % p == 0:
                    phi[idx] = (phi[idx] // p) * (p - 1)
                    
        return phi

    # We will pick random segments to search
    # To maximize the chance of finding a long run, we search in random locations
    # but we must ensure we cover the range efficiently.
    
    num_segments_to_try = 20
    
    for i in range(num_segments_to_try):
        if time.time() - start_time > deadline:
            break
            
        # Choose a random start position for the segment
        # We want to stay within [1, 10^12]
        # Let's bias towards higher numbers where collisions might be rarer?
        # Actually, for distinct phi values, the structure of n matters.
        # Random sampling is a good start.
        
        min_start = 1
        max_start = max_n - seg_size
        
        if max_start < min_start:
             max_start = min_start
             
        seg_start = random.randint(min_start, max_start)
        
        # Compute phi for this segment
        phi_vals = compute_phi_segment(seg_start, seg_size)
        
        # Now find the longest run of distinct phi values in this segment
        # We are looking for a contiguous subsequence of length k <= 200
        # such that all phi values are distinct.
        
        # We can use a sliding window approach.
        # However, checking distinctness for every window of size 200 is O(N * 200).
        # Since seg_size is 200,000, this is 40M operations, which is fine in Python if optimized,
        # but let's use a smarter check or just iterate.
        
        # Actually, we want the MAX k.
        # We can check for k=200 first. If found, we can return immediately (or update best).
        # If not, check k=199, etc.? No, that's expensive.
        
        # Better: Scan for the longest run.
        # We can maintain a set of phi values in the current window.
        # But we want the GLOBAL max in this segment.
        
        # Let's just check for the maximum possible length 200 first.
        # If we find a run of 200, it's the best possible.
        
        # We will check all windows of size 200.
        # If any window has distinct values, best_k = 200.
        
        found_200 = False
        if seg_size >= 200:
            for j in range(0, seg_size - 200 + 1):
                window = phi_vals[j:j+200]
                if len(set(window)) == 200:
                    found_200 = True
                    best_k = 200
                    best_s = seg_start + j
                    # Since 200 is the max, we can break out of everything
                    return {'start': best_s, 'length': best_k}
            
        # If we didn't find 200, we should try to find the next best.
        # But doing a full longest-subsequence search for distinct values is complex.
        # Given the rarity of long runs, finding 200 is the main goal.
        # If we don't find 200, we might find 199, 198, etc.
        
        # For efficiency, let's just check for lengths 199, 198, ... down to some limit?
        # Or just accept that if we don't find 200 in this segment, we move to the next.
        # The probability of finding a run of 200 is low, but non-zero.
        
        # To improve the score, we should at least record the best k found if it's > best_k.
        # Let's do a simple scan for the longest run of distinct phi values in this segment.
        # We can use a two-pointer technique with a frequency array?
        # The phi values can be large (up to 10^12), so a hash map is needed.
        
        # Let's try to find runs of length > best_k.
        # We'll just check windows of size best_k + 1.
        # If we find one, we increment best_k and repeat?
        # This is O(N * (200 - best_k)).
        
        # Given the time limit, let's just check for 200.
        # If we don't find 200, we'll just keep searching random segments.
        # This might result in best_k staying at 1 if we are unlucky.
        
        # To be safer, let's check for smaller runs if best_k is small.
        # But the task says "Maximize k".
        
        # Let's add a check for k=100, 50, etc. if best_k is low.
        # Actually, let's just check for the specific length 200.
        # If we find it, we return.
        # If we don't, we continue.
        
        # To avoid returning k=1 if we find nothing, we should ensure we at least return a valid result.
        # The initial best_k=1 is valid.
        
        # Let's add a bit more logic: if we haven't found 200, check for 199?
        # No, that's too slow.
        
        # Let's stick to checking for 200.
        # The "Knowledge" says: "Vectorized segmented phi sieve... finding runs far beyond the seed's 20000 limit".
        # Wait, the seed limit was 20000? The task says k <= 200.
        # The knowledge might be from a different context or a misinterpretation.
        # The task explicitly says 1 <= k <= 200.
        
        # So finding 200 is the maximum possible score improvement (from 1 to 200).
        
        # If we don't find 200, we might find 199.
        # Let's check for 199 if 200 is not found.
        
        if not found_200 and seg_size >= 199:
             for j in range(0, seg_size - 199 + 1):
                window = phi_vals[j:j+199]
                if len(set(window)) == 199:
                    if 199 > best_k:
                        best_k = 199
                        best_s = seg_start + j
                    # Don't break, we might find 200? No, we already checked 200.
                    # But we might find another 199? Doesn't matter, k is the same.
                    # We can break to save time.
                    break
             
        # Check for 198?
        if best_k < 198 and seg_size >= 198:
             for j in range(0, seg_size - 198 + 1):
                window = phi_vals[j:j+198]
                if len(set(window)) == 198:
                    if 198 > best_k:
                        best_k = 198
                        best_s = seg_start + j
                    break

        # We can continue checking lower lengths if time permits, but 200 is the main target.
        
    return {'start': best_s, 'length': best_k}
