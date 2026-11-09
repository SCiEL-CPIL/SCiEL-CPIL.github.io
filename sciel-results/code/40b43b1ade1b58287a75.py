import numpy as np
import time

def construct():
    def get_primes(limit):
        sieve = np.ones(limit + 1, dtype=bool)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = False
        return np.where(sieve)[0]

    # Primes up to 10^6 (sqrt of 10^12)
    primes = get_primes(10**6)
    
    best_k = 1
    best_s = 1
    
    SEG_SIZE = 10**6 # Smaller segment to fit in memory/cache better? 10^6 is 8MB.
    MAX_N = 10**12
    start_time = time.time()
    
    L = 1
    prev_tail = []
    prev_R = 0
    T = 200 # Tail length to keep
    
    while L <= MAX_N:
        if time.time() - start_time > 110:
            break
            
        R = min(L + SEG_SIZE - 1, MAX_N)
        N = R - L + 1
        
        # Compute phi for [L, R]
        phi_arr = np.arange(L, R + 1, dtype=np.int64)
        
        for p in primes:
            if p > R:
                break
            # Find first multiple of p in [L, R]
            start = ((L + p - 1) // p) * p
            if start > R:
                continue
            offset = start - L
            last = R - (R % p)
            if last < start:
                continue
            count = (last - start) // p + 1
            # Update phi
            # phi[x] = phi[x] / p * (p-1)
            # Ensure integer division
            phi_arr[offset : offset + count*p : p] //= p
            phi_arr[offset : offset + count*p : p] *= (p - 1)
        
        # Combine with previous tail
        # prev_tail contains phi values for numbers ending at prev_R
        # We need to map indices correctly
        # To avoid creating a huge list, we process the tail and then the array
        # But the sliding window logic needs a sequence.
        
        # Let's create a list of values for the combined window
        # Since T is small (200) and SEG_SIZE is 10^6, the list is ~10^6.
        # Creating the list is O(N).
        
        # Optimization: We can just iterate over the combined sequence without explicit list creation if we handle indices carefully.
        # But `prev_tail` is a list, `phi_arr` is an array.
        
        full_vals = prev_tail + phi_arr.tolist()
        
        # Sliding window for distinct values
        seen = {}
        left = 0
        local_max_k = 0
        local_max_start = 0
        
        for right, val in enumerate(full_vals):
            if val in seen:
                left = max(left, seen[val] + 1)
            seen[val] = right
            curr_k = right - left + 1
            if curr_k > local_max_k:
                local_max_k = curr_k
                local_max_start = left
        
        # Determine actual start number
        if local_max_k > best_k:
            # Calculate actual start
            if local_max_start < len(prev_tail):
                # Start is in prev_tail
                # prev_tail[i] corresponds to prev_R - (len(prev_tail) - 1) + i
                actual_start = prev_R - (len(prev_tail) - 1) + local_max_start
            else:
                # Start is in curr_vals
                idx_in_curr = local_max_start - len(prev_tail)
                actual_start = L + idx_in_curr
            
            # Verify constraints
            if actual_start >= 1 and actual_start + local_max_k - 1 <= MAX_N and local_max_k <= 200:
                best_k = local_max_k
                best_s = actual_start
                
        # Update tail
        # We need the last T phi values from the combined sequence
        if len(full_vals) > T:
            prev_tail = full_vals[-T:]
        else:
            prev_tail = full_vals
            
        prev_R = R
        L = R + 1
        
    return {'start': best_s, 'length': best_k}
