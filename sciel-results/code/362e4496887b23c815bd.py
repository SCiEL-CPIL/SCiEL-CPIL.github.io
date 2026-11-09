import time
import math

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # 1. Generate primes up to sqrt(10^12) = 10^6
    limit = 10**6
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]
    
    # 2. Segmented Sieve for Phi
    block_size = 10**6
    best_k = 0
    best_s = 1
    
    # We keep a buffer of phi values from the previous block to handle runs spanning blocks
    # We only need the last best_k values, but best_k can grow.
    # Let's keep a buffer of size 200 (max possible k) or slightly more.
    # Actually, to be safe, we can just store the phi values for the current block
    # and a tail from the previous one.
    
    prev_tail = [] # List of (phi_value, original_index)
    
    current_start = 1
    while current_start <= 10**12:
        if time.time() > deadline:
            break
            
        # Determine block end
        block_end = min(current_start + block_size - 1, 10**12)
        L = block_end - current_start + 1
        
        # Arrays for the block
        phi = [0] * L
        rem = [0] * L
        for i in range(L):
            val = current_start + i
            phi[i] = val
            rem[i] = val
            
        # Process primes
        for p in primes:
            if p * p > block_end:
                break
            # Find first multiple of p in [current_start, block_end]
            first = ((current_start + p - 1) // p) * p
            for m in range(first, block_end + 1, p):
                i = m - current_start
                if rem[i] % p == 0:
                    phi[i] -= phi[i] // p
                    while rem[i] % p == 0:
                        rem[i] //= p
                        
        # Handle remaining prime factors
        for i in range(L):
            if rem[i] > 1:
                phi[i] -= phi[i] // rem[i]
                
        # Combine with previous tail
        # We need to find the longest run of distinct phi values
        # in the sequence: prev_tail + phi
        # But we only care about runs that end in the current block or span into it.
        # Actually, we can just concatenate the values and run the sliding window.
        # To save memory/time, we only keep the last min(len(prev_tail), 200) values?
        # No, the run could start anywhere in prev_tail if it's long enough.
        # But max k is 200. So we only need the last 200 values from prev_tail.
        
        # Let's construct the combined list
        # We need to map indices back to original s
        # prev_tail contains values. We need their s as well?
        # Yes, to report best_s.
        
        # Let's store (value, s) in prev_tail
        # And for current block, we have phi[i] and s = current_start + i
        
        combined_vals = []
        combined_s = []
        
        # Add tail from previous block
        # We only need up to 200 elements from prev_tail
        if len(prev_tail) > 200:
            tail_start = len(prev_tail) - 200
        else:
            tail_start = 0
            
        for idx in range(tail_start, len(prev_tail)):
            combined_vals.append(prev_tail[idx][0])
            combined_s.append(prev_tail[idx][1])
            
        # Add current block
        for i in range(L):
            combined_vals.append(phi[i])
            combined_s.append(current_start + i)
            
        # Sliding window for distinct values
        # We want to find max window length where all values are distinct.
        # And record the start s and length k.
        
        count = {}
        left = 0
        # right will iterate from 0 to len(combined_vals) - 1
        for right in range(len(combined_vals)):
            val = combined_vals[right]
            count[val] = count.get(val, 0) + 1
            
            while count[val] > 1:
                left_val = combined_vals[left]
                count[left_val] -= 1
                left += 1
                
            current_k = right - left + 1
            if current_k > best_k:
                best_k = current_k
                best_s = combined_s[left]
                
        # Update prev_tail for next iteration
        # We need to store (value, s) for the current block's values
        # But we only need the last 200?
        # If the run spans into the next block, it will use values from the end of this block.
        # So we store the last 200 values of the current block.
        
        if L > 200:
            new_tail_start = L - 200
        else:
            new_tail_start = 0
            
        new_tail = []
        for i in range(new_tail_start, L):
            new_tail.append((phi[i], current_start + i))
            
        # Also, if the run started in the previous tail and extends into this block,
        # the "tail" for the NEXT block should effectively be the last 200 values of the CURRENT block.
        # Because any run extending into the next block must end in the current block (or next),
        # and start at least 200 positions before the end of the current block.
        # So storing the last 200 values of the current block is sufficient.
        
        prev_tail = new_tail
        
        current_start = block_end + 1
        
    return {'start': best_s, 'length': best_k}
