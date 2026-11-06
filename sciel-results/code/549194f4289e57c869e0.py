import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin

    def now():
        return time.time()

    # 1. Precompute primes up to 10^6
    LIMIT = 10**6
    is_prime = bytearray(b'\x01') * (LIMIT + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, LIMIT + 1) if is_prime[i]]

    # 2. Segmented sieve for phi
    def compute_phi_block(start, length):
        # phi[i] will hold phi(start + i)
        phi = list(range(start, start + length))
        
        for p in primes:
            p2 = p * p
            if p2 > start + length:
                # We only need primes up to sqrt(max_n)
                # But actually, if p > sqrt(max_n), it can't be a factor of any n in the block
                # unless n is p itself (which is prime) or n = p * k where k < p.
                # If k < p, then k < sqrt(n). So k would have been processed.
                # So we can break here?
                # Wait. If n = p * q, and p > sqrt(n), then q < sqrt(n).
                # So q would have been processed.
                # The only case where a prime factor > sqrt(n) is not processed is if n is prime.
                # In that case, no p <= sqrt(n) divides n.
                # So yes, we can break when p > sqrt(start + length).
                break
            
            # Find first multiple of p in [start, start + length)
            # m = start + (p - start % p) % p
            # If start % p == 0, m = start
            # Else m = start + (p - start % p)
            
            rem = start % p
            if rem == 0:
                m = start
            else:
                m = start + (p - rem)
            
            # Iterate over multiples
            # Optimization: local variables
            phi_local = phi
            for j in range(m, start + length, p):
                idx = j - start
                phi_local[idx] = phi_local[idx] // p * (p - 1)
        
        # Handle primes: if phi[i] == start + i, then start + i is prime
        # Because no prime factor p <= sqrt(start+i) divided it.
        # (Note: if start+i is composite, it has a factor <= sqrt(start+i), which would have been processed)
        for i in range(length):
            if phi[i] == start + i:
                phi[i] = start + i - 1
                
        return phi

    # 3. Find max distinct run in a list
    def max_distinct_run(vals):
        best_k = 0
        best_start_idx = 0
        n = len(vals)
        
        # Sliding window with a set
        # Since max k is 200, we can just check each start
        # But sliding window is O(N)
        
        seen = {}
        left = 0
        for right in range(n):
            v = vals[right]
            if v in seen:
                # If v is already in the window, move left until it's gone
                # But we need to be careful.
                # Standard approach:
                # If v is in seen, and seen[v] >= left, then we have a duplicate.
                # We must move left to seen[v] + 1.
                if seen[v] >= left:
                    left = seen[v] + 1
            seen[v] = right
            curr_k = right - left + 1
            if curr_k > best_k:
                best_k = curr_k
                best_start_idx = left
                if best_k == 200:
                    # Can't get better
                    return best_k, best_start_idx
                    
        return best_k, best_start_idx

    best_k = 1
    best_s = 1

    # 4. Search Strategy
    # Search near 10^12
    # Block size 10^6
    BLOCK_SIZE = 10**6
    
    # We will search a few blocks near the end
    # Let's start from 10^12 - 10^8 and go backwards
    # Or just pick a few random blocks in [10^11, 10^12]
    
    # To be deterministic and fast, let's just do a few sequential blocks from the end
    # 10^12 is the limit.
    # s + k - 1 <= 10^12.
    # So max start is 10^12 - 199.
    
    # Let's try blocks ending at 10^12, 10^12 - 10^6, etc.
    
    # Number of blocks we can do:
    # Sieve for 10^6 numbers takes ~0.1-0.2s?
    # Let's measure roughly.
    # 78k primes. Average 10 multiples per prime? 780k ops.
    # Python loop overhead is high. Maybe 1s per block?
    # We have 145s. So ~100 blocks.
    # 100 * 10^6 = 10^8 numbers.
    
    # Let's search in the range [10^12 - 10^9, 10^12]
    # That's 1000 blocks of 10^6. Too many.
    # Let's do 50 blocks.
    
    # We'll pick start points spaced out.
    # Or just iterate from the end backwards.
    
    # Let's start from 10^12 - BLOCK_SIZE and go down.
    # But we need to ensure we don't miss the very end.
    
    # Let's just do 20 blocks of 10^6 starting from 10^12 - 20*10^6.
    # And maybe some random ones.
    
    # Actually, let's just do a loop that picks a start, processes, and checks time.
    
    # Candidate starts
    # 1. Near 10^12
    # 2. Near 10^11
    # 3. Random
    
    candidates = []
    # End of range
    for i in range(20):
        s = 10**12 - (i + 1) * BLOCK_SIZE
        if s > 0:
            candidates.append(s)
            
    # Middle
    for i in range(10):
        s = 10**11 - (i + 1) * BLOCK_SIZE
        if s > 0:
            candidates.append(s)
            
    # Random
    for _ in range(10):
        s = random.randint(10**11, 10**12 - BLOCK_SIZE)
        candidates.append(s)
        
    # Process candidates
    for s in candidates:
        if now() > DEADLINE - 5.0:
            break
            
        if s < 1:
            continue
            
        # Ensure s + BLOCK_SIZE - 1 <= 10^12
        if s + BLOCK_SIZE - 1 > 10**12:
            # Adjust block size or start
            # Just skip or adjust
            # Let's just adjust end
            pass # The sieve function handles length, but we fixed length.
            # Better to ensure s is valid.
            if s > 10**12 - BLOCK_SIZE:
                continue
                
        try:
            vals = compute_phi_block(s, BLOCK_SIZE)
        except Exception:
            continue
            
        k, local_start_idx = max_distinct_run(vals)
        
        if k > best_k:
            best_k = k
            best_s = s + local_start_idx
            
            if best_k == 200:
                break
                
    return {'start': best_s, 'length': best_k}
