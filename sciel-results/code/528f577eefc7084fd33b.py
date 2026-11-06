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
    
    # 2. Efficient phi calculation
    def phi(n):
        if n == 1:
            return 1
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
        if temp > 1:
            result -= result // temp
        return result

    # 3. Helper to find max distinct run in a list of phi values
    def max_distinct_run(vals):
        best_k = 0
        best_start_idx = 0
        n = len(vals)
        # We only care up to k=200
        max_k_check = min(200, n)
        
        # Sliding window
        # For each start i, check distinctness of vals[i : i+k]
        # Optimization: 
        # If we know vals[i:i+K] are distinct, then vals[i+1:i+K+1] 
        # only fails if vals[i+K+1] is in vals[i+1:i+K].
        # So we can maintain a set and a count of duplicates?
        # Simpler: Just check window of size up to 200.
        # Since 200 is small, O(N * 200) is fine.
        
        # Actually, we can do better.
        # Let's just iterate start i, and extend k as long as possible.
        # But we want the GLOBAL max in the block.
        
        # Let's use a two-pointer approach or just simple nested loops since K is small.
        # For i in range(n):
        #    seen = set()
        #    for j in range(i, min(i+200, n)):
        #       if vals[j] in seen: break
        #       seen.add(vals[j])
        #       if j - i + 1 > best_k:
        #          best_k = j - i + 1
        #          best_start_idx = i
        #          if best_k == 200: return best_k, best_start_idx # Max possible
        
        # This is O(N * 200). For N=5000, that's 1,000,000 ops. Very fast.
        
        for i in range(n):
            if n - i < best_k:
                break
            seen = set()
            for j in range(i, min(i + 200, n)):
                v = vals[j]
                if v in seen:
                    break
                seen.add(v)
                curr_k = j - i + 1
                if curr_k > best_k:
                    best_k = curr_k
                    best_start_idx = i
                    if best_k == 200:
                        return best_k, best_start_idx
        return best_k, best_start_idx

    best_k = 1
    best_s = 1

    # 4. Search Strategy
    # We will search in blocks.
    # Block size: 5000.
    # Number of blocks we can do depends on time.
    # Let's try specific ranges.
    
    search_ranges = []
    
    # Small range (just in case)
    search_ranges.append((1, 5000))
    
    # Large ranges, spaced out
    # We want to cover the space 10^6 to 10^12 roughly.
    # Let's pick logarithmic steps.
    bases = [10**6, 10**7, 10**8, 10**9, 10**10, 10**11, 10**12]
    for b in bases:
        # Search around b
        # e.g., b - 1000 to b + 10000
        start = max(1, b - 1000)
        end = min(10**12, b + 10000)
        if start < end:
            search_ranges.append((start, end))
            
    # Add some random large ranges
    for _ in range(10):
        start = random.randint(10**8, 10**12 - 10000)
        search_ranges.append((start, start + 10000))

    # Sort ranges to be deterministic? Or just iterate.
    # We should prioritize ranges that are "likely" to have long runs?
    # No strong heuristic. Just spread them out.
    
    # We need to ensure we don't exceed time.
    # Estimate time per block of 5000 numbers.
    # Phi calc: ~50-200 us per number? 
    # 5000 * 200us = 1s.
    # So we can do ~100 blocks in 100s.
    # We have ~20 ranges defined above.
    # We can expand each range or add more random ones.
    
    # Let's refine the search:
    # Instead of fixed ranges, let's do a loop that picks a start point,
    # processes a block of 5000, and if time allows, moves to the next.
    
    # We'll use a generator for start points.
    # 1. Small
    # 2. Log spaced
    # 3. Random
    
    # To maximize coverage, let's just iterate through a list of candidate start points.
    candidates = []
    candidates.append(1)
    for b in [10**6, 10**7, 10**8, 10**9, 10**10, 10**11]:
        candidates.append(b)
        candidates.append(b // 2)
        candidates.append(b * 2)
        
    # Add randoms
    for _ in range(50):
        candidates.append(random.randint(10**7, 10**12 - 10000))
        
    # Shuffle to mix? Or just iterate.
    # Let's iterate.
    
    BLOCK_SIZE = 5000
    
    for s in candidates:
        if now() > DEADLINE - 5.0:
            break
            
        # Ensure s is valid
        if s < 1: s = 1
        if s + BLOCK_SIZE - 1 > 10**12:
            BLOCK_SIZE = 10**12 - s + 1
            if BLOCK_SIZE <= 0: continue
            
        # Compute phis
        try:
            vals = [phi(s + i) for i in range(BLOCK_SIZE)]
        except Exception:
            continue
            
        # Find best run in this block
        k, local_start_idx = max_distinct_run(vals)
        
        if k > best_k:
            best_k = k
            best_s = s + local_start_idx
            
            # If we found 200, we can stop early? 
            # The task says maximize k, max 200.
            # If we hit 200, it's the max possible.
            if best_k == 200:
                break
                
        # We can also try to extend the search if we found a good run near the edge?
        # For now, just move to next candidate.
        
    return {'start': best_s, 'length': best_k}
