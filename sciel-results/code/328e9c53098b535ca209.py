import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    primes_np = np.array(primes)
    
    # Helper to calculate the set of distinct small primes dividing a number
    def get_small_primes(x):
        s = set()
        for p in primes:
            if x % p == 0:
                s.add(p)
        return s
    
    # Helper to evaluate a starting index n
    # Returns the number of distinct small primes across all n+i (i=1..24)
    def evaluate(n):
        all_primes = set()
        for i in range(1, 25):
            all_primes.update(get_small_primes(n + i))
        return len(all_primes)
    
    best_n = 1000
    best_score = evaluate(best_n)
    
    # Strategy 1: Random sampling with local refinement
    # The space is 1000..1000000. 
    # We know the minimum possible is at least 2 (since we have 24 numbers, 
    # it's impossible for all to be divisible by only {2} or only {3}, etc., 
    # and likely impossible to have only 2 distinct primes across 24 consecutive ints 
    # unless we are very lucky, e.g., all even and all multiples of 3? 
    # No, consecutive ints cannot all be even. 
    # The distinct primes in the window will likely be around 4-6.
    
    # Let's try to find a window where the union of prime factors is small.
    
    # 1. Check a few known "good" structures or just random spots
    # Random sampling
    num_samples = 5000
    for _ in range(num_samples):
        if time.time() > deadline:
            break
            
        n = random.randint(1000, 1000000)
        score = evaluate(n)
        
        if score < best_score:
            best_score = score
            best_n = n
            
            # Local search around this new best
            # If we found a good one, check neighbors
            for delta in range(-10, 11):
                if time.time() > deadline:
                    break
                cand = n + delta
                if 1000 <= cand <= 1000000:
                    s = evaluate(cand)
                    if s < best_score:
                        best_score = s
                        best_n = cand
    
    # 2. Targeted search around multiples of small primorials
    # Numbers like 2*3*5*7 = 210. 
    # A window centered near a multiple of 210 might have high divisibility by 2,3,5,7.
    # Let's check windows starting near multiples of 210, 630, etc.
    
    step = 210
    # We can't iterate all multiples of 210 up to 1e6 (approx 4760 iterations) if it's fast enough.
    # 4760 * 24 * 9 ops is small. Let's do it if time permits.
    
    if time.time() < deadline - 10.0:
        # Iterate through multiples of 210 in the range
        # n+i for i in 1..24. 
        # If n is a multiple of 210, n+1..n+24 covers residues 1..24 mod 210.
        # This doesn't guarantee low distinct primes, but it's a structured search.
        
        # Better: Just iterate n from 1000 to 1000000 in steps? No, too slow.
        # Let's just do a linear scan if we have enough time, but optimized.
        # Actually, random + local search is usually sufficient for these "minimize distinct factors" tasks
        # because the landscape is relatively flat with occasional dips.
        
        # Let's try a denser random search in the first 100k and last 100k? 
        # No, uniform is fine.
        
        # Let's try to be smarter: 
        # The score is the count of distinct primes. 
        # Minimum possible is 2? 
        # Can we have 24 consecutive integers such that every integer is divisible by 2 or 3?
        # No, there are numbers coprime to 6.
        # Can we have 24 consecutive integers such that every integer is divisible by 2, 3, or 5?
        # The Jacobsthal function g(k) is the maximum gap between integers coprime to the product of first k primes.
        # g(3) for 2,3,5 is 10. So any window of length 10 has a number coprime to 30.
        # g(4) for 2,3,5,7 is 22. So any window of length 22 has a number coprime to 210.
        # g(5) for 2,3,5,7,11 is 34.
        # So in a window of 24, there MUST be at least one number coprime to 2*3*5*7 (210).
        # That number will have prime factors >= 11.
        # So the distinct primes in the window must include at least one prime >= 11.
        # Plus the small primes dividing the other numbers.
        # So minimum distinct primes is likely >= 3 or 4.
        
        # Let's just run the random search longer if we have time.
        for _ in range(20000):
            if time.time() > deadline:
                break
            n = random.randint(1000, 1000000)
            score = evaluate(n)
            if score < best_score:
                best_score = score
                best_n = n
                # Quick local refine
                for delta in range(-5, 6):
                    if time.time() > deadline:
                        break
                    cand = n + delta
                    if 1000 <= cand <= 1000000:
                        s = evaluate(cand)
                        if s < best_score:
                            best_score = s
                            best_n = cand

    # 3. Structured search using sieve-like approach
    # Let's try to find a window where the numbers are heavily divisible by small primes.
    # We can precompute for a range of numbers their small prime factors.
    
    if time.time() < deadline - 20.0:
        # Precompute small prime factors for a large block
        # We'll process in blocks to keep memory reasonable
        block_size = 100000
        num_blocks = (1000000 - 1000 + 1) // block_size + 1
        
        for block_idx in range(num_blocks):
            if time.time() > deadline:
                break
                
            block_start = 1000 + block_idx * block_size
            block_end = min(block_start + block_size, 1000001)
            
            # Create arrays for the block
            block_range = np.arange(block_start, block_end)
            
            # For each prime, mark which numbers in the block are divisible by it
            # We want to find a window of 24 consecutive numbers where the union of primes is small.
            
            # Instead of storing all factors, let's compute a "signature" for each number.
            # Signature is a bitmask of which primes divide it.
            signatures = np.zeros(len(block_range), dtype=np.int32)
            
            for i, p in enumerate(primes):
                mask = (block_range % p == 0)
                signatures[mask] |= (1 << i)
            
            # Now for each position j in the block, we want the bitwise OR of signatures[j:j+24]
            # (corresponding to numbers block_start+j to block_start+j+23)
            # Note: we need numbers n+1 to n+24. 
            # If n is in the block, n+1 is at index 1, ..., n+24 is at index 24.
            # So we need to slide a window of width 24 over the signatures.
            
            # We can use a sliding window OR.
            # However, OR is not invertible like sum. 
            # But we can compute it in O(N * log(P)) or just O(N * P) which is fine.
            # Actually, since P=9, we can just loop.
            
            # But wait, we need to count the number of bits set in the OR.
            
            # Let's try a different approach:
            # For each prime p, let S_p be the set of indices in the block divisible by p.
            # We want a window [k, k+23] that covers indices in the block.
            # The number of distinct primes is the count of p such that S_p intersects [k, k+23].
            
            # This is equivalent to: for each p, find the maximum gap between consecutive multiples of p.
            # If the gap > 24, then there exists a window of length 24 that does NOT contain a multiple of p.
            # We want to find a window that misses as many primes as possible.
            
            # This is getting complicated. Let's stick to the simpler approach:
            # Iterate over possible start positions in the block and compute the OR.
            
            # Optimization: Use numpy to compute the OR for all windows at once?
            # Not directly. But we can do it in a loop over the block, which is 100k iterations.
            # 100k * 9 ops is 900k ops, which is fast.
            
            for j in range(0, len(block_range) - 24):
                # Window corresponds to n = block_start + j - 1 ? No.
                # Number n+i corresponds to index j+i-1 in block_range if block_start = n?
                # Let's be careful.
                # block_range[k] = block_start + k.
                # We want n such that n+1 = block_start + j. So n = block_start + j - 1.
                # The numbers are n+1, ..., n+24 => block_start+j, ..., block_start+j+23.
                # Indices in block_range: j, j+1, ..., j+23.
                
                n = block_start + j - 1
                if n < 1000 or n > 1000000:
                    continue
                    
                # Compute OR of signatures[j:j+24]
                # We can do this efficiently
                current_or = 0
                for k in range(24):
                    current_or |= signatures[j+k]
                    
                num_distinct = bin(current_or).count('1')
                
                if num_distinct < best_score:
                    best_score = num_distinct
                    best_n = n
                    
                    # Local refine
                    for delta in range(-5, 6):
                        cand = n + delta
                        if 1000 <= cand <= 1000000:
                            s = evaluate(cand)
                            if s < best_score:
                                best_score = s
                                best_n = cand

    return best_n
