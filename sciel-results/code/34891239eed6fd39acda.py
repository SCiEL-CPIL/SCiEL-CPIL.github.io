import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s safety margin for 150s limit
    
    # 1. Generate primes up to 10^6 for the segmented sieve and initial checks
    #    We need primes up to sqrt(max_n). max_n ~ (10^6)^2 = 10^12, sqrt is 10^6.
    #    If we search p up to 10^6, n ~ 10^12.
    #    If we search p up to 10^7, n ~ 10^14, sqrt is 10^7.
    #    Let's aim for p up to 10^6 first, maybe higher if fast.
    #    Primes up to 10^6 are sufficient for sieving n up to 10^12.
    #    For n > 10^12, we need primes up to sqrt(n).
    #    Let's generate primes up to 2 * 10^6 to be safe for p up to ~2 * 10^6.
    #    Actually, if p is large, say 10^7, n is 10^14. sqrt(n) = 10^7.
    #    Generating primes up to 10^7 takes ~1s and ~10MB.
    #    Let's try 10^6 first to keep memory low and speed high, then expand if time permits?
    #    No, let's just do 10^6 for the sieve base. 
    #    If p > 10^6, we might miss factors in [10^6, p] if we only sieve with primes <= 10^6.
    #    Wait, the segmented sieve uses primes up to sqrt(max_n).
    #    If max_n = p^2 + 200, sqrt(max_n) ~ p.
    #    So we need primes up to p for the sieve.
    #    If we search p up to 10^6, we need primes up to 10^6.
    #    If we search p up to 10^7, we need primes up to 10^7.
    
    # Let's generate primes up to 10^6 first. This allows searching p up to 10^6.
    # If we find a good k, we might try larger p.
    # But generating primes up to 10^7 is fast enough (~1-2s).
    # Let's generate up to 10^6 to be safe on memory and time, and search p in [1000, 10^6].
    # Actually, the probability of long runs increases with p.
    # Let's generate primes up to 2 * 10^6.
    
    limit_small = 2 * 10**6
    sieve = [True] * (limit_small + 1)
    sieve[0:2] = [False, False]
    for i in range(2, int(math.isqrt(limit_small)) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    small_primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    # Helper: Check if a number is p-smooth given its SPF table
    # The SPF table is for the range [p^2+1, p^2+k]
    
    best_k = 0
    best_p = 3
    
    # Search primes. 
    # Strategy: Search larger primes first? 
    # Larger p -> larger n -> harder to be p-smooth? 
    # Actually, for fixed k, as p increases, the numbers p^2+i get larger.
    # The condition is "all prime factors <= p".
    # For small p, p^2 is small, factors are small. 
    # Example: p=3, 9+1=10 (2,5). 5>3. Fail.
    # p=5, 25+1=26 (2,13). 13>5. Fail.
    # p=11, 121+1=122 (2,61). 61>11. Fail.
    # It seems small p often fail quickly.
    # Larger p allows larger prime factors.
    # So we should search larger p first to find long runs faster?
    # Or maybe there's a sweet spot.
    # Let's iterate primes in descending order from limit_small down to some min.
    # But we must ensure we have primes up to p for the sieve.
    # small_primes contains primes up to 2*10^6.
    # So we can sieve for p up to 2*10^6.
    
    # Let's search p in [10^5, 2*10^6] first, descending.
    # If time remains, go lower.
    
    # To be efficient, we can skip small p where k is unlikely to be high.
    # Let's just iterate all primes in small_primes in reverse, but stop if time is up.
    
    # Optimization: 
    # For a given p, we create an array of size k=200 (or until break).
    # We sieve this array with all primes <= p.
    # Then we check smoothness.
    
    # We need to be careful with memory and time for the sieve.
    # Sieve size 200 is tiny.
    # Number of primes <= p is ~ p/log(p). For p=10^6, ~78k primes.
    # For each prime q, we mark multiples in the range [p^2+1, p^2+200].
    # The number of multiples is ~ 200/q.
    # Total operations ~ sum(200/q) for q <= p.
    # This is 200 * sum(1/q) ~ 200 * log(log(p)).
    # For p=10^6, log(log(10^6)) is small.
    # So sieving is very fast.
    
    # The bottleneck is generating the list of primes and the Python loop overhead.
    
    # Let's iterate primes in descending order.
    # We'll start from the largest prime in small_primes.
    
    # We need to handle the case where p > limit_small?
    # No, we limited small_primes to 2*10^6.
    # So max p is 2*10^6.
    
    # Let's refine the search range.
    # Maybe p around 10^5 - 10^6 is good.
    
    # We'll iterate from the end of small_primes.
    # But small_primes has 10^6 primes? No, ~148k primes up to 2*10^6.
    # Iterating 148k primes is fine.
    
    # However, we can add a quick check.
    # If p is very small, k is likely small.
    
    for i in range(len(small_primes) - 1, -1, -1):
        if time.time() > deadline:
            break
            
        p = small_primes[i]
        
        # Optional: skip if p is too small and we already have a good k?
        # If best_k is high, small p might not beat it.
        # But we don't know the distribution.
        # Let's just run.
        
        # Construct the segment
        # We want to check p^2 + 1, ..., p^2 + 200
        # But we can stop early if we find a non-smooth number.
        # However, to use the sieve efficiently, we sieve the whole block of 200.
        # Then we check sequentially.
        
        base = p * p
        k_max = 200
        
        # Initialize SPF array for the segment
        # spf[j] stores the smallest prime factor found so far for base + j
        # If spf[j] is 0, it means no factor found yet (or 1? no, 1 has no factors)
        # Actually, we want to know if ALL factors are <= p.
        # So we divide out all factors <= p.
        # If the remaining number is 1, it's p-smooth.
        
        # Let's use an array `rem` where rem[j] = base + j
        # We divide rem[j] by all primes q <= p.
        
        # To do this efficiently with a sieve-like approach:
        # For each prime q <= p:
        #   Find the first multiple of q in [base+1, base+k_max]
        #   Mark all multiples.
        
        # But we need to divide, not just mark.
        # So we maintain `rem[j]`.
        # For each q, for each j where (base+j) % q == 0:
        #   while rem[j] % q == 0: rem[j] //= q
        
        # This is efficient if we iterate q and find the starting offset.
        
        rem = [base + j for j in range(1, k_max + 1)]
        
        # Iterate primes q <= p
        # We have small_primes. We need q <= p.
        # Since we are iterating p in descending order, and small_primes is sorted,
        # we can find the index of p in small_primes.
        # i is the index of p.
        # So we iterate small_primes[0 : i+1].
        
        # But iterating 0..i for each p is O(N^2) if we do it naively?
        # No, the inner loop is over q.
        # Total work is sum over p of (number of q <= p * cost per q).
        # This might be too slow if we do this for 100k primes.
        
        # Alternative:
        # Only sieve with primes up to some limit?
        # No, we need to remove all factors <= p.
        
        # Let's optimize the inner loop.
        # For a fixed p, the cost is sum_{q<=p} (k_max / q).
        # This is ~ k_max * log(log(p)).
        # For k_max=200, this is ~ 200 * 3 = 600 operations per p?
        # Wait, the loop `while rem[j] % q == 0` adds cost.
        # But rem[j] decreases quickly.
        
        # Let's try this approach and see if it's fast enough.
        
        # To speed up, we can break early in the q loop if q*q > max(rem)?
        # No, we need to remove ALL factors <= p.
        # Even if q*q > rem[j], rem[j] might be a prime <= p.
        # But if we remove all factors q < sqrt(rem[j]), the remainder is 1 or prime.
        # If remainder is prime, is it <= p?
        # We don't know without checking.
        # But if we iterate all q <= p, we are safe.
        
        # Let's implement the sieve division.
        
        # Precompute start indices for each q? No, depends on base.
        
        # For each q in small_primes[0:i+1]:
        #   # Find first j in 1..k_max such that (base+j) % q == 0
        #   # (base+j) % q == 0 => j % q == (-base) % q
        #   rem_base_mod_q = (-base) % q
        #   start_j = rem_base_mod_q
        #   if start_j == 0:
        #       start_j = q # j must be >= 1? 
        #       # Wait, j is offset 1..k_max.
        #       # If rem_base_mod_q == 0, then base % q == 0.
        #       # Then base + q is divisible by q.
        #       # So start_j = q.
        #       # But what if q > k_max? Then no multiples in range.
        #       # If q <= k_max, start_j = q.
        #       # If q > k_max, no start.
        #   else:
        #       start_j = rem_base_mod_q
        #       # Check if start_j >= 1
        #       if start_j < 1:
        #           start_j += q # This shouldn't happen if we take mod q
            
        #   Actually, simpler:
        #   first = base + 1
        #   # Find smallest j >= 1 such that (base+j) % q == 0
        #   # (base+j) % q == 0
        #   # j % q == (-base) % q
        #   r = (-base) % q
        #   if r == 0:
        #       start = q
        #   else:
        #       start = r
        #       
        #   if start > k_max:
        #       continue
        #       
        #   for j in range(start, k_max + 1, q):
        #       while rem[j-1] % q == 0: # rem is 0-indexed, j is 1-indexed offset
        #           rem[j-1] //= q
        
        # This loop runs for each q.
        # Number of q is ~ p/log(p).
        # For p=10^6, 78k primes.
        # Inner loop runs k_max/q times.
        # Total operations ~ 78k * (200 * harmonic series part).
        # This is roughly 78k * 200 * log(10^6) / log(log(10^6))? No.
        # Sum_{q<=p} 200/q ~ 200 * log(log(p)).
        # So total operations per p is ~ 200 * log(log(p)).
        # For p=10^6, log(log(10^6)) ~ log(13.8) ~ 2.6.
        # So ~ 520 divisions per p?
        # Plus the overhead of the Python loops.
        
        # Let's try it.
        
        # Optimization: Use local variables for speed.
        rem_list = rem
        k_max_val = k_max
        base_val = base
        
        # We need to iterate primes <= p.
        # small_primes is a list.
        # We can slice it? No, slice copies.
        # Use index.
        
        # But wait, if we do this for 100k primes, it might be slow.
        # 100k * 500 ops = 50 million ops.
        # Python can do ~10-50 million simple ops per second.
        # So ~1-5 seconds.
        # This is acceptable!
        
        # Let's implement.
        
        # One more optimization: 
        # If rem[j-1] becomes 1, we don't need to divide further for that j.
        # But the sieve structure divides all j for a given q.
        # We can't easily skip j inside the q loop without breaking the sieve pattern.
        # But we can check `if rem_list[j-1] > 1` before the while loop.
        
        # Also, if q > p, we stop.
        
        # Let's run the sieve.
        
        # We only need primes up to p.
        # i is the index of p in small_primes.
        # So we iterate small_primes[0 : i+1].
        
        # To avoid slicing, use a for loop with index.
        
        # But we are iterating p in descending order.
        # So i goes from len(small_primes)-1 down to 0.
        
        # For each p, we iterate q from small_primes[0] to small_primes[i].
        # This is O(i) primes.
        
        # Total work: Sum_{i} (Cost for p_i).
        # Cost for p_i ~ log(log(p_i)).
        # Sum is ~ N * log(log(p_max)).
        # N ~ 150k.
        # Total ~ 150k * 3 = 450k iterations of the outer q loop?
        # No, the outer q loop runs i times.
        # Sum_{i=1}^{N} i ~ N^2/2.
        # This is 150k^2 / 2 = 10^10.
        # This is TOO SLOW.
        
        # We need a different approach.
        
        # We cannot re-sieve from scratch for each p.
        
        # Alternative:
        # Only check a subset of p?
        # Or use a faster check?
        
        # What if we just check p-smoothness for a few random p?
        # Or use the property that if p is large, k is likely small?
        
        # Let's rethink.
        # The task is to find p, k.
        # We can just pick a good p and verify.
        # But we need to MAXIMIZE k.
        
        # Maybe we should search p in a smarter way.
        # Or use a different verification method.
        
        # What if we use the fact that for a fixed k, we can sieve for p?
        # No, p is the variable.
        
        # Let's go back to the "descending p" idea but limit the search.
        # Maybe only check the top 1000 primes?
        # Or check primes in a specific range?
        
        # Let's try checking the largest 1000 primes first.
        # If we find a good k, we stop.
        
        # Also, we can optimize the inner loop by not iterating all q <= p.
        # We only need to divide by q if q <= p.
        # But we also need to ensure no factor > p.
        
        # Actually, if we divide by all q <= p, and rem > 1, then rem has a factor > p.
        # So we just need to divide by all q <= p.
        
        # The cost is dominated by the number of q.
        
        # Let's try a hybrid:
        # 1. Try a few large primes with full verification.
        # 2. If time permits, try more.
        
        # Let's try the top 100 primes.
        
        # But wait, the "inspiration" said "descending order finds longer runs faster".
        # So let's try the top 1000 primes.
        
        # Cost: 1000 * (cost per p).
        # Cost per p ~ 200 * log(log(p)).
        # ~ 1000 * 600 = 600,000 ops.
        # This is fast.
        
        # Let's do this.
        
        # We'll iterate the last 1000 primes in small_primes.
        
        num_candidates = 1000
        start_idx = max(0, len(small_primes) - num_candidates)
        
        for i in range(len(small_primes) - 1, start_idx - 1, -1):
            if time.time() > deadline:
                break
                
            p = small_primes[i]
            
            base = p * p
            k_max = 200
            rem = [base + j for j in range(1, k_max + 1)]
            
            # Sieve with primes <= p
            # We need primes up to p.
            # i is the index of p.
            # So we iterate small_primes[0 : i+1].
            
            # To speed up, we can precompute the list of primes <= p?
            # No, slicing is slow.
            
            # Let's just iterate.
            
            # Optimization: If q > p, break.
            # But we are iterating up to i, so q <= p is guaranteed.
            
            for q in small_primes[:i+1]:
                # Find start j
                r = (-base) % q
                if r == 0:
                    start = q
                else:
                    start = r
                    
                if start > k_max:
                    continue
                    
                for j in range(start, k_max + 1, q):
                    idx = j - 1
                    if rem[idx] > 1:
                        while rem[idx] % q == 0:
                            rem[idx] //= q
            
            # Check smoothness
            k = 0
            for j in range(1, k_max + 1):
                if rem[j-1] == 1:
                    k += 1
                else:
                    # rem > 1
                    # Check if rem <= p
                    if rem[j-1] <= p:
                        k += 1
                    else:
                        break
                        
            if k > best_k:
                best_k = k
                best_p = p
                
            # Early exit if we found k=200?
            if best_k == 200:
                break
                
    return {'p': best_p, 'k': best_k}
