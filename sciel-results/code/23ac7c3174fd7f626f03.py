"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import bisect

def construct():
    start_time = time.time()
    # Safety margin: return before 150s wall-clock limit
    DEADLINE = start_time + 120.0
    
    def time_left():
        return DEADLINE - time.time()
        
    def time_exceeded():
        return time.time() >= DEADLINE

    # 1. Generate powerful numbers
    # Powerful numbers n have the form product p_i^e_i with e_i >= 2.
    # We generate them up to a limit L.
    # The seed program used L=20000. The inspiration suggests going higher, e.g. 10^9, but warns about time.
    # Let's try a moderate increase first, say L = 10**7 or 10**8, and see if we can afford it.
    # Actually, the number of powerful numbers up to X is roughly proportional to X^(2/3) / (2 * zeta(3/2))? No, it's ~ C * X^(2/3).
    # For X = 10^6, count is ~ 10^4.
    # For X = 10^9, count is ~ 10^6.
    # Generating 10^6 numbers is fast. Counting pairs for a candidate N by iterating up to N/2 is O(count).
    # If N is small, it's fast.
    
    # Let's aim for L = 10**8.
    # p^2 <= L => p <= 10^4.
    # Primes up to 10^4 are ~1229.
    
    def get_primes(limit):
        if limit < 2: return []
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i in range(2, limit + 1) if sieve[i]]

    L = 10**8
    primes = get_primes(int(L**0.5) + 1)
    
    powerful_set = set([1])
    
    # DFS to generate powerful numbers using primes
    # State: (index_in_primes, current_value)
    # We only use primes up to sqrt(L) for the "base" generation?
    # No, a powerful number can be p^2 where p > sqrt(L) is impossible since p^2 > L.
    # So all prime factors p of a powerful number <= L must satisfy p^2 <= L => p <= sqrt(L).
    # Thus, we only need primes up to sqrt(L).
    
    def generate_powerful(idx, val):
        if time_exceeded():
            return
        if val > L:
            return
        if val > 1:
            powerful_set.add(val)
        
        # Try adding more primes from idx onwards
        for i in range(idx, len(primes)):
            p = primes[i]
            next_val = val * p * p
            if next_val > L:
                break
            generate_powerful(i + 1, next_val)
            # Increase power of p
            while next_val * p <= L:
                next_val *= p
                if time_exceeded():
                    return
                powerful_set.add(next_val)
                generate_powerful(i + 1, next_val)
                if time_exceeded():
                    return

    # Start with 1
    generate_powerful(0, 1)
    
    if time_exceeded():
        # Fallback to a smaller set if generation took too long (unlikely for 10^8)
        # But let's just proceed with what we have.
        pass

    pn_list = sorted(powerful_set)
    pn_set = powerful_set
    
    # 2. Find Best N
    # We want to maximize the number of pairs (a, b) such that a+b=N, a,b in pn_set, a<=b.
    # Strategy from inspiration:
    # - Generate candidates from sums of small powerful numbers.
    # - Sort candidates.
    # - Count pairs for each candidate.
    
    candidates = set()
    
    # Add small N values
    # The seed program searched N up to 20000.
    # Let's search a bit wider, but prioritize density.
    # Powerful numbers are dense at the bottom.
    # Let's take the first K powerful numbers and generate sums.
    
    K = 500 # 500 smallest powerful numbers
    small_pn = pn_list[:K]
    
    for i in range(len(small_pn)):
        for j in range(i, len(small_pn)):
            s = small_pn[i] + small_pn[j]
            if s <= L: # Keep N within our generated range for efficient lookup?
                # Actually N can be up to 2*L. But if N > L, then b = N-a > L-a.
                # If a is small, b might be > L, so b won't be in pn_set (since we only generated up to L).
                # So effectively, for a pair to be valid, both a and b must be <= L.
                # Thus N <= 2L.
                # But if N is large, counting is slow because we iterate up to N/2.
                # However, if N/2 > L, we only iterate up to L (since pn_list only goes to L).
                # So counting cost is O(min(N/2, L)).
                candidates.add(s)
    
    # Also, maybe check some larger N?
    # The score is the count. Small N might have more representations because the numbers are small and dense.
    # Let's stick to candidates derived from small powerful numbers.
    
    candidate_list = list(candidates)
    candidate_list.sort()
    
    best_N = 2
    best_count = 1
    best_pairs = [[1, 1]]
    
    # Optimization for counting:
    # For a given N, we iterate a in pn_list where a <= N/2.
    # We check if N-a is in pn_set.
    
    def count_pairs(N):
        half = N // 2
        # Find index of last element <= half
        idx = bisect.bisect_right(pn_list, half)
        count = 0
        pairs = []
        # Iterate only up to idx
        for i in range(idx):
            a = pn_list[i]
            b = N - a
            if b in pn_set:
                count += 1
                pairs.append([a, b])
        return count, pairs

    # Evaluate candidates
    # We might have many candidates.
    # Let's limit the number we check if time is tight.
    # But counting is fast for small N.
    
    # Sort candidates by N? Or by potential?
    # Small N is likely better.
    
    tried = 0
    max_tries = 10000 # Adjust based on time
    
    for N in candidate_list:
        if time_left() < 10:
            break
        if tried >= max_tries:
            break
            
        cnt, pairs = count_pairs(N)
        if cnt > best_count:
            best_count = cnt
            best_N = N
            best_pairs = pairs
            # If we find a very high count, we might continue to see if we can do better
            # But small N are generally better.
            
        tried += 1
        
    # If we didn't find anything better, return the seed's best or the default
    # The seed found a best in range(2, 20001).
    # Our candidate list includes sums of small pn, which covers small N.
    
    return {'N': best_N, 'pairs': best_pairs}

# EVOLVE-BLOCK-END
