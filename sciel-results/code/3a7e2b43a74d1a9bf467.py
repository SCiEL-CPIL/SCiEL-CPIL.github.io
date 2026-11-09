"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def construct():
    start_time = time.time()
    DEADLINE = 115  # Safety margin
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
        
    LIMIT = 10**18
    
    # 1. Generate Powerful Numbers
    # Powerful numbers are of the form p1^e1 * ... * pk^ek where ei >= 2.
    # We can generate them by iterating over primes and using a DFS/Stack.
    # To be fast, we limit the primes.
    # p^2 <= 10^18 => p <= 10^9.
    # But we don't need all primes. 
    # The product of the first k primes squared grows very fast.
    # 2^2 * 3^2 * 5^2 * 7^2 * 11^2 * 13^2 * 17^2 * 19^2 * 23^2 * 29^2 * 31^2 * 37^2
    # (2*3*5*7*11*13*17*19*23*29*31*37)^2 = (7420738134810)^2 ~ 5.5e25 >> 10^18.
    # So we only need primes up to ~37 or 41 to cover the "multi-prime" part.
    # For single prime powers p^e, p can be up to 10^9.
    
    # Strategy:
    # A. Generate all powerful numbers composed of primes <= 40 (multi-prime powerful numbers).
    # B. Generate all powerful numbers of the form p^e where p is prime and p > 40.
    #    These are "prime powerful numbers".
    
    def get_primes(limit):
        if limit < 2: return []
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(limit) + 1):
            if sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        return [i for i in range(2, limit+1) if sieve[i]]
    
    small_primes = get_primes(50)
    powerful = set()
    
    # A. Multi-prime powerful numbers using small primes
    # DFS: state is (index_in_primes, current_value)
    # We must ensure we don't exceed LIMIT.
    stack = [(0, 1)]
    while stack:
        idx, val = stack.pop()
        if val > LIMIT:
            continue
        if val > 1:
            powerful.add(val)
        
        # If we have used all small primes, we stop branching on new primes
        # but we can still increase the power of the current prime? 
        # No, the standard DFS for powerful numbers usually fixes the set of primes.
        # Here, we allow adding new primes from small_primes[idx:]
        
        for i in range(idx, len(small_primes)):
            p = small_primes[i]
            # Try p^2, p^3, ...
            next_val = val * p * p
            if next_val > LIMIT:
                break
            # Add to stack with next index i+1
            stack.append((i + 1, next_val))
            # Increase power of p
            while next_val * p <= LIMIT:
                next_val *= p
                stack.append((i + 1, next_val))
                if time_left() < 5:
                    # If we are running out of time during generation, 
                    # we might have an incomplete set, but let's try to finish this part
                    pass
    
    # B. Prime powerful numbers p^e for p > 50
    # p^2 <= 10^18 => p <= 10^9.
    # We can't iterate all primes up to 10^9.
    # But we only need to add p^e to the set if it's not already there.
    # Note: If p > 50, p^2 is not in the "small prime" set (which only uses primes <= 50).
    # So we need to generate these separately.
    # However, generating primes up to 10^9 is too slow.
    # Observation: Most powerful numbers with many representations involve small components.
    # Large prime powerful numbers (like $10^9+7$ squared) are sparse.
    # For maximizing the count of pairs, we likely rely on the dense region of small powerful numbers.
    # So we will ignore prime powerful numbers with p > 50 for the purpose of finding high-count N,
    # unless we find that N requires them. Given the density, small N or N composed of small powerful numbers is the best bet.
    
    # Let's verify the size of 'powerful'
    pn_list = sorted(powerful)
    pn_set = set(pn_list)
    
    if time_left() < 10:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # 2. Find Best N
    # Candidates:
    # We want N such that many (a, N-a) are in pn_set.
    # Since pn_list is sorted, we can use two pointers or binary search.
    # But iterating through all pn_list for each candidate is O(M).
    # M is roughly 1.2 million.
    # If we test 100 candidates, 10^8 ops. Might be tight.
    # Let's test fewer, better candidates.
    
    # Heuristic: N should be around 2 * (median powerful number) or smaller.
    # The powerful numbers are dense at the bottom.
    # Let's pick candidates from the sums of the first K powerful numbers.
    
    # To avoid O(M) per candidate, we can limit the search for 'a' to a < N/2.
    # For small N, this is fast.
    # For large N, it's slow.
    # So we prioritize smaller N candidates.
    
    candidates = set()
    
    # 1. Small N candidates
    # Try N from 2 to 10000?
    # Check how many powerful numbers are <= 5000.
    # If there are ~100, then counting is fast.
    # Let's generate candidates by summing pairs of small powerful numbers.
    
    small_pn = [x for x in pn_list if x <= 10000]
    if len(small_pn) > 0:
        for i in range(len(small_pn)):
            for j in range(i, len(small_pn)):
                s = small_pn[i] + small_pn[j]
                if s <= 100000: # Keep N small to ensure fast counting
                    candidates.add(s)
    
    # 2. Larger N candidates based on density
    # Sample some larger N values
    step = max(1, len(pn_list) // 1000)
    for i in range(0, len(pn_list), step):
        if i + 1 < len(pn_list):
            s = pn_list[i] + pn_list[i+1]
            if s <= LIMIT:
                candidates.add(s)
        # Also 2 * pn_list[i]
        s2 = 2 * pn_list[i]
        if s2 <= LIMIT:
            candidates.add(s2)
    
    candidate_list = list(candidates)
    # Sort by N, so we process small N first (faster counting)
    candidate_list.sort()
    
    best_N = 2
    best_count = 1
    best_pairs = [[1, 1]]
    
    # Count pairs for a given N
    # Optimization: Only check 'a' up to N//2.
    # Use bisect to find the index of the largest powerful number <= N//2.
    import bisect
    
    def count_pairs(N):
        # Find index of last element <= N//2
        half = N // 2
        # pn_list is sorted
        idx = bisect.bisect_right(pn_list, half)
        count = 0
        pairs = []
        for i in range(idx):
            a = pn_list[i]
            b = N - a
            if b in pn_set:
                # Ensure a <= b (guaranteed by a <= N//2)
                # Ensure distinct? Task says "distinct pairs a <= b". 
                # It doesn't say a != b. (a,a) is valid if 2a=N.
                count += 1
                pairs.append([a, b])
        return count, pairs

    # Evaluate candidates
    # We can't evaluate all if there are too many.
    # Limit the number of candidates to evaluate based on time.
    # Estimate time per candidate: ~ len(pn_list <= N/2) * cost_of_set_lookup.
    # For small N, this is very fast.
    
    max_candidates_to_try = 5000
    tried = 0
    
    for N in candidate_list:
        if time_left() < 10:
            break
        if tried >= max_candidates_to_try:
            break
            
        # Quick heuristic: If N is very large, counting is slow.
        # Skip N > 10^12 if we have time pressure, or just rely on the deadline.
        
        cnt, pairs = count_pairs(N)
        if cnt > best_count:
            best_count = cnt
            best_N = N
            best_pairs = pairs
            
        tried += 1
        
        # If we found a very high count, we might stop early?
        # No, we want to maximize.
        
    # If we haven't found a good count, try a few more specific large N?
    # The small N search should have covered the dense area.
    
    if not best_pairs:
        best_N = 2
        best_pairs = [[1, 1]]
        
    # Final Validation
    for a, b in best_pairs:
        assert a <= b
        assert a + b == best_N
        assert a in pn_set
        assert b in pn_set
        
    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
