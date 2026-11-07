"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _is_valid_set(S_list):
    """Check if sorted list S_list satisfies: for all distinct a,b,c, a does not divide b*c."""
    n = len(S_list)
    # Optimization: If n < 3, condition is vacuously true
    if n < 3:
        return True
    
    # Precompute for speed? The triple loop is O(N^3). N ~ 50. 50^3 = 125k.
    # Python can do ~10M ops/sec, so 125k is ~10ms.
    # We might call this many times.
    
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            for k in range(j + 1, n):
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def construct():
    deadline = time.time() + 50  # Safety margin for 60s limit
    n_max = 256
    
    # 1. Start with all primes
    primes = [x for x in range(2, n_max + 1) if _is_prime(x)]
    S = primes[:]
    S.sort()
    best = S[:]
    best_size = len(S)
    
    # 2. Try to add composites greedily
    # We iterate through all numbers not in S
    # To allow more additions, we might need to re-check after each addition
    # But adding one number might invalidate previously added ones? 
    # No, the condition is global. If S is valid, and we add x, we only need to check 
    # if x conflicts with pairs in S, and if pairs in S conflict with x?
    # Wait. The condition is: for EVERY triple (a,b,c) in the NEW set, a !| bc.
    # If S is valid, all triples within S are fine.
    # When adding x, we must check triples involving x.
    # Triples are (x, b, c), (b, x, c), (b, c, x).
    # 1. x !| b*c for all b,c in S, b!=c
    # 2. b !| x*c for all b,c in S, b!=c
    # 3. c !| b*x for all b,c in S, b!=c
    # Note: 2 and 3 are symmetric.
    
    def _can_add(S_set, x):
        L = list(S_set)
        n = len(L)
        # Check x !| b*c
        for i in range(n):
            b = L[i]
            for j in range(i+1, n):
                c = L[j]
                if (b * c) % x == 0:
                    return False
        
        # Check b !| x*c and c !| b*x
        # This is equivalent to: for all b,c in S, b !| x*c AND c !| b*x
        for i in range(n):
            b = L[i]
            for j in range(i+1, n):
                c = L[j]
                if (x * c) % b == 0:
                    return False
                if (b * x) % c == 0:
                    return False
        return True

    # Greedy addition
    # Try adding numbers in ascending order? Or descending?
    # Smaller numbers are more likely to divide products, so they are "more restrictive".
    # Larger numbers are less likely to be divisors of small products.
    # However, if we add a large number, it might divide the product of two medium numbers.
    # Let's try adding in descending order, as larger numbers are "safer" to add?
    # Or ascending?
    # Let's try multiple orders.
    
    candidates = [x for x in range(2, n_max + 1) if not _is_prime(x)]
    
    for order in [sorted(candidates, reverse=True), sorted(candidates)]:
        if time.time() > deadline:
            break
            
        current_S = best[:]
        current_S_set = set(current_S)
        
        # We need to re-evaluate validity if we add multiple?
        # If we add x1, then x2, we must ensure {S, x1, x2} is valid.
        # _can_add checks x against S. It does NOT check x1 against x2.
        # So we must maintain the full set and check validity of the whole set,
        # or update the check to include previously added numbers.
        
        # Simpler: Just try to add one by one, checking against the current full set.
        # If _can_add(current_S_set, x) is True, add it.
        # But _can_add as written only checks x against S.
        # If we add x1, S becomes S U {x1}.
        # When checking x2, we must check x2 against S U {x1}.
        # My _can_add does exactly that if I pass the updated set.
        
        for x in order:
            if time.time() > deadline:
                break
            if x in current_S_set:
                continue
            if _can_add(current_S_set, x):
                current_S.append(x)
                current_S_set.add(x)
        
        if len(current_S) > best_size:
            best = current_S[:]
            best_size = len(current_S)
            best.sort()

    # 3. Local Search: Try swapping one element out, another in
    # This is unlikely to increase size, but might find a different valid set of same size.
    # The goal is to MAXIMIZE cardinality. If we can't add, can we swap to enable an addition later?
    # This is a 2-step process. Given time constraints, we skip complex multi-step swaps.
    # However, we can try: Remove a prime p, try to add a composite c. If valid, we have same size.
    # Then try to add another composite c2. If valid, we have size + 1.
    
    # Let's do a simple hill climb on the best set found so far.
    # Try to improve best.
    
    all_nums = list(range(2, n_max + 1))
    
    # We'll try a few random restarts or swaps if we have time.
    # But the greedy addition is the primary driver.
    
    # Final verification
    if not _is_valid_set(best):
        # Fallback to primes if something went wrong
        return [int(x) for x in primes]
        
    return [int(x) for x in sorted(best)]

# EVOLVE-BLOCK-END
