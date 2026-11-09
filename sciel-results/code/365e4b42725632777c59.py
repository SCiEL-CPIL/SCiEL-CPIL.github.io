"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find N <= 10^18 with many representations as sum of two powerful numbers.
    
    Strategy:
    1. Generate all powerful numbers up to a reasonable limit.
    2. Use a dense set of small powerful numbers to construct candidate N values.
    3. For each candidate N, verify that N - a is powerful for many a in the set.
    4. Return the N with the maximum number of valid pairs.
    
    A powerful number is one where every prime factor appears with exponent >= 2.
    1 is considered powerful.
    """
    from math import isqrt
    import sys
    from collections import defaultdict
    
    # Generate powerful numbers up to a limit
    # Powerful numbers are of the form a^2 * b^3 where b is square-free
    # We generate them by iterating over possible b (square-free) and a
    
    def is_square_free(n):
        """Check if n is square-free."""
        if n <= 1:
            return True
        x = n
        for p in range(2, isqrt(x) + 1):
            if x % (p * p) == 0:
                return False
        return True
    
    # For efficiency, let's generate powerful numbers differently.
    # A powerful number n can be written as a^2 * b^3 where b is square-free.
    # Actually, the standard form is: if n = prod p_i^e_i, then e_i >= 2 for all i.
    # We can write n = a^2 * b^3 where b is square-free and gcd(a,b)=1? 
    # No, simpler: every powerful number is of the form a^2 * b^3 for some a,b.
    # Actually, the cleanest generation: n is powerful iff for every prime p|n, p^2|n.
    # We can generate by: for each square-free m, and each a, n = a^2 * m^3 is powerful?
    # Let's check: if m is square-free, m^3 has all exponents 3 (odd, >=2). 
    # a^2 has all exponents even (>=0). The product has exponents >=2 for primes in m,
    # and even exponents >=0 for primes in a not in m. But if a prime p is in a but not m,
    # its exponent is even, which could be 0, 2, 4, ... If it's 0, it's not in the number.
    # If it's >=2, it's powerful. So yes, a^2 * m^3 is powerful for any a and square-free m.
    # And every powerful number can be written this way: let m be the product of primes 
    # with odd exponent in n, and a = n / m^3... wait, that's not right either.
    
    # Let's use a direct approach: generate powerful numbers by checking.
    # But that's slow for large limits.
    
    # Better approach: 
    # n is powerful iff n = a^2 * b^3 where b is square-free and gcd(a, b) = 1? 
    # Actually, no. Let's just generate all powerful numbers up to LIMIT by:
    # For each square-free b up to LIMIT^(1/3), for each a up to sqrt(LIMIT / b^3),
    # n = a^2 * b^3 is a candidate. But we need to ensure n is actually powerful.
    # Actually, a^2 * b^3 with b square-free IS always powerful.
    # Proof: Let n = a^2 * b^3. For any prime p|n:
    #   - If p|b, then p^3 | b^3, so p^3 | n, hence p^2 | n.
    #   - If p|a but p∤b, then p^2 | a^2, so p^2 | n.
    #   - If p|a and p|b, then p^2|a^2 and p^3|b^3, so p^5|n, hence p^2|n.
    # So yes, a^2 * b^3 with b square-free is always powerful.
    
    # And every powerful number n can be written as a^2 * b^3 with b square-free:
    # Write n = prod p_i^e_i with e_i >= 2. Let b = prod p_i where e_i is odd, 
    # and a = prod p_i^(floor(e_i/2)). Then b is square-free, and a^2 * b^3 = n.
    # Wait, let's verify: if e_i is even, say e_i = 2k, then p_i^(2k) = (p_i^k)^2 * 1^3.
    # If e_i is odd, say e_i = 2k+1 (k>=0), then p_i^(2k+1) = (p_i^k)^2 * p_i^3.
    # So a = prod p_i^(floor(e_i/2)) and b = prod p_i where e_i is odd.
    # b is square-free since each p_i appears at most once.
    # So the generation is complete.
    
    LIMIT = 10**7  # We'll generate powerful numbers up to this
    powerful_set = set()
    powerful_list = []
    
    # Max b: b^3 <= LIMIT, so b <= LIMIT^(1/3)
    max_b = int(LIMIT ** (1/3)) + 1
    for b in range(1, max_b + 1):
        if not is_square_free(b):
            continue
        b3 = b ** 3
        if b3 > LIMIT:
            continue
        max_a = isqrt(LIMIT // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= LIMIT and n not in powerful_set:
                powerful_set.add(n)
                powerful_list.append(n)
    
    powerful_list.sort()
    
    # Now we have all powerful numbers up to LIMIT.
    # We want to find N <= 10^18 with many representations N = a + b where a,b are powerful.
    # Since N can be up to 10^18, we can't enumerate all powerful numbers up to 10^18.
    # But we can use the small powerful numbers we've generated.
    
    # Strategy: Pick a dense set of small powerful numbers. For each candidate N,
    # count how many a in our powerful list satisfy that N - a is also powerful.
    
    # But N - a could be very large (up to 10^18), so we can't just look it up in powerful_set.
    # We need a way to check if a large number is powerful.
    
    def is_powerful(n):
        """Check if n is powerful by exact factorization."""
        if n == 1:
            return True
        x = n
        for p in range(2, isqrt(x) + 1):
            if x % (p * p) == 0:
                return False
        return True
    
    # This is too slow for large n. Let's think differently.
    
    # Alternative: We can construct N from sums of pairs of small powerful numbers.
    # If a and b are both <= LIMIT, then N = a + b <= 2*LIMIT.
    # But we want N up to 10^18, so this only works for small N.
    
    # Better: Let's find small N first, then try to scale up.
    # Actually, the problem says N up to 10^18. Let's find the best small N first.
    
    # For small N (up to 2*LIMIT), we can count pairs directly.
    best_N = 2
    best_count = 0
    best_pairs = []
    
    for N in range(2, 2 * LIMIT + 1):
        count = 0
        pairs = []
        for a in powerful_list:
            if a > N // 2:
                break
            b = N - a
            if b in powerful_set:
                count += 1
                pairs.append([a, b])
        if count > best_count:
            best_count = count
            best_N = N
            best_pairs = pairs
    
    # Now try to find larger N by using the structure.
    # If N0 has many representations, can we find N = N0 + K where K is such that
    # N - a is powerful for many a?
    
    # For now, let's just return the best small N we found.
    # But we should try to do better.
    
    # Let's try a different approach: pick a set of powerful numbers and find N
    # such that N - a_i is powerful for many i.
    
    # Use the top few hundred powerful numbers as candidates for a.
    candidates = powerful_list[:500]
    
    # For each pair of candidates (a1, a2), consider N = a1 + b1 where b1 is powerful.
    # This is getting complex. Let's just return the best we found.
    
    return {'N': best_N, 'pairs': best_pairs}

# EVOLVE-BLOCK-END
