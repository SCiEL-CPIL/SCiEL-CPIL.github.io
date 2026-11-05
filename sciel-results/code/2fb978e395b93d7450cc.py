"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import isqrt
    
    # We need to find N <= 10^18 with many representations as sum of two powerful numbers.
    # Powerful numbers: n = a^2 * b^3 where b is square-free.
    # 1 is powerful (1 = 1^2 * 1^3).
    
    # Strategy:
    # 1. Generate a list of powerful numbers up to a certain limit.
    #    However, N can be up to 10^18. We can't generate all powerful numbers up to 10^18 efficiently if we just iterate.
    #    But the density of powerful numbers is low. The count of powerful numbers up to X is approx X^(2/3).
    #    For X = 10^18, count is ~ (10^18)^(2/3) = 10^12. This is too many to store in a list or iterate over fully in 120s.
    #    
    #    Alternative Strategy:
    #    The seed program found a small N (limit 20000) with some pairs.
    #    The score is the *number* of pairs.
    #    To maximize the number of pairs, we want an N that is a sum of many pairs (a, b) of powerful numbers.
    #    
    #    Let's analyze the structure.
    #    If we pick N to be a sum of two large powerful numbers, the number of representations might be small.
    #    However, if we pick N such that it can be written as a+b in many ways, we need a "popular" sum.
    #    
    #    Consider N = 2 * p^2 where p is a large prime? No, that's just one pair (p^2, p^2) if p^2 is powerful (it is).
    #    
    #    Let's look at the distribution.
    #    Powerful numbers are of the form a^2 b^3.
    #    Most powerful numbers are squares or cubes.
    #    Actually, the set of powerful numbers is quite sparse.
    #    
    #    Let's try to construct N using a specific structure that guarantees many representations.
    #    Consider N = A + B where A and B are powerful.
    #    If we fix N, the number of pairs is the number of powerful a <= N/2 such that N-a is powerful.
    #    
    #    Is there a known N with many representations?
    #    The problem states "Integers with many certified representations are finite evidence on the conjectured subpolynomial bound."
    #    This implies we should search for such an N.
    #    
    #    Since we have 120 seconds, we can do a smarter search than the seed.
    #    The seed used limit=20000. We can go much higher.
    #    However, generating powerful numbers up to 10^12 is feasible?
    #    Count of powerful numbers up to X ~ X^(2/3).
    #    If X = 10^12, count ~ 10^8. This is too many to store in a Python list and iterate over for pair counting (O(K^2) is impossible).
    #    
    #    We need a different approach.
    #    Instead of generating all powerful numbers and counting sums, we can:
    #    1. Generate powerful numbers up to a limit L.
    #    2. Use a hash map to count sums? No, that requires iterating all pairs, which is O(K^2).
    #    
    #    Alternative:
    #    Pick a target N? No, we need to find the best N.
    #    
    #    Let's look at the seed's result.
    #    Limit 20000. Powerful numbers up to 20000:
    #    1, 4, 8, 9, 16, 25, 27, 32, 36, 49, 64, 81, 100, 108, 121, 125, 128, 144, 169, 196, 216, 225, 243, 256, 289, 324, 343, 361, 384, 400, ...
    #    The seed found the best N in this range.
    #    
    #    Can we go to N = 10^6?
    #    Count of powerful numbers up to 10^6 ~ (10^6)^(2/3) = 10^4.
    #    O(K^2) = 10^8 operations. This might be too slow in Python (120s limit).
    #    10^8 simple operations in Python can take ~10-50 seconds. It's risky but possible if optimized.
    #    
    #    Can we go to N = 10^8?
    #    Count ~ 10^(16/3) = 10^5.3 ~ 2 * 10^5.
    #    O(K^2) = 4 * 10^10. Too slow.
    #    
    #    So we are limited to N ~ 10^6 or 10^7 for O(K^2) approach.
    #    
    #    Let's optimize the O(K^2) approach for N up to 10^6 or 10^7.
    #    We generate powerful numbers up to N_max.
    #    We use a dictionary `counts` where key is sum `a+b`, value is count.
    #    We iterate a from powerful list, b from powerful list (b >= a to avoid duplicates? No, we just count all, then divide by 2 or handle carefully).
    #    Actually, we just need the count of distinct pairs (a,b) with a<=b.
    #    If we iterate all a, b in powerful list:
    #    For each pair (a,b) with a+b = S, we increment count[S].
    #    We must ensure a <= b.
    #    
    #    Let's try N_max = 10^6.
    #    Generate powerful numbers up to 10^6.
    #    K ~ 10^4.
    #    Pairs ~ 10^8 / 2 = 5 * 10^7.
    #    In Python, a loop of 50 million iterations might take ~5-10 seconds.
    #    This is feasible.
    #    
    #    Let's try N_max = 5 * 10^6.
    #    K ~ (5*10^6)^(2/3) ~ 10^4.2 ~ 16000.
    #    Pairs ~ 16000^2 / 2 = 1.28 * 10^8.
    #    This might take ~10-20 seconds.
    #    
    #    Let's try to push it further.
    #    N_max = 10^7.
    #    K ~ 10^(14/3) = 10^4.66 ~ 45000.
    #    Pairs ~ 45000^2 / 2 = 10^9.
    #    This is likely too slow (>120s).
    #    
    #    So let's aim for N_max = 2 * 10^6 or 5 * 10^6.
    #    
    #    Wait, is there a smarter way?
    #    The task asks to maximize the number of pairs.
    #    The seed found a score of 1.0 (which is the baseline).
    #    We need to beat it.
    #    
    #    Let's implement the O(K^2) counting for N_max = 10^6 first and see.
    #    Actually, we can just hardcode a known good N if we find one offline, but we must compute it.
    #    
    #    Let's write the code to:
    #    1. Generate powerful numbers up to N_MAX.
    #    2. Count sums.
    #    3. Find the N with the max count.
    #    4. Return the pairs for that N.
    #    
    #    To generate powerful numbers up to N_MAX:
    #    n = a^2 * b^3.
    #    b must be square-free.
    #    Iterate b from 1 to N_MAX^(1/3).
    #    For each b, a can go up to sqrt(N_MAX / b^3).
    #    
    #    Let's refine N_MAX.
    #    If N_MAX = 10^6, K is small.
    #    If N_MAX = 10^7, K is larger.
    #    
    #    Let's try N_MAX = 2_000_000.
    #    
    #    One optimization:
    #    Instead of nested loops over the list, we can use a set for membership and iterate a, then check if N-a is in the set?
    #    No, we don't know N in advance. We want to find the best N.
    #    So we must count all sums.
    #    
    #    Let's use `collections.Counter`.
    #    
    #    Code structure:
    #    1. Generate powerful numbers list `pw`.
    #    2. Sort `pw`.
    #    3. `counts = Counter()`.
    #    4. For i, a in enumerate(pw):
    #         For b in pw[i:]:  # b >= a
    #             s = a + b
    #             if s > N_MAX: break
    #             counts[s] += 1
    #    5. Find max count.
    #    6. Retrieve pairs for that N.
    #    
    #    This is O(K^2).
    #    
    #    Let's set N_MAX = 1_000_000 to be safe on time.
    #    If time permits, we can increase it.
    #    Actually, let's try 2_000_000.
    #    
    #    Let's double check the powerful number generation.
    #    b^3 <= N_MAX => b <= N_MAX^(1/3).
    #    For N_MAX = 2e6, b <= 125.
    #    For each b, a^2 <= N_MAX / b^3 => a <= sqrt(2e6 / b^3).
    #    Sum of a's is roughly sum(sqrt(2e6)/b^1.5) ~ sqrt(2e6) * sum(1/b^1.5) ~ 1414 * 2 ~ 2800?
    #    Wait, the count of powerful numbers is approx N^(2/3).
    #    (2e6)^(2/3) = (2000000)^(0.666) ~ 16000.
    #    So K ~ 16000.
    #    Pairs ~ 16000^2 / 2 = 1.28e8.
    #    Python loop 1.28e8 iterations.
    #    Each iteration: addition, dictionary update.
    #    This might take 10-20 seconds.
    #    
    #    Let's try to optimize the inner loop.
    #    Using a local variable for the counter update might help.
    #    
    #    Let's write the code.

    N_MAX = 2_000_000
    
    # Generate powerful numbers
    pw_set = set()
    max_b = isqrt(isqrt(N_MAX))  # b^3 <= N_MAX => b <= N_MAX^(1/3). isqrt(isqrt(N)) is approx N^(1/4), which is smaller than N^(1/3) for large N?
    # No. N^(1/3) is larger than N^(1/4).
    # We need b^3 <= N_MAX.
    # b <= int(N_MAX ** (1/3))
    # Let's just use integer cube root approximation or loop while b^3 <= N_MAX.
    
    b = 1
    while b**3 <= N_MAX:
        # b is square-free?
        # We need b to be square-free for the form a^2 * b^3 to be the unique representation?
        # Actually, the definition of powerful number is just that every prime factor appears squared.
        # The form a^2 * b^3 with b square-free generates all powerful numbers uniquely.
        # So we should only iterate square-free b.
        
        # Check if b is square-free
        # A number is square-free if it is not divisible by any square > 1.
        # Simple check: for p in primes up to sqrt(b), if p*p divides b, not square-free.
        # Since b is small (< 126), we can just check.
        
        is_square_free = True
        d = 2
        temp_b = b
        while d*d <= temp_b:
            if temp_b % (d*d) == 0:
                is_square_free = False
                break
            d += 1
        if not is_square_free:
            b += 1
            continue
            
        max_a = isqrt(N_MAX // (b**3))
        for a in range(1, max_a + 1):
            n = a*a * b**3
            if n <= N_MAX:
                pw_set.add(n)
        b += 1
        
    pw = sorted(pw_set)
    
    # Count sums
    from collections import Counter
    counts = Counter()
    
    # Optimization: use local vars
    pw_len = len(pw)
    for i in range(pw_len):
        a = pw[i]
        # b must be >= a
        for j in range(i, pw_len):
            b_val = pw[j]
            s = a + b_val
            if s > N_MAX:
                break
            counts[s] += 1
            
    # Find best N
    if not counts:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
        
    best_N, best_count = max(counts.items(), key=lambda x: x[1])
    
    # Retrieve pairs
    pairs = []
    pw_set_local = pw_set
    for a in pw:
        if 2*a > best_N:
            break
        b_val = best_N - a
        if b_val in pw_set_local:
            pairs.append([a, b_val])
            
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
