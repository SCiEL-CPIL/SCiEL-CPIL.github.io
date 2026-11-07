def construct():
    # We want a set of distinct integers in 2..40 such that all 2^k subset products are distinct.
    # This is equivalent to requiring that the exponent vectors (in the prime factorization)
    # of the chosen numbers form a set with distinct subset sums.
    #
    # A powerful construction: use numbers that are pure powers of distinct primes.
    # If we pick numbers p1^a1, p2^a2, ..., pk^ak for distinct primes p1,...,pk,
    # then each number has a non-zero exponent in exactly one prime coordinate.
    # The subset products are distinct iff the subset sums of the exponents in each
    # coordinate are distinct.
    #
    # For a single prime p, we can pick powers p^e1, p^e2, ..., p^em (with ei >= 1, pi^ei <= 40)
    # such that all subset sums of {e1,...,em} are distinct.
    # The maximum number of exponents we can pick for prime p is limited by:
    #   - p^em <= 40
    #   - the subset sums of the chosen exponents must be distinct.
    #
    # For distinct subset sums, a sufficient condition is that the exponents are "superincreasing":
    # e1 < e2 < ... < em where ei > sum(e1,...,e_{i-1}).
    # But we can do better. The key constraint is that all 2^m subset sums are distinct,
    # which means the maximum subset sum (sum of all ei) must be at least 2^m - 1.
    # Also, we need p^(sum of all ei) <= 40 for the largest product? No, that's not right.
    # Actually, the subset sums just need to be distinct integers. The largest possible
    # subset sum is sum(ei). We need all 2^m subset sums to be distinct values in [0, sum(ei)].
    # This is possible iff sum(ei) >= 2^m - 1.
    #
    # So for each prime p, we want to maximize m such that there exist positive integers
    # e1,...,em with p^(ei) <= 40 for all i, and sum(ei) >= 2^m - 1.
    # To maximize m, we should minimize the exponents. The minimal sum for m exponents
    # with distinct subset sums is 2^m - 1 (achieved by exponents 1,1,2,4,8,... but wait,
    # we need distinct subset sums, not distinct elements).
    # Actually, the minimal maximum subset sum for m numbers with distinct subset sums is 2^m - 1,
    # achieved by {1, 2, 4, ..., 2^(m-1)}.
    # But we need p^(2^(m-1)) <= 40 for the largest exponent.
    #
    # Let's compute for each prime the maximum m:
    # p=2: 2^(2^(m-1)) <= 40. 2^1=2, 2^2=4, 2^4=16, 2^8=256>40. So m-1 <= 4, m <= 5? 
    #   Wait: exponents 1,2,4,8. Sum=15=2^4-1. 2^8=256>40. So we can't use exponent 8.
    #   Try exponents 1,2,4. Sum=7=2^3-1. 2^4=16<=40. So m=3 works: {2,4,16}.
    #   Can we do m=4? Need 4 exponents with distinct subset sums, sum >= 15.
    #   Minimal sum is 15 (exponents 1,2,4,8). But 2^8=256>40.
    #   Alternative: can we have 4 exponents with sum < 15 but still distinct subset sums?
    #   No, because 2^4=16 distinct subset sums need range [0,15], so sum >= 15.
    #   So m=3 for p=2. Exponents: 1,2,4. Numbers: 2,4,16.
    #
    # p=3: 3^(2^(m-1)) <= 40. 3^1=3, 3^2=9, 3^4=81>40. So m-1 <= 2, m <= 3?
    #   Try m=3: need sum >= 7. Exponents 1,2,4. 3^4=81>40. Can't.
    #   Try m=2: need sum >= 3. Exponents 1,2. 3^2=9<=40. Sum=3. Works. Numbers: 3,9.
    #   Can we do m=3 with different exponents? Need 3 exponents, sum >= 7, each 3^ei <= 40.
    #   Max ei: 3^3=27<=40, 3^4=81>40. So ei in {1,2,3}.
    #   Can we pick 3 exponents from {1,2,3} with distinct subset sums? 
    #   {1,2,3}: subset sums: 0,1,2,3,3,4,5,6. Duplicate 3. Not distinct.
    #   So m=2 for p=3. Numbers: 3,9.
    #
    # p=5: 5^1=5, 5^2=25, 5^3=125>40. Max ei=2.
    #   m=2: exponents 1,2. Sum=3. 5^2=25<=40. Works. Numbers: 5,25.
    #
    # p=7: 7^1=7, 7^2=49>40. Max ei=1.
    #   m=1: exponent 1. Number: 7.
    #
    # p=11,13,17,19,23,29,31,37: each gives m=1, one number each.
    #
    # Total: 3 + 2 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 16 numbers.
    #
    # Let's verify all 16 numbers are in 2..40 and distinct:
    # 2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37
    # All distinct, all in [2,40]. Count = 16.
    
    return [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
