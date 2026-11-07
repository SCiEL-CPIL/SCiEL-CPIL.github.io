import math
import time

def construct():
    start = time.time()
    deadline = start + 50  # safety margin before 60s kill

    MAXN = 10000000
    # We'll search for a,b with sigma(a)=sigma(b) and a/b close to 3/2.
    # Strategy: build a and b from prime powers with matching sigma products.
    # For each "sigma factor" S, we have a set of prime powers p^k with sigma(p^k)=S.
    # We assign some factors to a and some to b such that product of S's is equal.
    # Actually, simpler: enumerate pairs (a,b) where a and b are built from
    # the same multiset of sigma values, but different primes.

    # Precompute: for small primes, sigma(p) = p+1.
    # We'll consider composite sigma values too, but start with primes.

    # Generate primes up to MAXN (sieve)
    limit = int(math.isqrt(MAXN)) + 1
    # We need primes up to MAXN for single-prime a or b, but for products we need smaller.
    # Actually a and b can be up to 10^7, so primes up to 10^7.
    # Sieve up to 10^7
    sieve = bytearray(b'\x01') * (MAXN + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(MAXN)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i in range(2, MAXN + 1) if sieve[i]]

    # For each prime p, sigma(p) = p+1.
    # Group primes by sigma value: sigma_val -> list of primes
    from collections import defaultdict
    sigma_to_primes = defaultdict(list)
    for p in primes:
        s = p + 1
        sigma_to_primes[s].append(p)

    # Also consider prime powers p^k for k>=2, sigma(p^k) = (p^{k+1}-1)/(p-1)
    # But these are less common; include small ones.
    # For p^2: sigma = p^2+p+1. For p^3: p^3+p^2+p+1, etc.
    # We'll add these to sigma_to_primes as well, but track the actual n contribution.
    # Let's build a list of (sigma_val, n_value) pairs.
    # n_value is the actual prime power p^k.

    sigma_to_npows = defaultdict(list)
    # Single primes
    for p in primes:
        sigma_to_npows[p + 1].append(p)

    # Prime powers p^k, k>=2, p^k <= MAXN
    for p in primes:
        if p > int(math.isqrt(MAXN)):
            break
        pk = p * p
        while pk <= MAXN:
            # sigma(p^k) = (p^{k+1}-1)/(p-1)
            # Compute iteratively
            # For k=2: p^2+p+1
            # We'll just compute directly
            k = 2
            pk = p * p
            while pk <= MAXN:
                # sigma(p^k)
                sig = 1
                term = 1
                for _ in range(k + 1):
                    sig += term
                    term *= p
                sigma_to_npows[sig].append(pk)
                pk *= p
                k += 1

    # Now, for each sigma value S, we have a list of n-values (prime powers) that have sigma S.
    # We want to find two numbers a and b such that:
    #   a = product of some n-values (from distinct sigma factors, or same sigma factor used once per side)
    #   b = product of some n-values
    #   The multiset of sigma factors for a equals that for b.
    #   a/b is close to 3/2.

    # This is a complex combinatorial problem. Let's do a bounded search.
    # We'll consider using 1, 2, or 3 sigma factors.

    # For 1 factor: a = p, b = q, sigma(p)=sigma(q)=p+1=q+1 => p=q. Not useful.
    # So we need at least 2 factors where we can swap primes.

    # For 2 factors: a = p1*p2, b = q1*q2, where sigma(p1)=sigma(q1)=S1, sigma(p2)=sigma(q2)=S2.
    # We need p1 != q1 or p2 != q2 to have a != b.
    # a/b = (p1*p2)/(q1*q2). We want this close to 3/2.

    # Let's search over pairs of sigma values (S1, S2) and assignments.
    # To keep it tractable, limit to sigma values with at least 2 primes.

    # Build list of (S, [n1, n2, ...]) where len >= 2
    multi_sigma = []
    for S, npows in sigma_to_npows.items():
        if len(npows) >= 2:
            multi_sigma.append((S, npows))

    # Sort by number of primes descending for more options
    multi_sigma.sort(key=lambda x: -len(x[1]))

    best_ratio = None
    best_pair = None
    best_err = float('inf')

    # Search over pairs of sigma factors
    # Limit to first N sigma factors for speed
    max_sigma_factors = min(len(multi_sigma), 200)
    multi_sigma = multi_sigma[:max_sigma_factors]

    # For each pair (i, j), try all combinations of picking one prime from S_i for a,
    # one from S_i for b, one from S_j for a, one from S_j for b.
    # But that's 4 choices per factor = 16 combos per pair. With 200 sigma factors,
    # that's 200*200/2 * 16 = 320000 combos. Each combo is O(1). Should be fast.

    for i in range(len(multi_sigma)):
        if time.time() > deadline:
            break
        S1, npows1 = multi_sigma[i]
        for j in range(i + 1, len(multi_sigma)):
            if time.time() > deadline:
                break
            S2, npows2 = multi_sigma[j]

            # For each choice of a-prime from S1, b-prime from S1, a-prime from S2, b-prime from S2
            # Limit list sizes to avoid explosion
            npows1_l = npows1[:50]
            npows2_l = npows2[:50]

            for p1 in npows1_l:
                for q1 in npows1_l:
                    if p1 == q1:
                        continue  # need different to make a != b potentially
                    for p2 in npows2_l:
                        for q2 in npows2_l:
                            a = p1 * p2
                            b = q1 * q2
                            if a > MAXN or b > MAXN:
                                continue
                            if a == b:
                                continue
                            # Verify sigma(a) == sigma(b)
                            # sigma(a) = S1 * S2, sigma(b) = S1 * S2. OK by construction.
                            ratio = a / b
                            err = abs(ratio - 1.5)
                            if err < best_err:
                                best_err = err
                                best_ratio = ratio
                                best_pair = (a, b)

    # Also try 3 sigma factors
    # a = p1*p2*p3, b = q1*q2*q3
    # This is 8 combos per triple of sigma factors.
    # Limit to first 50 sigma factors for triples.
    max_sigma3 = min(len(multi_sigma), 50)
    multi_sigma3 = multi_sigma[:max_sigma3]

    for i in range(len(multi_sigma3)):
        if time.time() > deadline:
            break
        S1, npows1 = multi_sigma3[i]
        for j in range(i + 1, len(multi_sigma3)):
            if time.time() > deadline:
                break
            S2, npows2 = multi_sigma3[j]
            for k in range(j + 1, len(multi_sigma3)):
                if time.time() > deadline:
                    break
                S3, npows3 = multi_sigma3[k]

                npows1_l = npows1[:20]
                npows2_l = npows2[:20]
                npows3_l = npows3[:20]

                for p1 in npows1_l:
                    for q1 in npows1_l:
                        for p2 in npows2_l:
                            for q2 in npows2_l:
                                for p3 in npows3_l:
                                    for q3 in npows3_l:
                                        a = p1 * p2 * p3
                                        b = q1 * q2 * q3
                                        if a > MAXN or b > MAXN:
                                            continue
                                        if a == b:
                                            continue
                                        ratio = a / b
                                        err = abs(ratio - 1.5)
                                        if err < best_err:
                                            best_err = err
                                            best_ratio = ratio
                                            best_pair = (a, b)

    if best_pair is not None:
        a, b = best_pair
        return [int(a), int(b)]
    else:
        # Fallback: return a valid pair
        return [11, 6]

