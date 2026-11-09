"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    # Precompute powers of each small prime up to 10^6
    prime_powers = {}
    for p in small_primes:
        powers = []
        v = p
        while v <= 1000000:
            powers.append(v)
            v *= p
        prime_powers[p] = powers
    
    # Precompute for each small prime power, the set of residues mod p^k that are 0 mod p
    # Actually, we need to count distinct small primes dividing n+i.
    # A small prime p divides n+i iff n+i ≡ 0 (mod p), i.e., n ≡ -i (mod p).
    # For a given n, the set of small primes dividing n+i is {p : n ≡ -i (mod p)}.
    # We want to minimize the number of DISTINCT small primes appearing in the union over i=1..24.
    # Let S(n) = union_{i=1}^{24} {p in small_primes : n ≡ -i (mod p)}.
    # We want to minimize |S(n)|.
    
    # The number of distinct small primes is at most 9.
    # Can we achieve fewer? Let's think.
    # For each prime p, the residues -1, -2, ..., -24 mod p are distinct if p > 24.
    # But all our primes are <= 23.
    # For p=2: residues -1..-24 mod 2 are just 1,0,1,0,... so {0,1} = {1,0}. Actually -i mod 2:
    #   i=1: 1, i=2: 0, i=3: 1, i=4: 0, ...
    #   So the set of residues is {0, 1}. For any n, n mod 2 is either 0 or 1.
    #   If n is even (n mod 2 = 0), then for i even, n+i is even, so 2 divides n+i.
    #   If n is odd (n mod 2 = 1), then for i odd, n+i is even, so 2 divides n+i.
    #   So 2 ALWAYS divides at least one of n+1..n+24. Thus 2 is always in S(n).
    
    # For p=3: residues -1..-24 mod 3. -1=2, -2=1, -3=0, -4=2, ...
    #   The set of residues is {0, 1, 2}. For any n, n mod 3 is 0, 1, or 2.
    #   If n mod 3 = r, then for i ≡ -r (mod 3), n+i ≡ 0 (mod 3).
    #   Since i ranges 1..24, and 24 is a multiple of 3, every residue class mod 3 is hit.
    #   So 3 ALWAYS divides at least one of n+1..n+24. Thus 3 is always in S(n).
    
    # Similarly for any prime p <= 24, since 24 >= p, every residue class mod p is hit by some i in 1..24.
    # Wait, is that true? For p=23, i ranges 1..24. -i mod 23 for i=1..24 gives:
    #   -1=22, -2=21, ..., -22=1, -23=0, -24=22.
    #   So residues 0..22 are all present. Yes.
    # For p=19: i=1..24 covers all residues mod 19 since 24 >= 19.
    # In fact, for ANY prime p <= 24, since 24 >= p, the set {-i mod p : i=1..24} = all residues mod p.
    # Therefore, for ANY n, and ANY small prime p, there exists i in 1..24 such that p | (n+i).
    # This means S(n) = small_primes for ALL n!
    # So |S(n)| = 9 for all n in [1000, 10^6].
    
    # Wait, let me re-read the task.
    # "For each n+i, 1<=i<=24, extract its complete factor composed of primes STRICTLY BELOW 24."
    # "MINIMIZE the number of distinct extracted factors."
    # The "extracted factor" for n+i is the product of all small prime powers dividing n+i.
    # But the task says "distinct extracted factors", not "distinct primes".
    # Hmm, "complete factor composed of primes STRICTLY BELOW 24" — this is the small-prime part of n+i.
    # Let f(n+i) = largest divisor of n+i that is composed only of primes < 24.
    # Then we consider the set {f(n+1), f(n+2), ..., f(n+24)} and want to minimize its size.
    
    # So it's NOT about distinct primes, but distinct VALUES of the small-prime part.
    # This changes everything!
    
    # Let's recompute. For each n, compute g(n) = |{f(n+i) : i=1..24}| where f(m) is the largest
    # divisor of m with all prime factors < 24. We want to minimize g(n).
    
    # Precompute f(m) for all m up to 10^6 + 24.
    # f(m) = m / (m / f(m))... let's just compute it by dividing out all primes >= 24.
    
    max_val = 1000000 + 24
    f = list(range(max_val + 1))
    for i in range(2, max_val + 1):
        f[i] = f[i - 1]  # placeholder, we'll compute properly
    
    # Better: for each m, f(m) = m with all prime factors >= 24 removed.
    # Equivalently, f(m) = product of p^a for p < 24 where p^a || m.
    
    # Let's compute f(m) for all m.
    f_arr = [0] * (max_val + 1)
    f_arr[0] = 0
    f_arr[1] = 1
    for m in range(2, max_val + 1):
        x = m
        val = 1
        for p in small_primes:
            while x % p == 0:
                val *= p
                x //= p
        # x now has no small prime factors. f(m) = val.
        f_arr[m] = val
    
    # Now search for n in [1000, 10^6] minimizing the number of distinct f(n+i) for i=1..24.
    best_n = 1000
    best_count = 25  # max possible is 24
    
    # Try a random sample first to get a sense of the range.
    import random
    random.seed(42)
    
    # First, do a quick scan over a subset to find good candidates.
    # Then refine locally.
    
    # Quick scan: check every 1000th value.
    for n in range(1000, 1000001, 1000):
        if time.time() > deadline - 5:
            break
        factors = set()
        for i in range(1, 25):
            factors.add(f_arr[n + i])
        cnt = len(factors)
        if cnt < best_count:
            best_count = cnt
            best_n = n
    
    # Now do a local search around the best found.
    # Also try random samples.
    for _ in range(5000):
        if time.time() > deadline - 5:
            break
        n = random.randint(1000, 1000000)
        factors = set()
        for i in range(1, 25):
            factors.add(f_arr[n + i])
        cnt = len(factors)
        if cnt < best_count:
            best_count = cnt
            best_n = n
    
    # Local refinement around best_n.
    for delta in range(-100, 101):
        if time.time() > deadline - 5:
            break
        n = best_n + delta
        if n < 1000 or n > 1000000:
            continue
        factors = set()
        for i in range(1, 25):
            factors.add(f_arr[n + i])
        cnt = len(factors)
        if cnt < best_count:
            best_count = cnt
            best_n = n
    
    # Do a more thorough scan if time permits.
    if time.time() < deadline - 10:
        # Scan in chunks, checking all values.
        for start in range(1000, 1000001, 10000):
            if time.time() > deadline - 5:
                break
            for n in range(start, min(start + 10000, 1000001)):
                factors = set()
                for i in range(1, 25):
                    factors.add(f_arr[n + i])
                cnt = len(factors)
                if cnt < best_count:
                    best_count = cnt
                    best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
