"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def are_pairwise_coprime(nums):
    unique = list(set(nums))
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            if gcd(unique[i], unique[j]) != 1:
                return False
    return True

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_count = 5

    # Precompute primes and prime powers up to 256
    primes_set = set(x for x in range(2, 257) if is_prime(x))
    
    # Prime powers: p^k for primes p and k>=1, up to 256
    prime_powers = set(primes_set)
    for p in range(2, 257):
        if is_prime(p):
            pk = p
            while pk * p <= 256:
                pk *= p
                prime_powers.add(pk)
    
    # A number is "coprime-safe" if it's 1 or a prime power of a single prime
    # Any two distinct prime powers of different primes are coprime
    # But we need pairwise coprime among all 9 sums, so let's be more careful
    # Actually, for maximum cardinality 9, we want all 9 sums to be distinct
    # and pairwise coprime. The safest way is to have them all be distinct primes.
    # But prime powers of the same prime are NOT coprime with each other.
    # So we should restrict to primes only for the 9-sum target.

    # Phase 1: Targeted backtracking search for 9 distinct prime sums
    # Strategy: For each A = [a1, a2, a3], find B = [b1, b2, b3] such that
    # all 9 values a_i + b_j are distinct primes.
    
    # Precompute for each a in [1,128] and b in [1,128]: is a+b prime?
    # Then for a given A, the valid b's are those where a1+b, a2+b, a3+b are all prime.
    # Among those valid b's, we need to pick 3 such that all 9 sums are distinct.
    
    # Optimization: For a given A, if b1 and b2 are both valid, then the 6 sums
    # {a1+b1, a2+b1, a3+b1, a1+b2, a2+b2, a3+b2} must all be distinct primes.
    # Since they're all prime, they're automatically pairwise coprime.
    # We just need distinctness.
    
    # Let's precompute prime lookup as a set for O(1) checks
    prime_lookup = primes_set  # set of primes up to 256

    # Phase 1: Systematic search over A triples
    # For efficiency, iterate A and for each, collect valid b's, then find compatible triples
    
    # To make this faster, we can use the fact that if a1+b and a2+b are both prime,
    # then (a2-a1) must not share a factor with either prime. This is automatically
    # satisfied if both are prime and a2 != a1.
    
    # Let's do a more efficient search:
    # For each pair (a1, a2) with a1 < a2, compute the set of b where a1+b and a2+b are both prime.
    # Then for each a3, intersect with the set where a3+b is also prime.
    
    # Precompute: for each difference d = a2-a1, which b make both a1+b and a1+d+b prime?
    # This depends on a1, so let's just do it directly but with early termination.
    
    # Better approach: iterate over A, but use a smarter inner loop.
    # For a fixed A, the valid b's form a subset of [1,128]. We need to pick 3 from this
    # subset such that all 9 sums are distinct.
    
    # For distinctness: if b1 != b2, then a_i+b1 != a_j+b2 unless a_i - a_j = b2 - b1.
    # So collisions happen when a_i - a_j = b2 - b1 for some i,j.
    
    # Let's just do the search with time limits and see how far we get.
    
    # Precompute all primes up to 256
    # Already done in prime_lookup
    
    # Search A from 1 to 128, but be smart about ordering
    # Try smaller A first since sums a_i+b_j range from 2 to 256,
    # and prime density is higher for smaller numbers
    
    found_nine = False
    
    # Phase 1a: Search A with values in [1, 64]
    for a1 in range(1, 65):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, 65):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, 65):
                if time.time() > deadline:
                    break
                A = [a1, a2, a3]
                
                # Find all b in [1,128] where a1+b, a2+b, a3+b are all prime
                valid_b = []
                for b in range(1, 129):
                    if (a1 + b) in prime_lookup and (a2 + b) in prime_lookup and (a3 + b) in prime_lookup:
                        valid_b.append(b)
                
                if len(valid_b) < 3:
                    continue
                
                # Now find 3 b's from valid_b such that all 9 sums are distinct
                # Precompute sums for each valid b
                b_sums = {b: (a1 + b, a2 + b, a3 + b) for b in valid_b}
                
                # For each pair (b1, b2), check if the 6 sums are all distinct
                # Then for each such pair, try adding b3
                n_valid = len(valid_b)
                for i in range(n_valid):
                    b1 = valid_b[i]
                    s1 = b_sums[b1]
                    for j in range(i + 1, n_valid):
                        b2 = valid_b[j]
                        s2 = b_sums[b2]
                        # Check 6 sums are distinct
                        six_sums = set(s1 + s2)
                        if len(six_sums) < 6:
                            continue
                        # Try to add a third b
                        for k in range(j + 1, n_valid):
                            b3 = valid_b[k]
                            s3 = b_sums[b3]
                            nine_sums = set(s1 + s2 + s3)
                            if len(nine_sums) == 9:
                                # All 9 are distinct primes, hence pairwise coprime
                                B = [b1, b2, b3]
                                return [A, B]
    
    # Phase 1b: Search A with values in [64, 128]
    for a1 in range(64, 129):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, 129):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, 129):
                if time.time() > deadline:
                    break
                A = [a1, a2, a3]
                
                valid_b = []
                for b in range(1, 129):
                    if (a1 + b) in prime_lookup and (a2 + b) in prime_lookup and (a3 + b) in prime_lookup:
                        valid_b.append(b)
                
                if len(valid_b) < 3:
                    continue
                
                b_sums = {b: (a1 + b, a2 + b, a3 + b) for b in valid_b}
                
                n_valid = len(valid_b)
                for i in range(n_valid):
                    b1 = valid_b[i]
                    s1 = b_sums[b1]
                    for j in range(i + 1, n_valid):
                        b2 = valid_b[j]
                        s2 = b_sums[b2]
                        six_sums = set(s1 + s2)
                        if len(six_sums) < 6:
                            continue
                        for k in range(j + 1, n_valid):
                            b3 = valid_b[k]
                            s3 = b_sums[b3]
                            nine_sums = set(s1 + s2 + s3)
                            if len(nine_sums) == 9:
                                B = [b1, b2, b3]
                                return [A, B]
    
    # Phase 2: If we didn't find 9 distinct prime sums, try 9 distinct pairwise coprime sums
    # using a broader search. Allow prime powers as well.
    
    # For this, we need to check pairwise coprimality directly.
    # Let's do a structured random search with the best known fallback.
    
    import random
    rng = random.Random(42)
    
    attempts = 0
    while time.time() < deadline and attempts < 200000:
        # Random search
        A = sorted(rng.sample(range(1, 129), 3))
        B = sorted(rng.sample(range(1, 129), 3))
        
        sums = [A[i] + B[j] for i in range(3) for j in range(3)]
        distinct_sums = set(sums)
        count = len(distinct_sums)
        
        if count > best_count:
            if are_pairwise_coprime(distinct_sums):
                best_count = count
                best_A = A[:]
                best_B = B[:]
                if best_count == 9:
                    return [best_A, best_B]
        attempts += 1
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
