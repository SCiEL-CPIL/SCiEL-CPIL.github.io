"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    # The problem reduces to finding the maximum size of a subset of {2, ..., 128}
    # such that any two distinct elements are coprime.
    # This is equivalent to finding the maximum number of distinct prime factors
    # that can be assigned to numbers <= 128.
    # The optimal set is the set of all primes <= 128.
    # For each prime p, the residue class is (0, p) or (p, p) doesn't work as residue must be < modulus.
    # Wait, the pair is [residue, modulus].
    # The class is {x : x % m == r}.
    # If we choose modulus = p (prime), we can choose residue = 0.
    # The class is multiples of p.
    # Two such classes (multiples of p1 and multiples of p2) are disjoint iff gcd(p1, p2) == 1.
    # Since p1 and p2 are distinct primes, gcd(p1, p2) = 1.
    # So the set of all primes <= 128 works.
    
    # Deadline guard (though this is O(1) effectively)
    start_time = time.time()
    deadline = start_time + 58.0  # 2s margin before 60s limit
    
    max_mod = 128
    primes = []
    for m in range(2, max_mod + 1):
        if time.time() > deadline:
            break
        is_prime = True
        for i in range(2, int(math.isqrt(m)) + 1):
            if m % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(m)
    
    # Construct the list of [residue, modulus] pairs
    # Residue 0 is canonical (0 <= 0 < m)
    result = [[0, p] for p in primes]
    
    return result
# EVOLVE-BLOCK-END
