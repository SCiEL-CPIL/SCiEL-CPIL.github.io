"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _sieve_prime_factors(max_n):
    """Precompute distinct prime factors for all numbers up to max_n using sieve."""
    factors = [set() for _ in range(max_n + 1)]
    for d in range(2, max_n + 1):
        # Check if d is prime
        if not factors[d]:
            # d is prime, mark all multiples
            for m in range(d, max_n + 1, d):
                factors[m].add(d)
    return factors

def _count_distinct_primes(arr, pf):
    """Count distinct primes dividing all pairwise sums using precomputed factors."""
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes |= pf[s]
    return len(primes)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Precompute prime factors for all sums (max sum = 20000)
    pf = _sieve_prime_factors(20000)
    
    # Strategy: Find numbers that are multiples of small primes
    # If all numbers are multiples of 2, all sums are even (factor 2)
    # If all numbers are multiples of 6, all sums are multiples of 6 (factors 2, 3)
    # But we need to minimize distinct primes across ALL sums
    
    # Best approach: find a set of numbers where all pairwise sums 
    # have prime factors from a small set
    
    # Try: all numbers ≡ r mod M where M is product of small primes
    # and 2r ≡ 0 mod each prime factor of M
    
    best_arr = list(range(1, 13))
    best_count = _count_distinct_primes(best_arr, pf)
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    
    # Try single prime moduli
    for p in small_primes:
        if time.time() > deadline:
            return best_arr
        residues = [0] if p != 2 else [0, 1]
        for r in residues:
            start = r % p
            if start == 0:
                start = p
            arr = []
            val = start
            while val <= 10000 and len(arr) < 12:
                arr.append(val)
                val += p
            if len(arr) >= 12:
                arr = arr[:12]
                count = _count_distinct_primes(arr, pf)
                if count < best_count:
                    best_count = count
                    best_arr = arr
                    if best_count <= 3:
                        return best_arr
    
    # Try products of 2 primes
    for i in range(len(small_primes)):
        if time.time() > deadline:
            return best_arr
        for j in range(i + 1, len(small_primes)):
            mod = small_primes[i] * small_primes[j]
            if mod > 10000:
                continue
            residues = [0]
            if mod % 2 == 0:
                residues.append(mod // 2)
            for r in residues:
                start = r % mod
                if start == 0:
                    start = mod
                arr = []
                val = start
                while val <= 10000 and len(arr) < 12:
                    arr.append(val)
                    val += mod
                if len(arr) >= 12:
                    arr = arr[:12]
                    count = _count_distinct_primes(arr, pf)
                    if count < best_count:
                        best_count = count
                        best_arr = arr
                        if best_count <= 3:
                            return best_arr
    
    # Try products of 3 primes
    for i in range(len(small_primes)):
        if time.time() > deadline:
            return best_arr
        for j in range(i + 1, len(small_primes)):
            for k in range(j + 1, len(small_primes)):
                mod = small_primes[i] * small_primes[j] * small_primes[k]
                if mod > 10000:
                    continue
                residues = [0]
                if mod % 2 == 0:
                    residues.append(mod // 2)
                for r in residues:
                    start = r % mod
                    if start == 0:
                        start = mod
                    arr = []
                    val = start
                    while val <= 10000 and len(arr) < 12:
                        arr.append(val)
                        val += mod
                    if len(arr) >= 12:
                        arr = arr[:12]
                        count = _count_distinct_primes(arr, pf)
                        if count < best_count:
                            best_count = count
                            best_arr = arr
                            if best_count <= 3:
                                return best_arr
    
    # Greedy local search: start with a good base, try swapping elements
    import random
    random.seed(42)
    
    # Try multiple random starting points with local improvement
    for trial in range(50):
        if time.time() > deadline:
            return best_arr
        
        # Generate a random candidate set
        mod = random.choice([2, 3, 4, 5, 6, 7, 10, 12, 14, 15, 20, 21, 30, 42, 60, 70, 105, 210])
        r = random.randint(0, mod - 1)
        start = r % mod
        if start == 0:
            start = mod
        arr = []
        val = start
        while val <= 10000 and len(arr) < 12:
            arr.append(val)
            val += mod
        
        if len(arr) < 12:
            # Fill with random numbers
            while len(arr) < 12:
                x = random.randint(1, 10000)
                if x not in arr:
                    arr.append(x)
        
        arr = arr[:12]
        count = _count_distinct_primes(arr, pf)
        
        if count < best_count:
            best_count = count
            best_arr = arr
            if best_count <= 3:
                return best_arr
        
        # Local search: try replacing one element
        improved = True
        while improved and time.time() < deadline:
            improved = False
            count = _count_distinct_primes(arr, pf)
            for pos in range(12):
                old = arr[pos]
                # Try a few replacements
                for _ in range(20):
                    new = random.randint(1, 10000)
                    if new == old or new in arr:
                        continue
                    arr[pos] = new
                    new_count = _count_distinct_primes(arr, pf)
                    if new_count < count:
                        count = new_count
                        improved = True
                        if count < best_count:
                            best_count = count
                            best_arr = arr[:]
                            if best_count <= 3:
                                return best_arr
                    else:
                        arr[pos] = old
                if time.time() > deadline:
                    return best_arr
    
    return best_arr

# EVOLVE-BLOCK-END
