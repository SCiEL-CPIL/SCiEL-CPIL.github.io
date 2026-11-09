"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _sieve_prime_factors(max_n):
    """Precompute distinct prime factors for all numbers up to max_n using sieve."""
    factors = [set() for _ in range(max_n + 1)]
    for d in range(2, max_n + 1):
        if not factors[d]:
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

def _count_distinct_primes_incremental(arr, pf, pos, new_val, old_val):
    """Fast incremental count: only recompute pairs involving pos."""
    primes = set()
    n = len(arr)
    for i in range(n):
        if i == pos:
            continue
        for j in range(i + 1, n):
            if j == pos:
                continue
            s = arr[i] + arr[j]
            primes |= pf[s]
    # Add pairs involving pos
    for i in range(n):
        if i == pos:
            continue
        s = new_val + arr[i]
        primes |= pf[s]
    return len(primes)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Precompute prime factors for all sums (max sum = 20000)
    pf = _sieve_prime_factors(20000)
    
    # Structural approach: Try small prime sets as target, then find numbers
    # whose pairwise sums are P-smooth.
    
    # Key insight: if all numbers are ≡ r mod M where 2r ≡ 0 mod M,
    # then all pairwise sums ≡ 0 mod M. The remaining factors come from sum/M.
    
    # Best known: use M = 2*3 = 6, r = 0 or 3.
    # All numbers ≡ 0 mod 6: sums ≡ 0 mod 6, so factors include 2,3.
    # All numbers ≡ 3 mod 6: sums ≡ 6 ≡ 0 mod 6, same.
    
    # Try to find the optimal set by systematic search over APs with small d,
    # then do focused local search.
    
    best_arr = None
    best_count = float('inf')
    
    # Phase 1: Systematic AP search
    # For each small modulus d, try all residues r such that 2r ≡ 0 mod gcd(d,2)
    # Actually, just try all d up to 200 and all residues, take first 12 numbers.
    
    for d in range(2, 200):
        if time.time() > deadline:
            break
        for r in range(d):
            # First number >= 1
            first = r
            if first < 1:
                first += d
            arr = []
            val = first
            while val <= 10000 and len(arr) < 12:
                arr.append(val)
                val += d
            if len(arr) < 12:
                continue
            arr = arr[:12]
            count = _count_distinct_primes(arr, pf)
            if count < best_count:
                best_count = count
                best_arr = arr[:]
                if best_count <= 2:
                    return best_arr
    
    # Phase 2: Try products of small primes as modulus
    small_primes = [2, 3, 5, 7, 11, 13]
    from itertools import combinations
    
    for combo_size in range(1, 5):
        if time.time() > deadline:
            break
        for combo in combinations(small_primes, combo_size):
            M = 1
            for p in combo:
                M *= p
            if M > 10000:
                continue
            # Try residues r where 2r ≡ 0 mod M, i.e., r ≡ 0 mod M/gcd(2,M)
            # If M is even: r ≡ 0 mod M/2, so r = 0, M/2
            # If M is odd: r ≡ 0 mod M, so r = 0
            if M % 2 == 0:
                residues = [0, M // 2]
            else:
                residues = [0]
            for r in residues:
                first = r
                if first < 1:
                    first += M
                arr = []
                val = first
                while val <= 10000 and len(arr) < 12:
                    arr.append(val)
                    val += M
                if len(arr) < 12:
                    continue
                arr = arr[:12]
                count = _count_distinct_primes(arr, pf)
                if count < best_count:
                    best_count = count
                    best_arr = arr[:]
                    if best_count <= 2:
                        return best_arr
    
    # Phase 3: Focused local search with incremental evaluation
    import random
    random.seed(42)
    
    if best_arr is None:
        best_arr = list(range(1, 13))
        best_count = _count_distinct_primes(best_arr, pf)
    
    # Multiple restarts from good starting points
    for trial in range(30):
        if time.time() > deadline:
            break
        
        # Start from best_arr or a random AP
        if trial == 0:
            arr = best_arr[:]
        else:
            d = random.choice([2, 3, 4, 6, 12, 24, 30, 42, 60, 120, 210])
            r = random.randint(0, d - 1)
            first = r
            if first < 1:
                first += d
            arr = []
            val = first
            while val <= 10000 and len(arr) < 12:
                arr.append(val)
                val += d
            if len(arr) < 12:
                while len(arr) < 12:
                    x = random.randint(1, 10000)
                    if x not in arr:
                        arr.append(x)
            arr = arr[:12]
        
        count = _count_distinct_primes(arr, pf)
        if count < best_count:
            best_count = count
            best_arr = arr[:]
            if best_count <= 2:
                return best_arr
        
        # Local search: replace one element at a time
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for pos in range(12):
                old = arr[pos]
                # Try replacing with values that are "compatible"
                # Generate candidates: numbers that share common factors with existing sums
                candidates = set()
                # Try numbers near existing ones
                for delta in range(-50, 51, 10):
                    c = old + delta
                    if 1 <= c <= 10000 and c not in arr:
                        candidates.add(c)
                # Try random candidates
                for _ in range(30):
                    c = random.randint(1, 10000)
                    if c not in arr:
                        candidates.add(c)
                
                for c in list(candidates)[:50]:
                    arr[pos] = c
                    new_count = _count_distinct_primes(arr, pf)
                    if new_count < count:
                        count = new_count
                        improved = True
                        if count < best_count:
                            best_count = count
                            best_arr = arr[:]
                            if best_count <= 2:
                                return best_arr
                    else:
                        arr[pos] = old
                if time.time() > deadline - 5:
                    break
        
        if time.time() > deadline - 5:
            break
    
    return best_arr

# EVOLVE-BLOCK-END
