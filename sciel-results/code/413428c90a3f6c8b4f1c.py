import math
import time
from itertools import combinations

def _prime_factors(n):
    """Return set of distinct prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _count_distinct_primes(arr):
    """Count distinct primes dividing all pairwise sums of distinct elements."""
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes |= _prime_factors(s)
    return len(primes)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_arr = list(range(1, 13))
    best_count = _count_distinct_primes(best_arr)
    
    # Strategy 1: Residue classes
    # Try moduli that are products of small primes, with various residues
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    moduli_to_try = set()
    # Single primes
    for p in small_primes:
        moduli_to_try.add(p)
    # Products of 2
    for i in range(len(small_primes)):
        for j in range(i + 1, len(small_primes)):
            m = small_primes[i] * small_primes[j]
            if m <= 10000:
                moduli_to_try.add(m)
    # Products of 3
    for i in range(len(small_primes)):
        for j in range(i + 1, len(small_primes)):
            for k in range(j + 1, len(small_primes)):
                m = small_primes[i] * small_primes[j] * small_primes[k]
                if m <= 10000:
                    moduli_to_try.add(m)
    # Also try some composite moduli
    for m in range(2, 500):
        moduli_to_try.add(m)
    
    moduli_to_try = sorted(moduli_to_try)
    
    for modulus in moduli_to_try:
        if time.time() > deadline:
            return best_arr
        # Try a limited number of residues
        max_res = min(modulus, 50)
        for residue in range(max_res):
            if time.time() > deadline:
                return best_arr
            start = residue
            if start == 0:
                start = modulus
            if start > 10000:
                continue
            arr = []
            val = start
            while val <= 10000 and len(arr) < 12:
                arr.append(val)
                val += modulus
            if len(arr) < 12:
                continue
            count = _count_distinct_primes(arr)
            if count < best_count:
                best_count = count
                best_arr = arr
                if best_count <= 3:
                    return best_arr
    
    # Strategy 2: Smooth number search
    def is_smooth(n, allowed_primes):
        for p in allowed_primes:
            while n % p == 0:
                n //= p
        return n == 1
    
    prime_sets_to_try = [
        [2],
        [2, 3],
        [2, 5],
        [2, 7],
        [2, 3, 5],
        [2, 3, 7],
        [2, 3, 11],
        [2, 5, 7],
        [2, 5, 11],
        [2, 7, 11],
        [3, 5, 7],
        [2, 3, 5, 7],
        [2, 3, 5, 11],
        [2, 3, 7, 11],
        [2, 5, 7, 11],
    ]
    
    for primes_set in prime_sets_to_try:
        if time.time() > deadline:
            return best_arr
        if len(primes_set) >= best_count:
            continue
        
        # Find candidates where 2x is smooth
        candidates = []
        for x in range(1, 10001):
            if is_smooth(2 * x, primes_set):
                candidates.append(x)
            if len(candidates) >= 1000:
                break
        
        if len(candidates) < 12:
            continue
        
        # Greedy clique with multiple starting points
        best_clique = []
        for start_idx in range(0, min(len(candidates), 200), 10):
            if time.time() > deadline:
                return best_arr
            clique = [candidates[start_idx]]
            for x in candidates[start_idx + 1:]:
                if len(clique) == 12:
                    break
                can_add = True
                for y in clique:
                    if not is_smooth(x + y, primes_set):
                        can_add = False
                        break
                if can_add:
                    clique.append(x)
            if len(clique) > len(best_clique):
                best_clique = clique
        
        if len(best_clique) == 12:
            count = _count_distinct_primes(best_clique)
            if count < best_count:
                best_count = count
                best_arr = best_clique
                if best_count <= 3:
                    return best_arr
    
    # Strategy 3: Random search with local search
    import random
    random.seed(42)
    attempts = 0
    while time.time() < deadline and attempts < 30000:
        attempts += 1
        arr = random.sample(range(1, 10001), 12)
        count = _count_distinct_primes(arr)
        if count < best_count:
            best_count = count
            best_arr = arr
            if best_count <= 3:
                return best_arr
        else:
            # Local improvement
            improved = True
            local_count = count
            while improved and time.time() < deadline:
                improved = False
                for i in range(12):
                    for _ in range(5):
                        new_val = random.randint(1, 10000)
                        if new_val in arr and new_val != arr[i]:
                            continue
                        new_arr = arr.copy()
                        new_arr[i] = new_val
                        if len(set(new_arr)) != 12:
                            continue
                        new_count = _count_distinct_primes(new_arr)
                        if new_count < local_count:
                            local_count = new_count
                            arr = new_arr
                            improved = True
                            if local_count < best_count:
                                best_count = local_count
                                best_arr = arr
                                if best_count <= 3:
                                    return best_arr
                            break
    
    return best_arr
