import time
import math
import itertools

DEADLINE = 140.0
MAX_N = 10**18

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

def _divisors_of(n):
    """Generate all divisors of n in sorted order."""
    # Factorize n
    factors = []
    temp = n
    for p in _SMALL_PRIMES:
        if p * p > temp:
            break
        if temp % p == 0:
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            factors.append((p, count))
    if temp > 1:
        factors.append((temp, 1))
        
    divs = [1]
    for p, e in factors:
        new_divs = []
        for d in divs:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divs = new_divs
    divs.sort()
    return divs

def _tau_perp(divs):
    c = 0
    for i in range(1, len(divs)):
        if math.gcd(divs[i-1], divs[i]) == 1:
            c += 1
    return c

def _score_n(n):
    if n > MAX_N:
        return -1
    divs = _divisors_of(n)
    return _tau_perp(divs)

def _divisors_of_squarefree(primes):
    """Generate all divisors of a squarefree number given its prime list."""
    divs = [1]
    for p in primes:
        divs = divs + [d * p for d in divs]
    divs.sort()
    return divs

def _score_primes(primes):
    """Score for squarefree number with given primes."""
    n = 1
    for p in primes:
        n *= p
    if n > MAX_N:
        return -1, n
    divs = _divisors_of_squarefree(primes)
    return _tau_perp(divs), n

def _local_search_exponents(n, primes):
    """Try to improve n by increasing exponents of primes in 'primes'."""
    # n is assumed to be squarefree with respect to primes in 'primes'
    # We will try to increase exponents of small primes
    # and remove large primes if needed
    
    best_n = n
    best_score = _score_n(n)
    
    # Factorize n to get current exponents
    # We know the primes. Let's just try multiplying by p
    # and dividing by q
    
    # Strategy:
    # 1. Try increasing exponent of smallest primes
    # 2. Try swapping a large prime for a square of a small prime
    
    changed = True
    while changed:
        changed = False
        # Try increasing exponent of each prime
        for p in primes:
            if n % p == 0:
                # p is in n
                if n * p <= MAX_N:
                    new_n = n * p
                    new_score = _score_n(new_n)
                    if new_score > best_score:
                        best_score = new_score
                        best_n = new_n
                        n = new_n
                        changed = True
            else:
                # p is not in n, maybe we can add it?
                # But we are focusing on improving existing structure
                pass
        
        # Try swapping: remove a prime q, add p^2?
        # This is more complex. Let's stick to simple exponent increases first.
        
    return best_n, best_score

def construct():
    start = time.time()
    
    best_score = 0
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19  # seed value
    
    # First, evaluate the seed
    seed_primes = [2, 3, 5, 7, 11, 13, 17, 19]
    s, n = _score_primes(seed_primes)
    best_score = s
    best_n = n
    
    # Search for best squarefree base
    max_k = 15
    
    # Evaluate greedy smallest primes
    for k in range(2, max_k + 1):
        if time.time() - start > DEADLINE - 20:
            break
        primes = _SMALL_PRIMES[:k]
        n = 1
        for p in primes:
            n *= p
            if n > MAX_N:
                break
        else:
            s, _ = _score_primes(primes)
            if s > best_score:
                best_score = s
                best_n = n
            continue

    # Local search on squarefree subsets
    for k in range(10, min(max_k + 1, 16)):
        if time.time() - start > DEADLINE - 20:
            break
        base = _SMALL_PRIMES[:k]
        base_n = 1
        for p in base:
            base_n *= p
        
        if base_n > MAX_N:
            continue
        
        s, _ = _score_primes(base)
        if s > best_score:
            best_score = s
            best_n = base_n
        
        # Try single swaps
        used = set(base)
        for i in range(k):
            for j in range(k, min(len(_SMALL_PRIMES), k + 10)):
                if _SMALL_PRIMES[j] in used:
                    continue
                new_primes = base[:i] + [_SMALL_PRIMES[j]] + base[i+1:]
                new_n = base_n // base[i] * _SMALL_PRIMES[j]
                if new_n > MAX_N:
                    continue
                s2, _ = _score_primes(new_primes)
                if s2 > best_score:
                    best_score = s2
                    best_n = new_n
                if time.time() - start > DEADLINE - 20:
                    break
            if time.time() - start > DEADLINE - 20:
                break

    # Now, try to improve the best squarefree candidate by adding exponents
    # We need the prime list of best_n
    # Re-factorize best_n
    temp = best_n
    factors = []
    for p in _SMALL_PRIMES:
        if p * p > temp:
            break
        if temp % p == 0:
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            factors.append((p, count))
    if temp > 1:
        factors.append((temp, 1))
        
    primes_in_best = [p for p, e in factors]
    
    # Apply local search on exponents
    improved_n, improved_score = _local_search_exponents(best_n, primes_in_best)
    if improved_score > best_score:
        best_score = improved_score
        best_n = improved_n

    # Also try to improve the 2nd best, 3rd best, etc.
    # We didn't store them. Let's just run local search on the top few we found.
    # To do this, we should have kept a list of top candidates.
    # For now, just do it on the global best.
    
    # One more check: try adding powers of 2 to a slightly smaller squarefree number
    # E.g. if best_n uses 15 primes, maybe a number with 14 primes and 2^3 is better?
    
    # Let's do a quick scan of "nearby" numbers
    # Take best_n, divide by largest prime, multiply by 2, 4, 8...
    # This is covered by _local_search_exponents if we allow removing primes?
    # My _local_search_exponents only increases exponents.
    
    # Let's add a "swap" step in local search
    def _local_search_swap(n, primes):
        best_n = n
        best_score = _score_n(n)
        changed = True
        while changed:
            changed = False
            # Get current factors
            temp = n
            factors = {}
            for p in _SMALL_PRIMES:
                if p * p > temp:
                    break
                if temp % p == 0:
                    count = 0
                    while temp % p == 0:
                        temp //= p
                        count += 1
                    factors[p] = count
            if temp > 1:
                factors[temp] = 1
            
            # Try to increase exponent of any prime in factors
            for p in factors:
                if n * p <= MAX_N:
                    new_n = n * p
                    new_score = _score_n(new_n)
                    if new_score > best_score:
                        best_score = new_score
                        best_n = new_n
                        n = new_n
                        changed = True
                        break # Restart loop with new n
            
            # Try to remove a prime and increase another?
            # This is harder. 
            # Let's just try: for each prime q in factors, if we remove it, can we add p^2?
            # Actually, just increasing exponents is a good first step.
            
        return best_n, best_score

    improved_n, improved_score = _local_search_swap(best_n, primes_in_best)
    if improved_score > best_score:
        best_score = improved_score
        best_n = improved_n

    return best_n
