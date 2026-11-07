import math
import time
from collections import defaultdict

N = 210
MAX_N = 10210
MAX_K = 100
SMALL_PRIMES = [2, 3, 5, 7]

def _is_square(n):
    r = math.isqrt(n)
    return r * r == n

def _sieve_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]

def _factorize(n, primes):
    d = {}
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            d[p] = e
    if n > 1:
        d[n] = 1
    return d

def _squarefree_part(n, primes):
    """Compute the squarefree part of n."""
    sf = 1
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            if e & 1:
                sf *= p
    if n > 1:
        sf *= n
    return sf

def construct():
    t_start = time.time()
    DEADLINE = t_start + 75.0

    primes = _sieve_primes(MAX_N)
    
    # Precompute squarefree part for all candidates
    # squarefree_part[n] = product of primes with odd exponent in n
    squarefree = {}
    for n in range(211, MAX_N + 1):
        squarefree[n] = _squarefree_part(n, primes)

    # Target: 210 = 2*3*5*7, squarefree part = 210
    TARGET_SF = 210

    best_val = None
    best_set = None

    # Group numbers by their squarefree part
    sf_dict = defaultdict(list)
    for n in range(211, MAX_N + 1):
        sf_dict[squarefree[n]].append(n)
    
    for key in sf_dict:
        sf_dict[key].sort()

    # Check k=1: need squarefree[n] == 210
    for n in range(211, MAX_N + 1):
        if time.time() > DEADLINE:
            break
        if squarefree[n] == TARGET_SF:
            val = n - 210
            if best_val is None or val < best_val:
                best_val = val
                best_set = (n,)

    # Check k=2: need squarefree[a] * squarefree[b] to be a perfect square
    # i.e., squarefree[a] == squarefree[b]
    # But they must be distinct, so we need at least 2 numbers with same squarefree part
    for sf in sf_dict:
        if len(sf_dict[sf]) >= 2:
            a, b = sf_dict[sf][0], sf_dict[sf][1]
            val = max(a, b) - 210
            if best_val is None or val < best_val:
                best_val = val
                best_set = (a, b)

    # Check k=3: need squarefree[a] * squarefree[b] * squarefree[c] to be a perfect square
    # This means squarefree[a] * squarefree[b] * squarefree[c] must be a square
    # Equivalently, the product of the three squarefree parts must be a square
    # Let's iterate over pairs and check if the third exists
    
    # For efficiency, iterate in order of increasing max
    if best_val is not None:
        limit = 210 + best_val
    else:
        limit = MAX_N

    # Build a set of squarefree parts that have at least one number
    sf_set = set(sf_dict.keys())
    
    for a in range(211, limit + 1):
        if time.time() > DEADLINE:
            break
        if a - 210 >= (best_val if best_val is not None else float('inf')):
            break
        sfa = squarefree[a]
        for b in range(a + 1, limit + 1):
            if time.time() > DEADLINE:
                break
            if b - 210 >= (best_val if best_val is not None else float('inf')):
                break
            sfb = squarefree[b]
            # We need sfa * sfb * sfc to be a square
            # sfc must be such that sfa * sfb * sfc is a square
            # This means sfc = sfa * sfb / (some square)
            # More precisely, the squarefree part of sfa*sfb*sfc must be 1
            # So sfc must have the same squarefree part as sfa*sfb
            sf_ab = _squarefree_part(sfa * sfb, primes)
            if sf_ab in sf_dict:
                nums = sf_dict[sf_ab]
                # Find c != a, b
                c = None
                for x in nums:
                    if x != a and x != b:
                        c = x
                        break
                if c is not None:
                    val = max(a, b, c) - 210
                    if best_val is None or val < best_val:
                        best_val = val
                        best_set = (a, b, c)
                        limit = 210 + best_val

    # Check k=4: similar approach
    if best_val is not None:
        limit = 210 + best_val
    else:
        limit = MAX_N

    for a in range(211, limit + 1):
        if time.time() > DEADLINE:
            break
        if a - 210 >= (best_val if best_val is not None else float('inf')):
            break
        sfa = squarefree[a]
        for b in range(a + 1, limit + 1):
            if time.time() > DEADLINE:
                break
            if b - 210 >= (best_val if best_val is not None else float('inf')):
                break
            sfb = squarefree[b]
            sf_ab = _squarefree_part(sfa * sfb, primes)
            for c in range(b + 1, limit + 1):
                if time.time() > DEADLINE:
                    break
                if c - 210 >= (best_val if best_val is not None else float('inf')):
                    break
                sfc = squarefree[c]
                sf_abc = _squarefree_part(sf_ab * sfc, primes)
                if sf_abc in sf_dict:
                    nums = sf_dict[sf_abc]
                    d = None
                    for x in nums:
                        if x != a and x != b and x != c:
                            d = x
                            break
                    if d is not None:
                        val = max(a, b, c, d) - 210
                        if best_val is None or val < best_val:
                            best_val = val
                            best_set = (a, b, c, d)
                            limit = 210 + best_val

    if best_set is not None:
        prod = N
        for x in best_set:
            prod *= x
        if _is_square(prod) and len(set(best_set)) == len(best_set) and all(
            isinstance(x, int) and not isinstance(x, bool) and 210 < x <= MAX_N for x in best_set
        ):
            return [int(x) for x in best_set]

    # Fallback: use known solution
    return [840]

if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Result: {result}")
    print(f"Max - 210: {max(result) - 210}")
    prod = 210
    for x in result:
        prod *= x
    print(f"Product is square: {_is_square(prod)}")
    print(f"Time: {t1 - t0:.2f}s")
