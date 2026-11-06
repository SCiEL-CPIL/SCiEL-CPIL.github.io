import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 75

    # Precompute compatibility: comp[x] is a set of y such that lcm(x,y) <= limit
    comp = [set() for _ in range(limit + 1)]
    for i in range(1, limit + 1):
        ci = comp[i]
        for j in range(i, limit + 1):
            g = gcd(i, j)
            lcm_val = i * j // g
            if lcm_val <= limit:
                ci.add(j)
                comp[j].add(i)

    def is_valid_add(S, x):
        cx = comp[x]
        return all(y in cx for y in S)

    def get_divisors(n):
        """Return all divisors of n."""
        divs = set()
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return divs

    def count_divisors(n):
        """Count the number of divisors of n."""
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    def is_valid_set(S):
        """Check if all pairs in S have lcm <= limit."""
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                a, b = S_list[i], S_list[j]
                g = gcd(a, b)
                if a * b // g > limit:
                    return False
        return True

    # Strategy 1: Find the best highly composite number whose divisors form a valid set
    # The divisors of any number n form a valid set because if d1|n and d2|n, then lcm(d1,d2)|n, so lcm(d1,d2) <= n
    best = set()
    
    # Try all numbers up to limit, find those with most divisors where all divisors are <= limit
    # and the divisors themselves form a valid set (which they always do if all divisors <= limit)
    for n in range(1, limit + 1):
        if time.time() > deadline - 30:
            break
        divs = get_divisors(n)
        # All divisors must be <= limit (they are by definition since d|n implies d<=n<=limit)
        # The set of divisors is valid because lcm(d1,d2) | n for any d1,d2 | n
        if len(divs) > len(best):
            best = divs

    # Strategy 2: Also try unions of divisor sets from different highly composite numbers
    # But first, let's also do a greedy completion on the best divisor set
    def complete(S):
        S = set(S)
        changed = True
        while changed:
            changed = False
            for x in range(1, limit + 1):
                if time.time() > deadline - 5:
                    return S
                if x not in S and is_valid_add(S, x):
                    S.add(x)
                    changed = True
        return S

    best = complete(best)

    # Strategy 3: Try starting from different highly composite numbers and do local search
    # Find top candidates by divisor count
    candidates = []
    for n in range(1, limit + 1):
        dc = count_divisors(n)
        candidates.append((dc, n))
    candidates.sort(reverse=True)
    
    # Try the top 20 candidates
    for dc, n in candidates[:20]:
        if time.time() > deadline - 10:
            break
        divs = get_divisors(n)
        S = complete(divs)
        if len(S) > len(best):
            best = S

    # Strategy 4: Local search with 1-for-1 swaps on the best solution
    import random
    rng = random.Random(42)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        S = set(best)
        for e in list(S):
            if time.time() > deadline - 5:
                break
            S_minus = S - {e}
            for x in range(1, limit + 1):
                if time.time() > deadline - 5:
                    break
                if x not in S and is_valid_add(S_minus, x):
                    S2 = complete(S_minus | {x})
                    if len(S2) > len(S):
                        best = S2
                        improved = True
                    break

    # Strategy 5: Random restarts with bias towards small numbers
    for it in range(30):
        if time.time() > deadline - 5:
            break
        seed = {1}
        for _ in range(rng.randint(3, 10)):
            x = rng.randint(2, limit)
            if is_valid_add(seed, x):
                seed.add(x)
        S = complete(seed)
        if len(S) > len(best):
            best = S

    return sorted(best)
