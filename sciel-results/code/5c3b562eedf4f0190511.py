def construct():
    import time
    from math import gcd
    from itertools import combinations
    
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    
    def elapsed():
        return time.time() - start_time
    
    def is_valid_set(s):
        """Check if set s has no three distinct elements with all pairwise gcds equal."""
        for a, b, c in combinations(s, 3):
            g_ab = gcd(a, b)
            g_ac = gcd(a, c)
            g_bc = gcd(b, c)
            if g_ab == g_ac == g_bc:
                return False
        return True
    
    def can_add(s, x):
        """Check if adding x to set s maintains validity."""
        for a, b in combinations(s, 2):
            g_ab = gcd(a, b)
            g_ax = gcd(a, x)
            g_bx = gcd(b, x)
            if g_ab == g_ax == g_bx:
                return False
        return True
    
    # Strategy: Build a large set using numbers with unique prime factors
    # Then do local search to improve
    
    # First, try a greedy construction with multiple orderings
    best_set = set()
    
    # Ordering 1: numbers with large prime factors first
    def largest_prime_factor(n):
        lp = 1
        d = 2
        while d * d <= n:
            while n % d == 0:
                lp = d
                n //= d
            d += 1
        if n > 1:
            lp = n
        return lp
    
    # Try multiple orderings
    orderings = [
        # By largest prime factor descending
        sorted(range(1, 129), key=lambda x: (-largest_prime_factor(x), x)),
        # By number of prime factors ascending (primes and prime powers first)
        sorted(range(1, 129), key=lambda x: (x.bit_count(), x)),  # not exactly right
        # Primes first, then composites
        sorted(range(1, 129), key=lambda x: (0 if x in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127] else 1, x)),
        # Random-ish: interleave
        list(range(1, 129)),
        list(range(128, 0, -1)),
    ]
    
    for order in orderings:
        if elapsed() > DEADLINE:
            break
        s = set()
        for x in order:
            if elapsed() > DEADLINE:
                break
            if can_add(s, x):
                s.add(x)
        if len(s) > len(best_set):
            best_set = s.copy()
    
    # Local search: try to add more elements
    if elapsed() < DEADLINE:
        s = set(best_set)
        improved = True
        while improved and elapsed() < DEADLINE:
            improved = False
            for x in range(1, 129):
                if x in s:
                    continue
                if can_add(s, x):
                    s.add(x)
                    improved = True
                    break
    
    # Local search: try swaps
    if elapsed() < DEADLINE - 5:
        s = list(best_set)
        s_set = set(s)
        for _ in range(100):
            if elapsed() > DEADLINE - 5:
                break
            improved = False
            for i, a in enumerate(s):
                if elapsed() > DEADLINE - 5:
                    break
                for b in range(1, 129):
                    if b in s_set:
                        continue
                    # Try swapping a with b
                    new_s = s_set - {a} | {b}
                    if is_valid_set(new_s):
                        if len(new_s) > len(s_set):
                            s_set = new_s
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
    
    # Final greedy fill
    if elapsed() < DEADLINE:
        s = set(best_set)
        for x in range(1, 129):
            if elapsed() > DEADLINE:
                break
            if x not in s and can_add(s, x):
                s.add(x)
        if len(s) > len(best_set):
            best_set = s
    
    if not best_set:
        best_set = {1}
    
    return list(best_set)
