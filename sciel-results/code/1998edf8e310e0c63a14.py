import time
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 85
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute conflicts: conflicts[x] = set of frozenset({y,z}) where {x,y,z} is bad
    conflicts = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_table[a][b]
            for c in range(b + 1, N + 1):
                gbc = gcd_table[b][c]
                if gab == gbc:
                    gca = gcd_table[a][c]
                    if gca == gab:
                        conflicts[a].add(frozenset((b, c)))
                        conflicts[b].add(frozenset((a, c)))
                        conflicts[c].add(frozenset((a, b)))
    
    def can_add(x, selected):
        for y in selected:
            for z in selected:
                if y == z:
                    continue
                if frozenset((y, z)) in conflicts[x]:
                    return False
        return True
    
    def greedy(order):
        selected = set()
        for x in order:
            if can_add(x, selected):
                selected.add(x)
        return selected
    
    best = set()
    
    # Multiple orderings
    def num_distinct_primes(n):
        if n == 1:
            return 0
        m = n
        d = 2
        count = 0
        while d * d <= m:
            if m % d == 0:
                count += 1
                while m % d == 0:
                    m //= d
            d += 1
        if m > 1:
            count += 1
        return count
    
    def largest_prime(n):
        if n == 1:
            return 1
        m = n
        d = 2
        lp = 1
        while d * d <= m:
            if m % d == 0:
                lp = d
                while m % d == 0:
                    m //= d
            d += 1
        if m > 1:
            lp = m
        return lp
    
    orderings = [
        numbers,
        list(reversed(numbers)),
        sorted(numbers, key=num_distinct_primes),
        sorted(numbers, key=lambda x: -num_distinct_primes(x)),
        sorted(numbers, key=largest_prime),
        sorted(numbers, key=lambda x: -largest_prime(x)),
        sorted(numbers, key=lambda x: (num_distinct_primes(x), x)),
        sorted(numbers, key=lambda x: (-largest_prime(x), x)),
    ]
    
    for order in orderings:
        if time.time() - start_time > DEADLINE:
            break
        s = greedy(order)
        if len(s) > len(best):
            best = s
    
    # Local search: add
    def local_search_add(s):
        s = set(s)
        improved = True
        while improved and time.time() - start_time < DEADLINE:
            improved = False
            for x in numbers:
                if x in s:
                    continue
                if can_add(x, s):
                    s.add(x)
                    improved = True
        return s
    
    if time.time() - start_time < DEADLINE:
        best = local_search_add(best)
    
    # Local search: swap
    def local_search_swap(s):
        s = set(s)
        improved = True
        while improved and time.time() - start_time < DEADLINE:
            improved = False
            for x in numbers:
                if x in s:
                    continue
                for y in sorted(s):
                    s_trial = s - {y}
                    if can_add(x, s_trial):
                        s = s_trial | {x}
                        improved = True
                        break
                if improved:
                    break
        return s
    
    if time.time() - start_time < DEADLINE:
        best = local_search_swap(best)
    
    return sorted(best)
