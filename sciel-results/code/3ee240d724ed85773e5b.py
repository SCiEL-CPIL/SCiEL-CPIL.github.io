import time
from math import gcd
from itertools import combinations

def construct():
    deadline = time.time() + 85
    
    # Precompute gcd table
    n = 128
    nums = list(range(1, n+1))
    gcd_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def is_bad_triple(a, b, c):
        """Check if three distinct elements have all three pairwise gcds equal."""
        g1 = gcd_table[a][b]
        g2 = gcd_table[a][c]
        g3 = gcd_table[b][c]
        return g1 == g2 == g3
    
    def count_conflicts(selected_set):
        """Count number of bad triples in selected set."""
        s = list(selected_set)
        count = 0
        m = len(s)
        for i in range(m):
            for j in range(i+1, m):
                for k in range(j+1, m):
                    if is_bad_triple(s[i], s[j], s[k]):
                        count += 1
        return count
    
    def greedy_build(order):
        """Build a valid set greedily following the given order."""
        selected = set()
        for x in order:
            if time.time() > deadline:
                break
            ok = True
            for a, b in combinations(selected, 2):
                if is_bad_triple(a, b, x):
                    ok = False
                    break
            if ok:
                selected.add(x)
        return selected
    
    def local_search(selected_set):
        """Improve via swaps: remove one, add another if it reduces conflicts or keeps same size with fewer conflicts."""
        selected = set(selected_set)
        current_conflicts = count_conflicts(selected)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try all single swaps
            for rem in list(selected):
                for add in range(1, n+1):
                    if add in selected:
                        continue
                    new_set = (selected - {rem}) | {add}
                    if len(new_set) != len(selected):
                        continue
                    new_conflicts = count_conflicts(new_set)
                    if new_conflicts < current_conflicts:
                        selected = new_set
                        current_conflicts = new_conflicts
                        improved = True
                        break
                if improved:
                    break
        return selected
    
    # Try multiple greedy orders
    best = set()
    
    # Order 1: natural
    s1 = greedy_build(nums)
    s1 = local_search(s1)
    if len(s1) > len(best):
        best = s1
    
    # Order 2: reverse
    s2 = greedy_build(nums[::-1])
    s2 = local_search(s2)
    if len(s2) > len(best):
        best = s2
    
    # Order 3: prioritize numbers with fewer shared prime factors (more "coprime-like")
    # Score each number by how many other numbers share a prime factor with it
    from collections import defaultdict
    prime_factors = defaultdict(list)
    for x in range(1, n+1):
        d = 2
        while d*d <= x:
            if x % d == 0:
                prime_factors[d].append(x)
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            prime_factors[x].append(x)
    
    # Count for each number how many others share at least one prime factor
    share_count = [0]*(n+1)
    for p, lst in prime_factors.items():
        for x in lst:
            share_count[x] += len(lst) - 1
    
    # Sort by share_count ascending (prefer numbers with fewer shared factors)
    order3 = sorted(nums, key=lambda x: share_count[x])
    s3 = greedy_build(order3)
    s3 = local_search(s3)
    if len(s3) > len(best):
        best = s3
    
    # Order 4: prioritize primes and prime powers (they have fewer shared factors)
    def is_prime_power(x):
        if x <= 1:
            return False
        d = 2
        while d*d <= x:
            if x % d == 0:
                if x // d**2 != d:  # check if it's a pure power
                    temp = x
                    while temp % d == 0:
                        temp //= d
                    if temp != 1:
                        return False
                break
            d += 1
        return True
    
    # Actually, let's just prioritize numbers with small number of prime factors
    def num_prime_factors(x):
        count = 0
        d = 2
        while d*d <= x:
            if x % d == 0:
                count += 1
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            count += 1
        return count
    
    order4 = sorted(nums, key=lambda x: num_prime_factors(x))
    s4 = greedy_build(order4)
    s4 = local_search(s4)
    if len(s4) > len(best):
        best = s4
    
    # Final local search on the best
    if time.time() < deadline:
        best = local_search(best)
    
    return list(best)
