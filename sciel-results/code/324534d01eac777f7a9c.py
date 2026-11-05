import time
from math import gcd
from itertools import combinations

def construct():
    deadline = time.time() + 85
    
    def is_bad_triple(a, b, c):
        return gcd(a, b) == gcd(a, c) == gcd(b, c)
    
    def has_conflict(selected, x):
        """Check if adding x to selected creates a bad triple."""
        for a, b in combinations(selected, 2):
            if is_bad_triple(a, b, x):
                return True
        return False
    
    def greedy_build(order):
        selected = []
        for x in order:
            if not has_conflict(selected, x):
                selected.append(x)
        return selected
    
    def local_search(selected, time_limit):
        selected = list(selected)
        best = len(selected)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try single swaps: remove one, add another
            for i, remove in enumerate(selected):
                candidate = selected[:i] + selected[i+1:]
                for add in range(1, 129):
                    if add in candidate:
                        continue
                    if not has_conflict(candidate, add):
                        new_size = len(candidate) + 1
                        if new_size > best:
                            selected = candidate + [add]
                            best = new_size
                            improved = True
                            break
                if time.time() >= time_limit:
                    break
        return selected
    
    # Multiple greedy strategies
    best_set = []
    
    # Strategy 1: ascending order
    s1 = greedy_build(range(1, 129))
    best_set = s1
    
    # Strategy 2: descending order
    s2 = greedy_build(range(128, 0, -1))
    if len(s2) > len(best_set):
        best_set = s2
    
    # Strategy 3: by number of divisors (fewer divisors first = safer)
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 2
                if i * i == n:
                    count -= 1
        return count
    
    order3 = sorted(range(1, 129), key=lambda x: (num_divisors(x), x))
    s3 = greedy_build(order3)
    if len(s3) > len(best_set):
        best_set = s3
    
    # Strategy 4: primes first (but we can only have at most 2 primes!)
    # Actually, let's do: numbers with few prime factors first
    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    order4 = sorted(range(1, 129), key=lambda x: (num_prime_factors(x), x))
    s4 = greedy_build(order4)
    if len(s4) > len(best_set):
        best_set = s4
    
    # Local search on the best
    if time.time() < deadline:
        best_set = local_search(best_set, deadline)
    
    return best_set
