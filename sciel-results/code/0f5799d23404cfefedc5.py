"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    n = 256
    deadline = time.time() + 50  # safety margin before 90s kill
    
    # Precompute gcd table or just use math.gcd
    # Strategy: greedy descending, but try multiple orderings
    
    def try_ordering(order):
        selected = []
        for a in order:
            if time.time() > deadline:
                break
            can_add = True
            for b in selected:
                if math.gcd(a, b) > 1:
                    can_add = False
                    break
            if can_add:
                selected.append(a)
        return selected
    
    # Ordering 1: descending (largest first)
    order1 = list(range(n, 0, -1))
    result1 = try_ordering(order1)
    sum1 = sum(result1)
    
    best = result1
    best_sum = sum1
    
    # Ordering 2: try to prefer numbers with large prime factors
    # For each number, compute its largest prime factor, sort by that descending
    def largest_prime_factor(x):
        if x <= 1:
            return 1
        orig = x
        lp = 1
        d = 2
        while d * d <= x:
            while x % d == 0:
                lp = d
                x //= d
            d += 1
        if x > 1:
            lp = x
        return lp
    
    order2 = sorted(range(n, 0, -1), key=lambda x: (-largest_prime_factor(x), -x))
    result2 = try_ordering(order2)
    sum2 = sum(result2)
    if sum2 > best_sum:
        best = result2
        best_sum = sum2
    
    # Ordering 3: numbers that are prime powers (high value per prime used)
    # Actually, let's try: sort by value/number_of_prime_factors descending
    def num_prime_factors(x):
        count = 0
        d = 2
        while d * d <= x:
            if x % d == 0:
                count += 1
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            count += 1
        return count
    
    order3 = sorted(range(n, 0, -1), key=lambda x: (-x / max(1, num_prime_factors(x)), -x))
    result3 = try_ordering(order3)
    sum3 = sum(result3)
    if sum3 > best_sum:
        best = result3
        best_sum = sum3
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
