"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l
    
    def is_bad_triple(a, b, c):
        """Check if three distinct elements have equal pairwise LCMs."""
        lab = lcm_table[a][b]
        lac = lcm_table[a][c]
        lbc = lcm_table[b][c]
        return lab == lac == lbc
    
    def is_compatible(x, selected):
        """Check if x can be added to selected without creating a bad triple."""
        for a, b in combinations(selected, 2):
            if is_bad_triple(a, b, x):
                return False
        return True
    
    def greedy_solve(order):
        """Greedy: add elements in given order if compatible."""
        selected = []
        for x in order:
            if is_compatible(x, selected):
                selected.append(x)
        return selected
    
    def local_improve(selected):
        """Try to improve by swapping or adding elements."""
        selected = list(selected)
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            # Try adding any missing element
            in_set = set(selected)
            for x in range(1, n + 1):
                if x not in in_set and is_compatible(x, selected):
                    selected.append(x)
                    in_set.add(x)
                    improved = True
                    if time.time() > deadline - 5:
                        break
            # Try swapping: remove one, see if we can add two
            if time.time() > deadline - 5:
                break
            for i, a in enumerate(list(selected)):
                if time.time() > deadline - 5:
                    break
                temp = selected[:i] + selected[i+1:]
                temp_set = set(temp)
                # Try to add elements not in temp
                added = 0
                for x in range(1, n + 1):
                    if x not in temp_set and is_compatible(x, temp):
                        temp.append(x)
                        temp_set.add(x)
                        added += 1
                        if added >= 2:
                            break
                if len(temp) > len(selected):
                    selected = temp
                    improved = True
                    break
        return selected
    
    # Try multiple orderings
    best = []
    
    # Order 1: natural order
    if time.time() < deadline - 10:
        result = greedy_solve(numbers)
        if len(result) > len(best):
            best = result
    
    # Order 2: reverse
    if time.time() < deadline - 10:
        result = greedy_solve(list(reversed(numbers)))
        if len(result) > len(best):
            best = result
    
    # Order 3: by number of divisors (fewer first - "simpler" numbers)
    if time.time() < deadline - 10:
        def num_divisors(x):
            count = 0
            for d in range(1, int(x**0.5) + 1):
                if x % d == 0:
                    count += 1
                    if d != x // d:
                        count += 1
            return count
        order = sorted(numbers, key=num_divisors)
        result = greedy_solve(order)
        if len(result) > len(best):
            best = result
    
    # Order 4: by prime factorization complexity (number of distinct prime factors)
    if time.time() < deadline - 10:
        def num_distinct_primes(x):
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
        order = sorted(numbers, key=num_distinct_primes)
        result = greedy_solve(order)
        if len(result) > len(best):
            best = result
    
    # Order 5: by value of lcm with 1..n (more "isolated" first)
    if time.time() < deadline - 10:
        def lcm_complexity(x):
            total = 0
            for y in range(1, n + 1):
                if y != x:
                    total += lcm_table[x][y]
            return total
        order = sorted(numbers, key=lcm_complexity)
        result = greedy_solve(order)
        if len(result) > len(best):
            best = result
    
    # Local improvement on best
    if time.time() < deadline - 10:
        best = local_improve(best)
    
    # If we have time, try more random-ish orderings
    if time.time() < deadline - 20:
        import random
        random.seed(42)
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            order = list(numbers)
            random.shuffle(order)
            result = greedy_solve(order)
            if len(result) > len(best):
                best = result
    
    # Final local improvement
    if time.time() < deadline - 5:
        best = local_improve(best)
    
    return best
# EVOLVE-BLOCK-END
