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
    
    def greedy_add(selected, candidate):
        """Check if candidate can be added to selected without creating a bad triple."""
        for a, b in combinations(selected, 2):
            if is_bad_triple(a, b, candidate):
                return False
        return True
    
    # Try multiple orderings and pick the best
    best_result = []
    
    # Ordering 1: Natural order (current approach)
    selected = []
    for x in numbers:
        if greedy_add(selected, x):
            selected.append(x)
    if len(selected) > len(best_result):
        best_result = selected
    
    # Ordering 2: Reverse order
    selected = []
    for x in reversed(numbers):
        if greedy_add(selected, x):
            selected.append(x)
    if len(selected) > len(best_result):
        best_result = selected
    
    # Ordering 3: Only numbers with few divisors (prime-like numbers first)
    # Numbers that are prime or have few divisors are less likely to form bad triples
    def divisor_count(x):
        count = 0
        for d in range(1, int(x**0.5) + 1):
            if x % d == 0:
                count += 1
                if d != x // d:
                    count += 1
        return count
    
    sorted_by_divs = sorted(numbers, key=lambda x: divisor_count(x))
    selected = []
    for x in sorted_by_divs:
        if time.time() > deadline:
            break
        if greedy_add(selected, x):
            selected.append(x)
    if len(selected) > len(best_result):
        best_result = selected
    
    # Ordering 4: Large numbers first (they have fewer multiples in range)
    selected = []
    for x in sorted(numbers, reverse=True):
        if time.time() > deadline:
            break
        if greedy_add(selected, x):
            selected.append(x)
    if len(selected) > len(best_result):
        best_result = selected
    
    # Ordering 5: Primes first, then composites
    def is_prime(x):
        if x < 2:
            return False
        for d in range(2, int(x**0.5) + 1):
            if x % d == 0:
                return False
        return True
    
    primes = [x for x in numbers if is_prime(x)]
    composites = [x for x in numbers if not is_prime(x)]
    order5 = primes + composites
    selected = []
    for x in order5:
        if time.time() > deadline:
            break
        if greedy_add(selected, x):
            selected.append(x)
    if len(selected) > len(best_result):
        best_result = selected
    
    # Ordering 6: Randomized greedy with multiple restarts
    import random
    random.seed(42)
    for restart in range(10):
        if time.time() > deadline:
            break
        order = numbers[:]
        random.shuffle(order)
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            if greedy_add(selected, x):
                selected.append(x)
        if len(selected) > len(best_result):
            best_result = selected
    
    # Final improvement: try to add any remaining numbers
    for x in numbers:
        if time.time() > deadline:
            break
        if x not in best_result:
            if greedy_add(best_result, x):
                best_result.append(x)
    
    return best_result
# EVOLVE-BLOCK-END
