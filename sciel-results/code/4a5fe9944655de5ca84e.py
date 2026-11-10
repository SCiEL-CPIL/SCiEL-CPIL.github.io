"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    n = 128
    numbers = list(range(1, n + 1))

    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l

    # Precompute for each x, the set of pairs (a,b) with a<b, a!=x, b!=x
    # such that lcm(a,b) == lcm(a,x) == lcm(b,x)
    bad_pairs_for_x = [set() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for a in range(1, n + 1):
            if a == x:
                continue
            for b in range(a + 1, n + 1):
                if b == x:
                    continue
                lab = lcm_table[a][b]
                lxa = lcm_table[x][a]
                lxb = lcm_table[x][b]
                if lab == lxa == lxb:
                    bad_pairs_for_x[x].add((a, b))

    def can_add(selected_set, x):
        """Check if x can be added to selected_set without creating a bad triple."""
        for a, b in bad_pairs_for_x[x]:
            if a in selected_set and b in selected_set:
                return False
        return True

    best_result = set()

    def try_ordering(order):
        selected = set()
        for x in order:
            if time.time() > deadline:
                break
            if can_add(selected, x):
                selected.add(x)
        return selected

    # Ordering 1: Natural order
    selected = try_ordering(numbers)
    if len(selected) > len(best_result):
        best_result = selected

    # Ordering 2: Reverse order
    selected = try_ordering(reversed(numbers))
    if len(selected) > len(best_result):
        best_result = selected

    # Ordering 3: By divisor count (fewer divisors first)
    def divisor_count(x):
        count = 0
        for d in range(1, int(x**0.5) + 1):
            if x % d == 0:
                count += 1
                if d != x // d:
                    count += 1
        return count

    sorted_by_divs = sorted(numbers, key=lambda x: divisor_count(x))
    selected = try_ordering(sorted_by_divs)
    if len(selected) > len(best_result):
        best_result = selected

    # Ordering 4: Large numbers first
    selected = try_ordering(sorted(numbers, reverse=True))
    if len(selected) > len(best_result):
        best_result = selected

    # Ordering 5: Primes first
    def is_prime(x):
        if x < 2:
            return False
        for d in range(2, int(x**0.5) + 1):
            if x % d == 0:
                return False
        return True

    primes = [x for x in numbers if is_prime(x)]
    composites = [x for x in numbers if not is_prime(x)]
    selected = try_ordering(primes + composites)
    if len(selected) > len(best_result):
        best_result = selected

    # Randomized greedy with multiple restarts
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline:
            break
        order = numbers[:]
        random.shuffle(order)
        selected = try_ordering(order)
        if len(selected) > len(best_result):
            best_result = selected

    # Local search: try to add any remaining numbers
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in numbers:
            if x not in best_result:
                if can_add(best_result, x):
                    best_result.add(x)
                    improved = True

    # Greedy-remove-then-add-back local search
    for _ in range(200):
        if time.time() > deadline:
            break
        
        best_swap = None
        best_swap_size = len(best_result)
        
        # Try removing each element and see if we can get a larger set
        for remove_x in list(best_result):
            trial = best_result - {remove_x}
            
            # Try adding elements back in a specific order
            # We want to maximize the size, so try adding all possible elements
            for x in numbers:
                if x not in trial and x != remove_x:
                    if can_add(trial, x):
                        trial.add(x)
            
            if len(trial) > best_swap_size:
                best_swap_size = len(trial)
                best_swap = trial
        
        if best_swap is not None:
            best_result = best_swap
        else:
            break

    # Final pass: try to add any remaining numbers
    for x in numbers:
        if time.time() > deadline:
            break
        if x not in best_result:
            if can_add(best_result, x):
                best_result.add(x)

    return list(best_result)
# EVOLVE-BLOCK-END
