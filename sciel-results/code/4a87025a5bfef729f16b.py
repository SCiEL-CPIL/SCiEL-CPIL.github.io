import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_lcms(numbers):
    """Precompute LCM table for all pairs in [1, max(numbers)]."""
    n = max(numbers)
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    return lcm_table

def _greedy_construct_fast(order, lcm_table):
    """
    Greedily add numbers in the given order if they don't form a bad triple.
    Uses precomputed LCM table for speed.
    """
    selected = []
    for x in order:
        ok = True
        for i in range(len(selected)):
            a = selected[i]
            la = lcm_table[a][x]
            for j in range(i + 1, len(selected)):
                b = selected[j]
                if la == lcm_table[a][b] == lcm_table[b][x]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            selected.append(x)
    return selected

def _local_search_improve(selected, lcm_table, numbers, deadline):
    """
    Try to improve the subset by swapping or adding elements.
    Returns an improved subset if found.
    """
    selected_set = set(selected)
    best = list(selected)
    best_len = len(best)
    
    # Try adding elements not in the set
    for x in numbers:
        if time.time() > deadline:
            break
        if x in selected_set:
            continue
        # Check if x can be added
        ok = True
        for i in range(len(best)):
            a = best[i]
            la = lcm_table[a][x]
            for j in range(i + 1, len(best)):
                b = best[j]
                if la == lcm_table[a][b] == lcm_table[b][x]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            best.append(x)
            selected_set.add(x)
            best_len = len(best)
    
    # Try swaps: remove one element, add another
    for i in range(len(best)):
        if time.time() > deadline:
            break
        removed = best[i]
        # Temporarily remove
        test_list = best[:i] + best[i+1:]
        for x in numbers:
            if x in selected_set and x != removed:
                continue
            # Check if x can be added to test_list
            ok = True
            for j in range(len(test_list)):
                a = test_list[j]
                la = lcm_table[a][x]
                for k in range(j + 1, len(test_list)):
                    b = test_list[k]
                    if la == lcm_table[a][b] == lcm_table[b][x]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                new_list = test_list + [x]
                if len(new_list) > best_len:
                    best = new_list
                    best_len = len(best)
                    selected_set = set(best)
                    break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (limit is 90s)
    
    numbers = list(range(1, 129))
    lcm_table = _precompute_lcms(numbers)
    
    # Strategy 1: Primes and Prime Powers first
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    def is_prime_power(n):
        if n < 2:
            return False
        p = n
        while p > 1:
            if p * p > n:
                break
            if n % p == 0:
                n //= p
            else:
                return False
        return True

    primes = [x for x in numbers if is_prime(x)]
    prime_powers = [x for x in numbers if is_prime_power(x) and x not in primes]
    others = [x for x in numbers if x not in primes and x not in prime_powers]
    
    # Order 1: Primes, Prime Powers, Others (sorted by value)
    order1 = primes + prime_powers + others
    result1 = _greedy_construct_fast(order1, lcm_table)
    
    # Order 2: Standard 1..128
    order2 = numbers
    result2 = _greedy_construct_fast(order2, lcm_table)
    
    # Order 3: Sorted by number of divisors (ascending)
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    order3 = sorted(numbers, key=num_divisors)
    result3 = _greedy_construct_fast(order3, lcm_table)
    
    best_result = max([result1, result2, result3], key=len)
    
    # Try many random shuffles if time permits
    random.seed(42)
    for i in range(500):
        if time.time() > deadline:
            break
        shuffled = numbers[:]
        random.shuffle(shuffled)
        res = _greedy_construct_fast(shuffled, lcm_table)
        if len(res) > len(best_result):
            best_result = res
    
    # Local search improvement on the best result
    if time.time() < deadline - 5:
        improved = _local_search_improve(best_result, lcm_table, numbers, deadline)
        if len(improved) > len(best_result):
            best_result = improved
    
    return best_result
