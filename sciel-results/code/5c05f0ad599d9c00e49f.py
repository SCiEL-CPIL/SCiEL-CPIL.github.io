import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_lcms(numbers):
    """Precompute LCM table for all pairs in [1, 128]."""
    n = max(numbers)
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    return lcm_table

def _build_conflict_table(lcm_table, N=128):
    """
    Precompute conflict bitsets.
    conflict[x][a] is a bitmask of b such that (a,b,x) is a bad triple.
    """
    conflict = [[0] * (N + 1) for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x:
                continue
            la_x = lcm_table[a][x]
            mask = 0
            for b in range(1, N + 1):
                if b == x or b == a:
                    continue
                if lcm_table[a][b] == la_x == lcm_table[b][x]:
                    mask |= (1 << b)
            conflict[x][a] = mask
    return conflict

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin (90s limit)
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute LCM table
    lcm_table = _precompute_lcms(numbers)
    
    # Precompute conflict bitsets
    conflict = _build_conflict_table(lcm_table, N)
    
    def popcount(m):
        return bin(m).count('1')
    
    def can_add(x, selected_mask):
        """Check if x can be added to the set represented by selected_mask."""
        for a in range(1, N + 1):
            if selected_mask & (1 << a):
                if conflict[x][a] & selected_mask:
                    return False
        return True
    
    def mask_to_list(mask):
        return [i for i in range(1, N + 1) if mask & (1 << i)]
    
    def greedy(order, selected_mask):
        """Greedily add numbers in order to selected_mask."""
        for x in order:
            if selected_mask & (1 << x):
                continue
            if can_add(x, selected_mask):
                selected_mask |= (1 << x)
        return selected_mask
    
    # Strategy 1: Primes and Prime Powers first
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

    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = [x for x in numbers if is_prime(x)]
    prime_powers = [x for x in numbers if is_prime_power(x) and x not in primes]
    others = [x for x in numbers if x not in primes and x not in prime_powers]
    
    # Order 1: Primes, Prime Powers, Others
    order1 = primes + prime_powers + others
    mask1 = greedy(order1, 0)
    
    # Order 2: Standard 1..128
    mask2 = greedy(numbers, 0)
    
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
    mask3 = greedy(order3, 0)
    
    best_mask = max([mask1, mask2, mask3], key=popcount)
    best_size = popcount(best_mask)
    
    # Try many random shuffles
    random.seed(42)
    for i in range(10000):
        if time.time() > deadline - 20:  # Leave 20s for local search
            break
        random.shuffle(numbers)
        res_mask = greedy(numbers, 0)
        res_size = popcount(res_mask)
        if res_size > best_size:
            best_mask = res_mask
            best_size = res_size
    
    # Local search: try to add elements, then swap
    def local_search(mask, deadline_local):
        improved = True
        while improved:
            improved = False
            if time.time() > deadline_local:
                break
            # Try adding each excluded number
            for x in range(1, N + 1):
                if time.time() > deadline_local:
                    break
                if mask & (1 << x):
                    continue
                if can_add(x, mask):
                    mask |= (1 << x)
                    improved = True
                    break
            # Try swapping: remove one element, try to add a different one
            if not improved:
                selected = mask_to_list(mask)
                for i in range(len(selected)):
                    if time.time() > deadline_local:
                        break
                    candidate_remove = selected[i]
                    new_mask = mask & ~(1 << candidate_remove)
                    for x in range(1, N + 1):
                        if time.time() > deadline_local:
                            break
                        if new_mask & (1 << x):
                            continue
                        if can_add(x, new_mask):
                            new_mask |= (1 << x)
                            if popcount(new_mask) > popcount(mask):
                                mask = new_mask
                                improved = True
                                break
                    if improved:
                        break
        return mask
    
    if time.time() < deadline - 2:
        best_mask = local_search(best_mask, deadline - 2)
    
    return mask_to_list(best_mask)
