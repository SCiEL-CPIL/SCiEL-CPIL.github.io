"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 55  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute prime factors for each candidate
    prime_factors = {}
    for i in range(2, N + 1):
        n = i
        factors = set()
        d = 2
        while d * d <= n:
            if n % d == 0:
                factors.add(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            factors.add(n)
        prime_factors[i] = factors
    
    # Get all unique primes up to N
    all_primes = sorted(set().union(*[prime_factors[i] for i in range(2, N+1)]))
    prime_to_idx = {p: i for i, p in enumerate(all_primes)}
    
    # Represent each number as a bitmask of its prime factors
    def get_bitmask(n):
        if n == 1:
            return 0
        mask = 0
        for p in prime_factors[n]:
            mask |= (1 << prime_to_idx[p])
        return mask
    
    bitmasks = [get_bitmask(i) for i in range(1, N + 1)]
    
    # Greedy: sort by value descending
    sorted_candidates = sorted(range(1, N + 1), key=lambda x: -x)
    
    selected = set()
    
    for num in sorted_candidates:
        mask = bitmasks[num - 1]
        if mask == 0:
            selected.add(num)
            continue
        conflict = False
        for other in selected:
            if other == num:
                continue
            if (mask & bitmasks[other - 1]) != 0:
                conflict = True
                break
        if not conflict:
            selected.add(num)
    
    best_set = set(selected)
    best_sum = sum(best_set)
    
    def can_add(num, current_set):
        if num in current_set:
            return False
        mask = bitmasks[num - 1]
        if mask == 0:
            return True
        for other in current_set:
            if other == num:
                continue
            if (mask & bitmasks[other - 1]) != 0:
                return False
        return True
    
    # Aggressive local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = set(best_set)
        current_sum = best_sum
        
        # Try adding any unselected number
        for num in range(1, N + 1):
            if num in current_set:
                continue
            if can_add(num, current_set):
                current_set.add(num)
                current_sum += num
                improved = True
        
        # Try 1-for-1 exchanges
        if time.time() > deadline - 5:
            break
        for remove_num in list(current_set):
            for add_num in range(remove_num + 1, N + 1):
                if add_num in current_set:
                    continue
                temp_set = current_set - {remove_num}
                if can_add(add_num, temp_set):
                    current_set = temp_set
                    current_set.add(add_num)
                    current_sum += (add_num - remove_num)
                    improved = True
        
        # Try 2-for-1 exchanges (remove 2, add 1) - only if sum increases
        if time.time() > deadline - 15:
            break
        for r1, r2 in combinations(sorted(current_set), 2):
            if r1 + r2 >= N:
                continue
            for add_num in range(r1 + r2 + 1, N + 1):
                if add_num in current_set:
                    continue
                temp_set = current_set - {r1, r2}
                if can_add(add_num, temp_set):
                    current_set = temp_set
                    current_set.add(add_num)
                    current_sum += add_num - r1 - r2
                    improved = True
        
        # Try 1-for-2 exchanges (remove 1, add 2)
        if time.time() > deadline - 25:
            break
        for remove_num in list(current_set):
            for a1 in range(remove_num + 1, N + 1):
                if a1 in current_set:
                    continue
                temp_set = current_set - {remove_num}
                if not can_add(a1, temp_set):
                    continue
                for a2 in range(a1 + 1, N + 1):
                    if a2 in current_set or a2 == a1:
                        continue
                    if a1 + a2 <= remove_num:
                        continue
                    if can_add(a2, temp_set | {a1}):
                        current_set = temp_set
                        current_set.add(a1)
                        current_set.add(a2)
                        current_sum += a1 + a2 - remove_num
                        improved = True
                        break
        
        if improved:
            best_set = current_set
            best_sum = current_sum
    
    # Final validation
    result = sorted(best_set)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                return [1] + [a for a in range(2, N + 1) if len(prime_factors[a]) == 1 and prime_factors[a] and min(prime_factors[a]) > 1]
    
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
