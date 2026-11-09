"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start = time.time()
    deadline = start + 55  # Safety margin for 60s limit
    
    N = 256
    candidates = list(range(1, N + 1))
    
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
    num_primes = len(all_primes)
    
    # Represent each number as a bitmask of its prime factors
    # Use Python integers as bitsets
    def get_bitmask(n):
        if n == 1:
            return 0
        mask = 0
        for p in prime_factors[n]:
            mask |= (1 << prime_to_idx[p])
        return mask
    
    bitmasks = [get_bitmask(i) for i in range(1, N + 1)]
    
    # Greedy: start with all numbers, remove conflicts
    # We want to maximize sum, so we try to keep large numbers
    # Sort by value descending
    sorted_candidates = sorted(range(1, N + 1), key=lambda x: -x)
    
    # Try to build a set greedily
    selected = set()
    selected_masks = []
    
    for num in sorted_candidates:
        mask = bitmasks[num - 1]
        if mask == 0:  # num is 1, always coprime with everything
            selected.add(num)
            continue
        # Check if this number conflicts with any selected number
        conflict = False
        for sm in selected_masks:
            if mask & sm:
                conflict = True
                break
        if not conflict:
            selected.add(num)
            selected_masks.append(mask)
    
    best_set = set(selected)
    best_sum = sum(best_set)
    
    # Local search: try 1-for-1 exchanges and 2-for-1 exchanges
    def can_add(num, current_set):
        if num in current_set:
            return False
        mask = bitmasks[num - 1]
        if mask == 0:
            return True
        for other in current_set:
            if other == num:
                continue
            other_mask = bitmasks[other - 1]
            if mask & other_mask:
                return False
        return True
    
    # Iterative improvement
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
                new_sum = current_sum + num
                if new_sum > current_sum:
                    current_set.add(num)
                    current_sum = new_sum
                    improved = True
        
        # Try 1-for-1 exchanges: remove one, add another
        if time.time() > deadline - 5:
            break
        for remove_num in list(current_set):
            for add_num in range(1, N + 1):
                if add_num in current_set or add_num == remove_num:
                    continue
                # Check if add_num can be added after removing remove_num
                temp_set = current_set - {remove_num}
                if can_add(add_num, temp_set):
                    delta = add_num - remove_num
                    if delta > 0:
                        new_sum = current_sum + delta
                        if new_sum > current_sum:
                            current_set = temp_set
                            current_set.add(add_num)
                            current_sum = new_sum
                            improved = True
        
        # Try 2-for-1 exchanges: remove two, add one (if it increases sum)
        if time.time() > deadline - 10:
            break
        if improved:
            best_set = current_set
            best_sum = current_sum
    
    # Final validation
    result = sorted(best_set)
    # Verify pairwise coprimality
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                # Fallback to safe solution
                return [1] + [a for a in range(2, N + 1) if len(prime_factors[a]) == 1 and prime_factors[a] and min(prime_factors[a]) > 1]
    
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
