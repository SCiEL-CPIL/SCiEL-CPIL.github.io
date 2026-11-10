"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 75  # 75s safety margin (90s kill limit)
    
    def is_mean_of_subset(element, others):
        """Check if element is the arithmetic mean of any subset of others with size >= 2.
        Uses integer bitsets for fast DP."""
        n = len(others)
        if n < 2:
            return False
        target = element
        max_sum = sum(others)
        
        for k in range(2, n + 1):
            target_sum = k * target
            if target_sum > max_sum:
                break
            
            # Bitset DP: dp[j] is an integer where bit s is set if sum s is achievable with j elements
            dp = [0] * (k + 1)
            dp[0] = 1  # sum 0 achievable with 0 elements
            
            for val in others:
                for j in range(k, 0, -1):
                    if dp[j - 1]:
                        dp[j] |= (dp[j - 1] << val)
                # Early exit
                if (dp[k] >> target_sum) & 1:
                    return True
        
        return False
    
    def can_add_element(element, current_set):
        """Check if element can be added to current_set without violating the constraint."""
        # Check 1: is element the mean of any subset of current_set?
        if len(current_set) >= 2 and is_mean_of_subset(element, current_set):
            return False
        
        # Check 2: for each a in current_set, does adding element create a new mean violation?
        # a is mean of subset of (current_set - {a}) + {element} that includes element
        # => exists subset T of current_set - {a} with |T| >= 1 such that sum(T) = (|T|+1)*a - element
        for a in current_set:
            others_without_a = [x for x in current_set if x != a]
            n = len(others_without_a)
            if n == 0:
                continue
            
            max_sum = sum(others_without_a)
            for j in range(1, n + 1):
                target_sum = (j + 1) * a - element
                if target_sum <= 0 or target_sum > max_sum:
                    continue
                
                # Bitset DP
                dp = [0] * (j + 1)
                dp[0] = 1
                
                for val in others_without_a:
                    for jj in range(j, 0, -1):
                        if dp[jj - 1]:
                            dp[jj] |= (dp[jj - 1] << val)
                    if (dp[j] >> target_sum) & 1:
                        return False
        
        return True

    def is_valid_set(s):
        """Verify the entire set satisfies the constraint."""
        s_list = list(s)
        n = len(s_list)
        for i in range(n):
            element = s_list[i]
            others = [s_list[j] for j in range(n) if j != i]
            if is_mean_of_subset(element, others):
                return False
        return True

    # Strategy: Use a well-spread initial set, then greedily add
    best_set = []
    best_score = 0
    
    # Generate multiple candidate orderings
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0:
                return False
        return True
    
    orderings = []
    
    # Ordering 1: alternating small/large (spread out)
    alt = []
    for i in range(256):
        alt.append(i + 1)
        alt.append(512 - i)
    orderings.append(alt)
    
    # Ordering 2: descending
    orderings.append(list(range(512, 0, -1)))
    
    # Ordering 3: ascending
    orderings.append(list(range(1, 513)))
    
    # Ordering 4: multiples of large primes first
    def get_order_by_mod(mod):
        groups = [[] for _ in range(mod)]
        for x in range(1, 513):
            groups[x % mod].append(x)
        # Sort groups by size descending, interleave
        result = []
        while any(groups):
            for g in groups:
                if g:
                    result.append(g.pop(0))
        return result
    
    orderings.append(get_order_by_mod(7))
    orderings.append(get_order_by_mod(5))
    
    # Ordering 5: powers of 2 interleaved
    p2 = [2**i for i in range(9) if 2**i <= 512]
    rest = [x for x in range(1, 513) if x not in set(p2)]
    orderings.append(p2 + rest)
    
    # Ordering 6: random
    rnd = list(range(1, 513))
    random.seed(42)
    random.shuffle(rnd)
    orderings.append(rnd)
    
    # Ordering 7: primes first
    primes = [x for x in range(2, 513) if is_prime(x)]
    non_primes = [x for x in range(1, 513) if not is_prime(x)]
    orderings.append(primes + non_primes)
    
    # Ordering 8: odds then evens
    orderings.append([x for x in range(1, 513) if x % 2 == 1] + [x for x in range(1, 513) if x % 2 == 0])
    
    # Ordering 9: Fibonacci-like spread
    fib = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    fib = [x for x in fib if x <= 512]
    rest_fib = [x for x in range(1, 513) if x not in set(fib)]
    orderings.append(fib + rest_fib)
    
    # Ordering 10: geometric progression
    geo = []
    val = 1
    while val <= 512:
        geo.append(val)
        val = int(val * 1.5) + 1
    geo = list(set(geo))
    rest_geo = [x for x in range(1, 513) if x not in set(geo)]
    orderings.append(geo + rest_geo)
    
    # Ordering 11: random with different seed
    rnd2 = list(range(1, 513))
    random.seed(123)
    random.shuffle(rnd2)
    orderings.append(rnd2)
    
    # Ordering 12: random with different seed
    rnd3 = list(range(1, 513))
    random.seed(999)
    random.shuffle(rnd3)
    orderings.append(rnd3)
    
    # Ordering 13: spread by 13
    orderings.append(get_order_by_mod(13))
    
    # Ordering 14: spread by 11
    orderings.append(get_order_by_mod(11))
    
    # Ordering 15: composite numbers first
    composites = [x for x in range(4, 513) if not is_prime(x) and x > 1]
    primes_only = [x for x in range(2, 513) if is_prime(x)]
    ones = [1]
    orderings.append(composites + primes_only + ones)
    
    for ordering in orderings:
        if time.time() > deadline - 8:
            break
        
        selected = []
        selected_set = set()
        
        for x in ordering:
            if time.time() > deadline - 8:
                break
            if x in selected_set:
                continue
            
            if can_add_element(x, selected):
                selected.append(x)
                selected_set.add(x)
        
        if len(selected) > len(best_set):
            best_set = selected[:]
            best_score = len(best_set)
            if len(best_set) >= 48:
                break
    
    # Local improvement: try to add more elements from remaining
    if best_set and time.time() < deadline - 15:
        improved = best_set[:]
        improved_set = set(improved)
        remaining = [x for x in range(1, 513) if x not in improved_set]
        
        for x in remaining:
            if time.time() > deadline - 8:
                break
            if can_add_element(x, improved):
                improved.append(x)
                improved_set.add(x)
        
        if len(improved) > len(best_set):
            best_set = improved[:]
            best_score = len(best_set)
    
    # Try single-element swaps to improve
    if best_set and time.time() < deadline - 20:
        for i in range(len(best_set)):
            if time.time() > deadline - 8:
                break
            removed = best_set[i]
            without_i = [x for j, x in enumerate(best_set) if j != i]
            without_i_set = set(without_i)
            remaining2 = [x for x in range(1, 513) if x not in without_i_set and x != removed]
            
            # Try to add as many as possible
            new_set = without_i[:]
            new_set_set = without_i_set.copy()
            added = 0
            for x in remaining2:
                if time.time() > deadline - 8:
                    break
                if can_add_element(x, new_set):
                    new_set.append(x)
                    new_set_set.add(x)
                    added += 1
            
            if added > 1:  # Net gain of at least 1
                best_set = new_set[:]
                best_score = len(best_set)
                break
    
    # Try to find even larger sets using randomized local search
    if best_set and time.time() < deadline - 25:
        current_best = best_set[:]
        for _ in range(10):
            if time.time() > deadline - 8:
                break
            
            # Start from a random subset of current best
            if len(current_best) > 10:
                start_size = max(5, len(current_best) // 2)
                start_set = random.sample(current_best, start_size)
            else:
                start_set = current_best[:]
            
            start_set_set = set(start_set)
            remaining3 = [x for x in range(1, 513) if x not in start_set_set]
            
            for x in remaining3:
                if time.time() > deadline - 8:
                    break
                if can_add_element(x, start_set):
                    start_set.append(x)
                    start_set_set.add(x)
            
            if len(start_set) > len(best_set):
                best_set = start_set[:]
                best_score = len(best_set)
    
    # Final verification
    if best_set and not is_valid_set(best_set):
        # Fall back to a simple valid set
        simple = [1, 3, 7, 15, 31, 63, 127, 255, 511]
        for x in range(2, 513):
            if x not in set(simple) and can_add_element(x, simple):
                simple.append(x)
        best_set = simple
    
    return best_set
# EVOLVE-BLOCK-END
