"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from functools import lru_cache
from itertools import combinations

def construct():
    """
    Find 1..12 DISTINCT integers all greater than 12, with product exactly 12!.
    Minimize the LARGEST factor.
    """
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # 12! = 479001600
    # Prime factorization of 12!:
    # 2^10 * 3^5 * 5^2 * 7^1
    # Let's verify:
    # 2: floor(12/2)+floor(12/4)+floor(12/8) = 6+3+1 = 10
    # 3: floor(12/3)+floor(12/9) = 4+1 = 5
    # 5: floor(12/5) = 2
    # 7: floor(12/7) = 1
    # 11: floor(12/11) = 1
    # Wait, 11 is also a prime. 12! includes 11.
    # So 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    
    target = math.factorial(12)
    
    # Prime factorization of target
    # 2^10, 3^5, 5^2, 7^1, 11^1
    
    # We need to find distinct integers > 12 that multiply to target.
    # Since we want to minimize the largest factor, we want factors to be as small as possible (but > 12) and as many as possible (up to 12).
    
    # Strategy:
    # 1. Generate all possible divisors of 12! that are > 12.
    # 2. Use backtracking to find a set of distinct divisors > 12 that multiply to 12!.
    # 3. Among all valid sets, find the one with the minimum maximum element.
    
    # To make this efficient, we can use branch and bound.
    # Lower bound on the largest factor:
    # If we use k factors, the smallest possible product of k distinct integers > 12 is 13*14*...*12+k.
    # We need this product <= 12! for a solution to exist with k factors.
    
    # Let's find the maximum number of factors k_max such that 13*14*...*(12+k) <= 12!.
    # 13*14 = 182
    # 13*14*15 = 2730
    # 13*14*15*16 = 43680
    # 13*14*15*16*17 = 742560
    # 13*14*15*16*17*18 = 13366080
    # 13*14*15*16*17*18*19 = 253955520
    # 13*14*15*16*17*18*19*20 = 5079110400 > 479001600
    # So k can be at most 7.
    
    # We want to minimize the largest factor.
    # Let's try k from 7 down to 1, and for each k, try to find a set of k distinct integers > 12
    # with product 12! and minimum possible maximum.
    
    # Actually, it's better to search for the minimum possible maximum value M.
    # For a given M, check if there exists a set of distinct integers in (12, M] whose product is 12!.
    
    # But the space of M is large. Let's use a different approach.
    
    # Generate all divisors of 12! that are > 12.
    # There are (10+1)*(5+1)*(2+1)*(1+1)*(1+1) = 11*6*3*2*2 = 792 divisors.
    # This is small enough to work with.
    
    # Get all divisors
    divisors = []
    for a in range(11):  # 2^a
        for b in range(6):  # 3^b
            for c in range(3):  # 5^c
                for d in range(2):  # 7^d
                    for e in range(2):  # 11^e
                        val = (2**a) * (3**b) * (5**c) * (7**d) * (11**e)
                        if val > 12:
                            divisors.append(val)
    
    # Sort divisors
    divisors.sort()
    
    # We want to find a subset of distinct divisors from this list that multiply to 12!.
    # And we want to minimize the largest element in the subset.
    
    # Since the number of divisors is ~700, we can't do exhaustive search.
    # But we can use backtracking with pruning.
    
    # Let's use a recursive search that tries to build a set of factors.
    # We'll try to use as many factors as possible (to keep them small) but we must minimize the max.
    
    # Actually, let's think about it differently.
    # We want to minimize max(factor). Let's binary search on the answer M.
    # For a given M, we check if there's a valid set using only factors in (12, M].
    
    # But the number of factors is variable. Let's just do a DFS that tries to find any valid set,
    # and we'll order the factors in a way that helps us find good solutions early.
    
    # Let's try a different approach: since we want to minimize the max, let's try to find
    # a solution with the smallest possible max.
    
    # We can iterate over possible max values M, starting from the smallest possible,
    # and for each M, check if there's a valid decomposition using only factors <= M.
    
    # The smallest possible M is at least 13.
    # But we need the product to be 12!, so M must be at least as large as the largest factor in any valid decomposition.
    
    # Let's find the theoretical minimum M.
    # If we use k factors, the minimum possible max is at least the k-th smallest divisor > 12.
    # But we also need the product to be exactly 12!.
    
    # Let's just do a smart search.
    
    # Since the number of divisors is small (~700), we can precompute some things.
    
    # Let's use a backtracking algorithm that:
    # 1. Maintains the remaining product needed.
    # 2. Maintains the set of used factors.
    # 3. Tries to add factors in a way that keeps the max small.
    
    # We'll sort divisors and try to use small factors first, but we need to be careful
    # because using a small factor might force us to use a very large factor later.
    
    # Let's try a different strategy:
    # We know that the factors must be divisors of 12! and > 12.
    # Let's enumerate possible factor sets by considering the prime exponents.
    
    # Actually, let's just do a DFS with good pruning.
    
    # State: (remaining_product, min_allowed_factor, max_factor_so_far)
    # But this might be too slow.
    
    # Let's try a more constrained search.
    # Since 12! = 2^10 * 3^5 * 5^2 * 7 * 11, and all factors > 12,
    # each factor must be a divisor of 12! that is > 12.
    
    # Let's generate all possible factorizations by distributing the prime exponents.
    # But this is complex.
    
    # Let's try a simpler approach: since the number of divisors is small,
    # we can try all combinations of up to 7 divisors (since k_max = 7) and check
    # if their product is 12!.
    
    # But C(700, 7) is too large.
    
    # Let's use a meet-in-the-middle or a smarter DFS.
    
    # DFS approach:
    # - Start with remaining = 12!
    # - Try each divisor d > 12 that divides remaining
    # - Recurse with remaining/d
    # - Keep track of used factors to ensure distinctness
    # - Prune if remaining < smallest possible product of remaining factors
    
    # Let's implement this with memoization or just careful pruning.
    
    best_solution = None
    best_max = float('inf')
    
    # Sort divisors in ascending order
    # We'll try to build a set of factors
    
    # To minimize the max, we should try to use as many factors as possible,
    # and keep them as small as possible.
    
    # Let's try a DFS that tries to add factors in increasing order.
    
    def dfs(remaining, min_factor, current_set, current_max):
        nonlocal best_solution, best_max
        
        # Check time
        if time.time() > deadline:
            return
        
        # Pruning: if current_max >= best_max, no point continuing
        if current_max >= best_max:
            return
        
        # If remaining is 1, we have a complete solution
        if remaining == 1:
            if len(current_set) >= 1:  # Need at least 1 factor
                if current_max < best_max:
                    best_max = current_max
                    best_solution = list(current_set)
            return
        
        # If remaining is a divisor > 12 and not used, we can use it as the last factor
        if remaining > 12 and remaining >= min_factor and remaining not in current_set:
            if remaining > 12 and remaining >= min_factor:
                new_max = max(current_max, remaining)
                if new_max < best_max:
                    current_set.add(remaining)
                    dfs(1, remaining + 1, current_set, new_max)
                    current_set.remove(remaining)
        
        # Try to split remaining into two or more factors
        # For each divisor d of remaining such that d >= min_factor and d > 12
        # and d not in current_set
        for d in divisors:
            if d < min_factor:
                continue
            if d in current_set:
                continue
            if d > remaining:
                break
            if remaining % d != 0:
                continue
            
            new_remaining = remaining // d
            new_max = max(current_max, d)
            
            if new_max >= best_max:
                break  # Since divisors are sorted, larger d will only increase max
            
            # Pruning: new_remaining must be expressible as product of distinct factors > d
            # The smallest possible product of k factors > d is (d+1)*(d+2)*...
            # But we don't know k. Let's just check if new_remaining > 1 and new_remaining >= min_factor
            # Actually, new_remaining must be >= 1, and if new_remaining > 1, it must be >= min_factor
            # But min_factor for the next level is d+1
            
            if new_remaining == 1:
                # This is already handled above
                continue
            
            # Check if new_remaining can be formed by factors > d
            # The smallest factor we can use next is d+1
            # If new_remaining < d+1 and new_remaining != 1, it's impossible
            if new_remaining < d + 1 and new_remaining != 1:
                continue
            
            # Additional pruning: if new_remaining is prime and new_remaining < d+1, impossible
            # But new_remaining > d is required if new_remaining != 1
            
            if new_remaining > 1 and new_remaining < d + 1:
                continue
            
            current_set.add(d)
            dfs(new_remaining, d + 1, current_set, new_max)
            current_set.remove(d)
            
            # Early termination if we found a good solution
            if best_max <= d:
                break
    
    # Start DFS
    dfs(target, 13, set(), 0)
    
    if best_solution is not None:
        return best_solution
    else:
        # Fallback: return [12!] if no better solution found (should not happen)
        return [target]

# EVOLVE-BLOCK-END
