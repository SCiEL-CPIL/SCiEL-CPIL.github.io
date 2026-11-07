import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # 1. Generate Primes up to 256
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    primes = get_primes(256)
    small_primes = [p for p in primes if p <= 16] # 2, 3, 5, 7, 11, 13
    large_primes = [p for p in primes if p > 16]
    
    best_set = set(large_primes) # Initial valid set (size 48)
    # We can also just return all primes as baseline
    baseline_size = len(primes) # 54
    best_solution = primes
    
    # 2. Generate candidate pool U for composites/small primes
    # U contains numbers in [2, 256] whose prime factors are all in small_primes
    # Note: We can also include the small primes themselves in C
    # If we include a small prime p in C, it is NOT in large_primes, so no conflict.
    # But it must satisfy conditions with other elements in C.
    
    U = []
    for x in range(2, 257):
        y = x
        valid = True
        for p in small_primes:
            while y % p == 0:
                y //= p
        if y == 1:
            U.append(x)
    
    # Precompute prime factorization (as dict or just list of primes with multiplicity? 
    # For divisibility a | b, we just need to check a | b.
    # For a | b*c, we need to check a | b*c.
    
    # Sort U to help pruning? Smaller numbers have more multiples.
    # Usually, in antichain problems, picking larger numbers is safer?
    # Or smaller?
    # Let's just iterate.
    
    def is_valid_cand(a, b, c):
        # Check if a divides b*c
        # b*c can be up to 256*256 = 65536. Fits in int.
        return (b * c) % a != 0

    def check_set_validity(C):
        # C is a list of ints
        # 1. Antichain: for all distinct a,b in C, a !| b
        n = len(C)
        for i in range(n):
            for j in range(i+1, n):
                if C[i] % C[j] == 0 or C[j] % C[i] == 0:
                    return False
        # 2. Product-free: for all distinct a,b,c in C, a !| b*c
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if not is_valid_cand(C[i], C[j], C[k]):
                        return False
                    if not is_valid_cand(C[j], C[i], C[k]):
                        return False
                    if not is_valid_cand(C[k], C[i], C[j]):
                        return False
        return True

    def can_add_to_set(x, C):
        """Check if x can be added to set C while maintaining validity"""
        # Antichain check
        for y in C:
            if x % y == 0 or y % x == 0:
                return False
        # Product check with pairs in C
        for i in range(len(C)):
            for j in range(i+1, len(C)):
                b = C[i]
                y = C[j]
                if (b * y) % x == 0:  # x | b*y
                    return False
                if (x * y) % b == 0:  # b | x*y
                    return False
                if (x * b) % y == 0:  # y | x*b
                    return False
        return True

    # Search for max C
    # Greedy approach first to get a good lower bound
    # Try to add elements to C one by one
    current_C = []
    # Sort U by value? Or by number of divisors?
    # Let's try sorting by value descending (larger numbers are "harder" to divide)
    U_sorted = sorted(U, reverse=True)
    
    for x in U_sorted:
        if can_add_to_set(x, current_C):
            current_C.append(x)
            
    if len(current_C) > 0:
        candidate_solution = large_primes + current_C
        if len(candidate_solution) > len(best_solution):
            best_solution = candidate_solution

    # Now try to improve with local search / backtracking if time permits
    # Given the small size, we can try a simple DFS to see if we can beat the greedy result.
    # But 90s is plenty.
    
    # Let's try to find the absolute max C using DFS with pruning.
    # Since U is ~100 items, and we found a C of size ~6-10, 
    # we can try to see if we can get more.
    
    # Actually, the greedy might be suboptimal.
    # Let's do a simple randomized restart or hill climbing.
    
    best_C_size = len(current_C)
    best_C = current_C[:]
    
    import random
    for _ in range(1000): # 1000 restarts
        if time.time() > deadline:
            break
        random.shuffle(U_sorted)
        c = []
        for x in U_sorted:
            if can_add_to_set(x, c):
                c.append(x)
        if len(c) > best_C_size:
            best_C_size = len(c)
            best_C = c[:]
            # Update best solution
            candidate_solution = large_primes + best_C
            if len(candidate_solution) > len(best_solution):
                best_solution = candidate_solution
    
    # Add local search: try swapping elements to potentially improve
    # This allows us to escape local optima from greedy
    def local_search_improve(C, U_pool, max_iterations=100):
        """Try to improve set C by removing one element and adding another"""
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            if time.time() > deadline:
                break
            improved = False
            iterations += 1
            C_set = set(C)
            for i, remove_elem in enumerate(C):
                # Try removing remove_elem and adding each element from U_pool
                for add_elem in U_pool:
                    if add_elem in C_set:
                        continue
                    # Check if we can swap: remove remove_elem, add add_elem
                    new_C = [x for x in C if x != remove_elem]
                    if can_add_to_set(add_elem, new_C):
                        # Verify the new set is valid
                        new_C.append(add_elem)
                        if check_set_validity(new_C):
                            C = new_C
                            improved = True
                            break
                if improved:
                    break
        return C
    
    # Run local search on the best C found
    if time.time() < deadline:
        # Use U as the pool of potential additions
        improved_C = local_search_improve(best_C, U, max_iterations=50)
        if len(improved_C) > best_C_size:
            best_C_size = len(improved_C)
            best_C = improved_C
            candidate_solution = large_primes + best_C
            if len(candidate_solution) > len(best_solution):
                best_solution = candidate_solution
    
    # Try more randomized restarts with different strategies
    if time.time() < deadline:
        # Try sorting by different criteria
        for sort_key in [lambda x: x, lambda x: -x, lambda x: len([d for d in range(2, x+1) if x % d == 0])]:
            if time.time() > deadline:
                break
            U_alt = sorted(U, key=sort_key)
            c = []
            for x in U_alt:
                if can_add_to_set(x, c):
                    c.append(x)
            if len(c) > best_C_size:
                best_C_size = len(c)
                best_C = c[:]
                candidate_solution = large_primes + best_C
                if len(candidate_solution) > len(best_solution):
                    best_solution = candidate_solution
    
    # Final check: ensure types are int and list
    result = list(best_solution)
    # Verify final result just in case
    # (Optional but safe)
    # for i in range(len(result)):
    #   for j in range(i+1, len(result)):
    #     for k in range(j+1, len(result)):
    #       if result[i] % (result[j]*result[k]) == 0: # Wait, condition is a !| b*c
    #          pass
    #       # Actually condition is a !| b*c.
    #       # My check_set_validity did (b*c) % a != 0.
    #       # This is equivalent to a !| b*c.
    #       pass
        
    return result

# EVOLVE-BLOCK-END
