import time
import sys

def construct():
    from math import factorial
    from itertools import product
    
    target_val = factorial(12)
    
    # Prime factors of 12!
    # 2, 3, 5, 7, 11
    primes = [2, 3, 5, 7, 11]
    
    def get_prime_vector(n):
        vec = [0] * len(primes)
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        if n != 1:
            return None # Contains prime factor > 11
        return tuple(vec)

    target_vec = get_prime_vector(target_val)
    
    # Generate candidates
    # We search for min max factor.
    # Max factor likely < 100.
    max_limit = 200
    
    candidates = []
    for i in range(13, max_limit + 1):
        vec = get_prime_vector(i)
        if vec is not None:
            candidates.append((i, vec))
            
    # Sort by value
    candidates.sort(key=lambda x: x[0])
    
    best_solution = None
    best_max_factor = float('inf')
    
    start_time = time.time()
    
    # Iterate possible max factor M
    # We can just iterate through the sorted candidates and treat them as "available"
    # But we want to minimize the largest.
    # So we iterate M from 13 to max_limit.
    # For each M, we check if a solution exists using only candidates <= M.
    
    # To speed up, we can pre-filter or just run DFS.
    # Given small size, simple loop is fine.
    
    for idx, (val, vec) in enumerate(candidates):
        M = val
        # Check deadline
        if time.time() - start_time > 50:
            break
            
        # Run DFS on candidates[0:idx+1]
        # State: (current_index, current_vec)
        # We want to find a subset of candidates[0:idx+1] that sums to target_vec.
        
        def dfs(start, curr_vec):
            if any(curr_vec[i] > target_vec[i] for i in range(len(primes))):
                return None
            if start == idx + 1:
                if all(curr_vec[i] == target_vec[i] for i in range(len(primes))):
                    return []
                return None
            
            # Option 1: Skip candidate at start
            res = dfs(start + 1, curr_vec)
            if res is not None:
                return res
                
            # Option 2: Take candidate at start
            new_vec = tuple(curr_vec[i] + candidates[start][1][i] for i in range(len(primes)))
            # Prune
            if all(new_vec[i] <= target_vec[i] for i in range(len(primes))):
                res = dfs(start + 1, new_vec)
                if res is not None:
                    return [candidates[start][0]] + res
            return None

        sol = dfs(0, tuple([0]*len(primes)))
        if sol:
            best_solution = sol
            best_max_factor = M
            break # Found min max factor
            
    if best_solution:
        return best_solution
    else:
        # Fallback
        return [target_val]
