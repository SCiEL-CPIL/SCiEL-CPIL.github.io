"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import factorial, gcd
    from itertools import product
    
    N = 12
    target = factorial(N)
    
    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # We represent each divisor by its exponent vector (e2, e3, e5, e7, e11)
    # Target vector is (10, 5, 2, 1, 1)
    
    # Generate all divisors > 12 with their exponent vectors
    divisors = []
    
    # Generate all combinations of exponents
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    for e11 in range(2):
                        val = (2 ** e2) * (3 ** e3) * (5 ** e5) * (7 ** e7) * (11 ** e11)
                        if val > N:
                            divisors.append((val, (e2, e3, e5, e7, e11)))
    
    # Sort divisors by value
    divisors.sort(key=lambda x: x[0])
    
    # We want to find a subset of distinct divisors whose product is exactly target.
    # In terms of exponent vectors, we want the sum of vectors to equal (10, 5, 2, 1, 1).
    # We minimize the maximum value in the subset.
    
    # Strategy: Iterate through possible maximum values M in increasing order.
    # For each M, check if there exists a subset of divisors with values <= M 
    # such that the sum of their exponent vectors equals the target vector.
    
    target_vec = (10, 5, 2, 1, 1)
    n_divs = len(divisors)
    
    # Precompute prefix sums of exponent vectors? No, that's not how subset sum works.
    # We need a subset sum search in 5D.
    
    # Since the dimension is small (5) and the target values are small (max 10),
    # we can use a dynamic programming approach or a memoized DFS.
    
    # Let's use a recursive search with memoization.
    # State: (index, remaining_vec)
    # remaining_vec is a tuple of 5 ints.
    
    # However, the index can go up to ~700, and the remaining_vec space is 11*6*3*2*2 = 792.
    # Total states: 700 * 792 ~ 550,000. This is manageable.
    
    # But we need to find *a* subset, not just count them.
    # And we need to minimize the max.
    
    # Let's iterate M. For each M, we restrict to divisors with value <= M.
    # Then we run a DFS/DP to see if we can reach the target vector.
    
    # To speed up, we can precompute for each M the set of available divisors.
    # But M changes.
    
    # Alternative: 
    # 1. Sort divisors by value.
    # 2. For k from 1 to n_divs:
    #    - Consider the first k divisors (smallest k values).
    #    - Check if there is a subset of these k divisors whose vector sum is target_vec.
    #    - If yes, the answer is divisors[k-1][0] (the k-th smallest value, which is the max in this subset).
    #      Wait, the max in the subset is at most divisors[k-1][0]. 
    #      Actually, if we find a subset using only the first k divisors, the max is at most the k-th value.
    #      But the optimal max might be smaller if the subset doesn't include the k-th divisor.
    #      However, we are iterating k. The first k for which a solution exists gives the minimal max 
    #      that is equal to the k-th divisor value? 
    #      Not necessarily. The max of the subset could be divisors[j][0] where j < k-1.
    #      But if a solution exists using divisors[0..j], it would have been found at step j+1.
    #      So the first k where a solution exists using divisors[0..k-1] implies that the minimal max is divisors[k-1][0]?
    #      No. It implies that there is a solution with max <= divisors[k-1][0].
    #      Since we check in increasing order of the upper bound, the first time we find a solution,
    #      the upper bound is the minimal possible maximum value that is a divisor in our list.
    #      But what if the minimal max is not a divisor in our list? 
    #      The max must be one of the selected divisors, so it must be in the list.
    #      So iterating through the sorted list of divisors as the upper bound M is correct.
    
    # Let's implement the check for a given set of divisors (first k).
    
    def can_form_target(div_list):
        """
        Check if there is a subset of div_list whose vector sum is target_vec.
        Returns the subset if found, else None.
        div_list is a list of (value, vec) tuples.
        """
        # Memoization: set of (index, remaining_vec)
        # To save memory, we can use a dict or a set.
        # Since we only need to know if it's possible, and then reconstruct, 
        # we can store the choice (include or not) in a dict.
        
        # Actually, let's just do DFS with memoization of failures.
        # state: (idx, rem_vec) -> bool (if we can reach target from here)
        
        memo = {}
        
        def dfs(idx, rem_vec):
            if rem_vec == (0, 0, 0, 0, 0):
                return True
            if idx >= len(div_list):
                return False
            if (idx, rem_vec) in memo:
                return memo[(idx, rem_vec)]
            
            # Option 1: Skip div_list[idx]
            if dfs(idx + 1, rem_vec):
                memo[(idx, rem_vec)] = True
                return True
            
            # Option 2: Include div_list[idx]
            val, vec = div_list[idx]
            new_rem = tuple(rem_vec[i] - vec[i] for i in range(5))
            # Check if new_rem has any negative components
            if any(x < 0 for x in new_rem):
                memo[(idx, rem_vec)] = False
                return False
            
            if dfs(idx + 1, new_rem):
                memo[(idx, rem_vec)] = True
                return True
            
            memo[(idx, rem_vec)] = False
            return False
        
        return dfs(0, target_vec)
    
    # We need to reconstruct the subset too.
    # Let's modify the function to return the subset.
    
    def find_subset(div_list):
        """
        Find a subset of div_list whose vector sum is target_vec.
        Returns list of values or None.
        """
        memo = {}
        
        def dfs(idx, rem_vec, path):
            if rem_vec == (0, 0, 0, 0, 0):
                return list(path)
            if idx >= len(div_list):
                return None
            if (idx, rem_vec) in memo:
                return None
            
            # Option 1: Skip div_list[idx]
            res = dfs(idx + 1, rem_vec, path)
            if res is not None:
                return res
            
            # Option 2: Include div_list[idx]
            val, vec = div_list[idx]
            new_rem = tuple(rem_vec[i] - vec[i] for i in range(5))
            if any(x < 0 for x in new_rem):
                memo[(idx, rem_vec)] = True
                return None
            
            res = dfs(idx + 1, new_rem, path + [val])
            if res is not None:
                return res
            
            memo[(idx, rem_vec)] = True
            return None
        
        return dfs(0, target_vec, [])
    
    # Iterate through divisors as potential max
    for k in range(1, n_divs + 1):
        # Check if a subset exists using divisors[0:k]
        subset = find_subset(divisors[:k])
        if subset is not None:
            return subset
    
    # Fallback
    return [target]

# EVOLVE-BLOCK-END
