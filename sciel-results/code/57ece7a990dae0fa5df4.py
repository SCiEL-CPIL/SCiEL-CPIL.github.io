import time
from math import factorial
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    N = 479001600  # 12!
    
    # Prime factorization of 12!
    # 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    target_exp = [10, 5, 2, 1, 1]
    
    # Generate all divisors of N
    # Divisor is defined by exponents e_i in [0, target_exp[i]]
    divisors = []
    
    def gen_divisors(idx, current_exp, current_val):
        if idx == len(primes):
            if current_val > 12:
                divisors.append((current_val, tuple(current_exp)))
            return
        for e in range(target_exp[idx] + 1):
            current_exp.append(e)
            gen_divisors(idx + 1, current_exp, current_val * (primes[idx] ** e))
            current_exp.pop()
            
    gen_divisors(0, [], 1)
    
    # Sort divisors by value
    divisors.sort(key=lambda x: x[0])
    
    n_divs = len(divisors)
    
    # We want to find the smallest M such that there exists a subset of distinct divisors
    # all <= M, whose product is N.
    
    # Precompute suffix sums of exponents for pruning
    # suffix_exp_sum[i] = sum of exponents of divisors[i:]
    # This helps prune if the remaining divisors can't cover the remaining exponents.
    suffix_exp_sum = [None] * (n_divs + 1)
    suffix_exp_sum[n_divs] = [0] * 5
    for i in range(n_divs - 1, -1, -1):
        prev = suffix_exp_sum[i+1]
        curr_exp = divisors[i][1]
        suffix_exp_sum[i] = [prev[j] + curr_exp[j] for j in range(5)]
        
    @lru_cache(maxsize=None)
    def can_complete(rem_exp_tuple, start_idx):
        """
        Returns True if we can form rem_exp_tuple using divisors[start_idx:]
        """
        rem_exp = list(rem_exp_tuple)
        
        # Check if already complete
        if all(e == 0 for e in rem_exp):
            return True
            
        # Pruning: if remaining divisors can't provide enough exponents
        suffix = suffix_exp_sum[start_idx]
        for j in range(5):
            if suffix[j] < rem_exp[j]:
                return False
                
        # Try picking a divisor from start_idx onwards
        for i in range(start_idx, n_divs):
            val, exp = divisors[i]
            # Check if this divisor fits
            fits = True
            for j in range(5):
                if exp[j] > rem_exp[j]:
                    fits = False
                    break
            if fits:
                new_rem = tuple(rem_exp[j] - exp[j] for j in range(5))
                if can_complete(new_rem, i + 1):
                    return True
        return False

    # Binary search on the maximum allowed divisor value
    # The maximum value must be one of the divisor values.
    # We search for the smallest index k such that can_complete(target, 0) is True
    # using only divisors[0..k].
    # Note: can_complete as defined above uses ALL divisors from start_idx to end.
    # We need to restrict to divisors <= M.
    # So we pass a limit index.
    
    # Let's modify can_complete to take a limit_idx
    # But lru_cache with limit_idx changes the state space significantly?
    # State: (rem_exp, start_idx, limit_idx).
    # rem_exp: 792 states.
    # start_idx: 0..792.
    # limit_idx: 0..792.
    # Total states: 792 * 792 * 792 is too big.
    
    # Alternative:
    # Filter the list of divisors to those <= M.
    # Then run the DFS on this filtered list.
    # Since we binary search M, we do this ~10 times.
    # The filtered list size varies.
    
    def check_validity(max_val):
        """
        Check if a valid set of distinct divisors <= max_val exists.
        Returns the set of factors if found, else None.
        """
        # Filter divisors
        valid_divs = [d for d in divisors if d[0] <= max_val]
        if not valid_divs:
            return None
            
        n_v = len(valid_divs)
        
        # Recompute suffix sums for this subset
        suf = [None] * (n_v + 1)
        suf[n_v] = [0] * 5
        for i in range(n_v - 1, -1, -1):
            prev = suf[i+1]
            curr_exp = valid_divs[i][1]
            suf[i] = [prev[j] + curr_exp[j] for j in range(5)]
            
        # Check if total exponents in valid_divs can cover target
        for j in range(5):
            if suf[0][j] < target_exp[j]:
                return None

        # DFS with memoization
        # State: (rem_exp_tuple, start_idx)
        # We use a dictionary for memo to allow different valid_divs lists
        memo = {}
        
        def dfs(rem_exp_tuple, start_idx):
            key = (rem_exp_tuple, start_idx)
            if key in memo:
                return memo[key]
                
            rem_exp = list(rem_exp_tuple)
            if all(e == 0 for e in rem_exp):
                memo[key] = [] # Empty list of factors needed
                return []
                
            # Pruning
            if start_idx >= n_v:
                memo[key] = None
                return None
                
            suffix = suf[start_idx]
            for j in range(5):
                if suffix[j] < rem_exp[j]:
                    memo[key] = None
                    return None
                    
            for i in range(start_idx, n_v):
                val, exp = valid_divs[i]
                fits = True
                for j in range(5):
                    if exp[j] > rem_exp[j]:
                        fits = False
                        break
                if fits:
                    new_rem = tuple(rem_exp[j] - exp[j] for j in range(5))
                    res = dfs(new_rem, i + 1)
                    if res is not None:
                        # Prepend current factor
                        result = [val] + res
                        memo[key] = result
                        return result
                        
            memo[key] = None
            return None
            
        return dfs(tuple(target_exp), 0)

    # Binary search on M
    # unique_values are the sorted divisor values
    unique_vals = sorted(set(d[0] for d in divisors))
    
    best_result = None
    low = 0
    high = len(unique_vals) - 1
    
    # We must ensure at least one factor is found.
    # The trivial solution is [N]. N is in unique_vals.
    
    while low <= high:
        if time.time() > deadline:
            break
            
        mid = (low + high) // 2
        M = unique_vals[mid]
        
        res = check_validity(M)
        
        if res is not None:
            best_result = res
            high = mid - 1
        else:
            low = mid + 1
            
    if best_result is None:
        # Fallback to trivial solution
        best_result = [N]
        
    return best_result
