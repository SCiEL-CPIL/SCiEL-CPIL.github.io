import sys
from math import factorial
from functools import lru_cache
import time

def construct():
    start_time = time.time()
    timeout = 55.0

    target = factorial(12)
    
    # Primes involved in 12!
    primes = [2, 3, 5, 7, 11]
    num_primes = len(primes)
    
    def get_prime_factors(n):
        exps = [0] * num_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                exps[i] += 1
                n //= p
        return tuple(exps)
    
    target_vec = get_prime_factors(target)
    
    # Get all divisors of target
    all_divs = []
    i = 1
    while i * i <= target:
        if target % i == 0:
            if i > 12:
                all_divs.append(i)
            j = target // i
            if j != i and j > 12:
                all_divs.append(j)
        i += 1
    
    valid_divs = sorted(all_divs)
    n_divs = len(valid_divs)
    
    if n_divs == 0:
        return [target]

    # Precompute prime factor vectors for all valid divisors
    div_vecs = [get_prime_factors(d) for d in valid_divs]
    
    def check_solution(max_idx):
        """
        Checks if a solution exists using valid_divs[0..max_idx].
        Returns the list of factors if found, else None.
        """
        candidates = valid_divs[:max_idx + 1]
        cand_vecs = div_vecs[:max_idx + 1]
        n = len(candidates)
        
        if n == 0:
            return None

        # Precompute suffix sums of prime exponents for pruning
        # suffix_sums[i][j] = sum of cand_vecs[k][j] for k from i to n-1
        suffix_sums = [[0] * num_primes for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(num_primes):
                suffix_sums[i][j] = suffix_sums[i + 1][j] + cand_vecs[i][j]
        
        # Check if full target is formable
        # If suffix_sums[0][j] < target_vec[j] for any j, impossible
        for j in range(num_primes):
            if suffix_sums[0][j] < target_vec[j]:
                return None

        @lru_cache(maxsize=None)
        def dfs(i, rem_vec, count):
            # rem_vec is a tuple of remaining exponents
            # Check if we have satisfied the target
            if all(x == 0 for x in rem_vec):
                return True
            
            if i >= n or count >= 12:
                return False
            
            # Pruning: Can we form rem_vec with remaining candidates?
            # If the sum of available exponents is less than required, fail
            for j in range(num_primes):
                if suffix_sums[i][j] < rem_vec[j]:
                    return False
            
            # Option 1: Skip candidates[i]
            if dfs(i + 1, rem_vec, count):
                return True
            
            # Option 2: Take candidates[i]
            # Check if candidates[i] can be subtracted from rem_vec
            cv = cand_vecs[i]
            can_take = True
            new_rem = [0] * num_primes
            for j in range(num_primes):
                if rem_vec[j] < cv[j]:
                    can_take = False
                    break
                new_rem[j] = rem_vec[j] - cv[j]
            
            if can_take:
                if dfs(i + 1, tuple(new_rem), count + 1):
                    return True
            
            return False

        if dfs(0, target_vec, 0):
            # Reconstruct
            result = []
            i = 0
            rem_vec = target_vec
            count = 0
            while not all(x == 0 for x in rem_vec):
                if i >= n:
                    break
                cv = cand_vecs[i]
                
                # Check if we can solve by skipping i
                # We re-evaluate the cache
                skip_possible = dfs(i + 1, rem_vec, count)
                
                if skip_possible:
                    i += 1
                else:
                    # Must take i
                    # Verify it's valid
                    can_take = True
                    new_rem = [0] * num_primes
                    for j in range(num_primes):
                        if rem_vec[j] < cv[j]:
                            can_take = False
                            break
                        new_rem[j] = rem_vec[j] - cv[j]
                    
                    if can_take and dfs(i + 1, tuple(new_rem), count + 1):
                        result.append(candidates[i])
                        rem_vec = tuple(new_rem)
                        count += 1
                        i += 1
                    else:
                        # Should not happen if dfs returned True
                        break
            
            return result
        else:
            return None

    # Binary search for minimal max_idx
    low = 0
    high = n_divs - 1
    best_result = None
    
    while low <= high:
        if time.time() - start_time > timeout:
            break
            
        mid = (low + high) // 2
        sol = check_solution(mid)
        
        if sol is not None:
            best_result = sol
            high = mid - 1
        else:
            low = mid + 1
            
    if best_result is not None:
        return best_result
        
    return [target]
